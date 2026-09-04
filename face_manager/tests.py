"""face_manager tests.

The previous version of this file (still visible in git history) tested
against `populateFromImage`/`placeInDatabase`/`establish_server_connection`
and the `image_face_extractor` submodule's client-server dlib pipeline.
None of those functions exist anymore -- the app was rewritten around
insightface (`face_extract_encode.FaceExtractor`, `pyramidal_detector.
PyramidalDetector`) and this file was never updated to match, so it could
not even be collected by the test runner. This is a full rewrite against
the current pipeline.

Tests that run real insightface inference are tagged 'slow' and go through
test_face_cache.cached_detect so repeat runs against an unchanged image +
unchanged pyramidal_detector.py source don't pay CPU inference cost again.
Run just the fast ones with:
    manage.py test face_manager --exclude-tag=slow
"""
import os
import unittest
import zlib
from io import BytesIO

import cv2
import numpy as np
import psycopg2
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.db import connection
from django.test import TestCase, override_settings, tag

from django.core.management import call_command

from unittest.mock import patch, MagicMock

from face_manager.assign_faces import faceAssigner
from face_manager.face_extract_encode import FaceExtractor
from face_manager.models import Face, Person, get_default_blank_person, clear_confirmed_ignore_face_encodings
from face_manager.pyramidal_detector import PyramidalDetector
from face_manager.test_face_cache import cached_detect
from face_manager.verification_clustering import cluster_all_unverified_faces
from filepopulator.models import Directory, DuplicateFile, ImageFile
from filepopulator.scripts import create_image_file


def _tiny_jpeg_bytes(size=(50, 50)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return BytesIO(buf).read()


def _embedding(base_idx, seed, dim=512, noise_scale=0.01):
    """A synthetic 512-d embedding clustered around one of several
    near-orthogonal base directions (one per base_idx), with a small
    amount of per-face noise. At noise_scale=0.01 same-base_idx vectors
    land around cos~0.95 (comfortably above the 0.7 default cluster
    threshold) while different-base_idx vectors land near cos~0 -- lets
    tests build "these faces should cluster together" / "these shouldn't"
    fixtures without depending on real face data."""
    vec = np.zeros(dim)
    vec[base_idx] = 1.0
    rng = np.random.RandomState(seed)
    vec = vec + rng.normal(scale=noise_scale, size=dim)
    return vec


def make_person(name):
    p = Person.objects.create(person_name=name)
    p.highlight_img.save(f"{name}.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
    return p


def make_image(relative_fixture="naming/good/1.JPG"):
    path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/{relative_fixture}"
    create_image_file(path)
    return ImageFile.objects.get(filename=path)


def make_preexisting_image_row(path):
    """Insert an ImageFile row directly via bulk_create, bypassing
    ImageFile.save() entirely.

    ImageFile.save() unconditionally calls _generate_md5_hash(), which
    decodes the full image -- so it's not just ingestion of a *new*
    corrupt file that's a problem (that's caught by add_from_root_dir's
    per-file try/except and simply never creates a row), it's that ANY
    ImageFile.save() call on a row whose file has since become corrupt on
    disk (e.g. bit rot, an interrupted sync) will also raise. That's the
    real-world scenario for the 5 fixture files here: they were ingested
    fine when healthy, and only failed later once the underlying file
    was damaged. bulk_create is the only way to get such a row into the
    test DB without immediately reproducing that crash ourselves.
    """
    directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
    image = ImageFile(
        filename=path,
        directory=directory,
        thumbnail_big="",
        thumbnail_medium="",
        thumbnail_small="",
        width=100,
        height=100,
        isProcessed=False,
    )
    ImageFile.objects.bulk_create([image])
    return ImageFile.objects.get(filename=path)


def make_face(image_file, declared_name=None, **overrides):
    if declared_name is None:
        declared_name = make_person("Test Person")
    w, h = image_file.width, image_file.height
    box = dict(box_left=1, box_top=1, box_right=min(40, w - 1), box_bottom=min(40, h - 1))
    box.update({k: v for k, v in overrides.items() if k in box})
    remaining = {k: v for k, v in overrides.items() if k not in box}

    face = Face(declared_name=declared_name, source_image_file=image_file, **box)
    for k, v in remaining.items():
        setattr(face, k, v)
    face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
    face.save()
    return face


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class PersonModelTests(TestCase):
    def test_blank_sentinel_cannot_be_deleted(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        blank.delete()
        self.assertTrue(Person.objects.filter(person_name=settings.BLANK_FACE_NAME).exists())

    def test_deleting_person_removes_highlight_file(self):
        person = make_person("Deletable")
        highlight_path = person.highlight_img.path
        self.assertTrue(os.path.exists(highlight_path))
        person.delete()
        self.assertFalse(os.path.exists(highlight_path))

    def test_get_default_blank_person_returns_existing_sentinel(self):
        # Covers face_manager/models.py get_default_blank_person(). Its
        # "person doesn't exist yet" fallback branch is unreachable/broken
        # (references an undefined `sq_thumb` and uses BytesIO/ContentFile
        # without importing them) -- not exercised here since the sentinel
        # is expected to always exist; see NOTES on bootstrap gap.
        blank = get_default_blank_person()
        self.assertEqual(blank.person_name, settings.BLANK_FACE_NAME)

    def test_increment_decrement_counters(self):
        person = make_person("Counted")
        person.increment_assigned()
        person.increment_unverified()
        person.increment_possible_num()
        person.refresh_from_db()
        self.assertEqual(person.num_faces, 1)
        self.assertEqual(person.num_unverified_faces, 1)
        self.assertEqual(person.num_possibilities, 1)

        person.decrement_assigned()
        person.decrement_unverified()
        person.decrement_possible_num()
        person.refresh_from_db()
        self.assertEqual(person.num_faces, 0)
        self.assertEqual(person.num_unverified_faces, 0)
        self.assertEqual(person.num_possibilities, 0)

    def test_decrement_below_zero_clamps_to_zero(self):
        person = make_person("Clamped")
        person.decrement_assigned()
        person.decrement_unverified()
        person.decrement_possible_num()
        person.refresh_from_db()
        self.assertEqual(person.num_faces, 0)
        self.assertEqual(person.num_unverified_faces, 0)
        self.assertEqual(person.num_possibilities, 0)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class FaceModelTests(TestCase):
    def setUp(self):
        self.image = make_image()

    def test_face_save_rejects_inverted_box(self):
        person = make_person("Boxy")
        face = Face(
            declared_name=person,
            source_image_file=self.image,
            box_left=10,
            box_top=10,
            box_right=5,  # right < left
            box_bottom=20,
        )
        face.face_thumbnail.save("t.jpg", ContentFile(_tiny_jpeg_bytes()), save=False)
        with self.assertRaises(ValidationError):
            face.save()

    def test_face_save_rejects_missing_thumbnail_file(self):
        person = make_person("NoThumb")
        face = Face(
            declared_name=person,
            source_image_file=self.image,
            box_left=1,
            box_top=1,
            box_right=20,
            box_bottom=20,
        )
        # face_thumbnail is left unset -> face.face_thumbnail.file access
        # in Face.save() should fail validation rather than silently save.
        with self.assertRaises(Exception):
            face.save()

    def test_face_delete_removes_thumbnail_file(self):
        face = make_face(self.image)
        thumb_path = face.face_thumbnail.path
        self.assertTrue(os.path.exists(thumb_path))
        face.delete()
        self.assertFalse(os.path.exists(thumb_path))

    def test_face_delete_tolerates_already_missing_thumbnail_file(self):
        # Regression test for a fixed bug: delete() unconditionally called
        # os.remove() with no guard for the file already being gone (e.g.
        # a duplicate delete against a restored DB snapshot whose rows
        # point at the same shared media path an earlier/live run already
        # cleaned up) -- used to raise FileNotFoundError instead of still
        # removing the row.
        face = make_face(self.image)
        os.remove(face.face_thumbnail.path)
        face.delete()
        self.assertFalse(Face.objects.filter(pk=face.pk).exists())

    def test_deleting_image_cascades_to_faces(self):
        face = make_face(self.image)
        face_id = face.id
        thumb_path = face.face_thumbnail.path
        self.image.delete()
        self.assertFalse(Face.objects.filter(id=face_id).exists())
        # Regression test for a fixed bug: Face.source_image_file's
        # on_delete=CASCADE means Django's cascade-delete collector used
        # to remove cascaded Face rows via a bulk SQL DELETE, which does
        # NOT call each instance's overridden delete() -- so the DB row
        # above was always cleaned up correctly, but this thumbnail file
        # was silently left orphaned on disk. ImageFile.delete() now
        # explicitly deletes each related Face first (invoking Face's own
        # delete() override) before deleting itself.
        self.assertFalse(os.path.exists(thumb_path))

    def test_associate_person_updates_declared_name_and_weight(self):
        original = make_person("Original")
        target = make_person("Target")
        face = make_face(self.image, declared_name=original)
        face.set_possible_person(target.id, 1, 0.9)

        face.associate_person(target.id)
        face.refresh_from_db()

        self.assertEqual(face.declared_name_id, target.id)
        self.assertEqual(face.weight_1, 0.0)
        self.assertFalse(face.validated)

    def test_associate_person_clears_possible_identity(self):
        # Regression test for a fixed bug: remove_poss_ident() used to clear
        # the FK by poking `self.__dict__['poss_identN_id'] = None` directly
        # instead of `self.poss_identN = None`, so Model.save()'s FK-cache
        # reconciliation silently restored the old value. Now uses real
        # setattr()/getattr(), matching how reject_association() always did.
        original = make_person("Original2")
        target = make_person("Target2")
        face = make_face(self.image, declared_name=original)
        face.set_possible_person(target.id, 1, 0.9)

        face.associate_person(target.id)
        face.refresh_from_db()

        self.assertIsNone(face.poss_ident1)

    def test_verify_person_in_image(self):
        person = make_person("Verifiable")
        person.increment_unverified()
        face = make_face(self.image, declared_name=person)
        face.verify_person_in_image()
        face.refresh_from_db()
        self.assertTrue(face.validated)

    def test_reset_to_pool_returns_face_to_blank_sentinel(self):
        # reset_to_pool() must land the face on the blank sentinel Person,
        # NOT NULL -- a NULL declared_name is invisible to the "Unassigned"
        # bucket and to assign_faces.py's re-classification, both of which
        # filter declared_name__person_name == settings.BLANK_FACE_NAME.
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        owner = make_person("PrevOwner")
        owner.increment_assigned()
        guesser = make_person("Guesser")
        face = make_face(self.image, declared_name=owner)
        face.set_possible_person(guesser.id, 1, 0.8)

        face.reset_to_pool()
        face.refresh_from_db()
        owner.refresh_from_db()

        self.assertEqual(face.declared_name_id, blank.id)
        self.assertIsNotNone(face.declared_name)
        self.assertFalse(face.validated)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(owner.num_faces, 0)

    def test_reject_association_removes_from_possibles(self):
        person = make_person("Rejectable")
        face = make_face(self.image)
        face.set_possible_person(person.id, 1, 0.75)
        face.reject_association(person.id)
        face.refresh_from_db()
        self.assertIsNone(face.poss_ident1)
        self.assertIn(person.id, face.rejected_fields)

    def test_add_to_rejected_fields_appends_and_dedupes_without_saving(self):
        face = make_face(self.image)
        self.assertIsNone(face.rejected_fields)

        face.add_to_rejected_fields(101)
        self.assertEqual(face.rejected_fields, [101])
        # Not saved yet - a caller doing other unsaved changes in the same
        # operation (e.g. close_assigned's "Remove from person" case) can
        # rely on a single later save() persisting both.
        face.refresh_from_db()
        self.assertIsNone(face.rejected_fields)

        face.add_to_rejected_fields(101)
        face.add_to_rejected_fields(202)
        face.add_to_rejected_fields(101)  # duplicate
        self.assertEqual(sorted(face.rejected_fields), [101, 202])

    def test_face_encoding_512_stores_at_float32_precision(self):
        """face_encoding_512 is now backed by `real` (single precision),
        not `double precision` -- insightface's embeddings are natively
        float32 (verified directly against real detections), so storing
        at float32 precision loses nothing the model didn't already lose.
        A value with real double-precision-only bits should get rounded
        to its nearest float32 representation on save, not preserved
        exactly."""
        double_only_value = 0.123456789012345  # not exactly representable in float32
        face = make_face(self.image)
        face.face_encoding_512 = [double_only_value] * 512
        face.save()
        face.refresh_from_db()

        expected = float(np.float32(double_only_value))
        self.assertEqual(face.face_encoding_512[0], expected)
        self.assertNotEqual(face.face_encoding_512[0], double_only_value)


@tag("slow")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class PyramidalDetectorRealInferenceTests(TestCase):
    """Runs real insightface inference against known fixture photos.
    Uses test_face_cache so repeat runs (unchanged image + unchanged
    pyramidal_detector.py) skip the actual CPU inference."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.detector = PyramidalDetector(iou_thresh=0.3)

    def test_detects_faces_in_known_multi_face_fixture(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        faces = cached_detect(self.detector, path)
        self.assertGreaterEqual(len(faces), 1)
        for f in faces:
            self.assertEqual(len(f["embedding"]), 512)

    def test_detects_faces_in_second_known_fixture(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_same_faces.jpg"
        faces = cached_detect(self.detector, path)
        self.assertGreaterEqual(len(faces), 1)


@tag("slow")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class FaceExtractorCorruptedImageTests(TestCase):
    """Characterizes current behavior against the 5 real corrupted JPEGs
    pulled from production logs (see
    /mnt/fast_storage/appdata/django_picasa/test_suite/corrupted_images/NOTES.md).
    Loads the real FaceAnalysis model (not mocked, not cached -- these
    fail before reaching detection, so there's no inference cost)."""

    CORRUPTED_DIR = "/photos/corrupted"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extractor = FaceExtractor()

    def test_corrupted_images_get_marked_processed_and_flagged(self):
        # Regression test for a fixed bug: the `except Exception: ...
        # continue` branch that catches the PIL "image file is truncated" /
        # "broken data stream" errors used to never set
        # img_obj.isProcessed = True, so these files were retried by the
        # scheduled face_extraction task on every run, forever. Now marks
        # isProcessed (stop retrying) and image_load_failed/
        # image_load_error (so the failure is recorded instead of just
        # silently no-op'd -- see the planned frontend "failed to open"
        # list in CLAUDE.md). Uses ImageFile.objects.filter(...).update()
        # rather than img_obj.save(), since save() would itself re-decode
        # the (still corrupted) image via _generate_md5_hash() and raise.
        corrupted_files = sorted(os.listdir(self.CORRUPTED_DIR))
        self.assertEqual(len(corrupted_files), 5)

        image_objs = []
        for fname in corrupted_files:
            path = os.path.join(self.CORRUPTED_DIR, fname)
            image_objs.append(make_preexisting_image_row(path))

        self.extractor.find_and_encode_faces()

        for img_obj in image_objs:
            img_obj.refresh_from_db()
            self.assertTrue(
                img_obj.isProcessed,
                f"{img_obj.filename} was not marked processed -- would be "
                "retried by face_extraction on every future run.",
            )
            self.assertTrue(img_obj.image_load_failed)
            self.assertTrue(img_obj.image_load_error)
            self.assertEqual(Face.objects.filter(source_image_file=img_obj).count(), 0)

    def test_good_image_gets_processed_and_gains_faces(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)
        self.assertFalse(img_obj.isProcessed)

        self.extractor.find_and_encode_faces()

        img_obj.refresh_from_db()
        self.assertTrue(img_obj.isProcessed)
        self.assertGreaterEqual(Face.objects.filter(source_image_file=img_obj).count(), 1)

    def test_new_face_gains_kps_from_real_detection(self):
        """add_new_face() should now also store the 5-point landmarks
        InsightFace's detector produces, alongside the embedding it
        already stored -- needed so a face's embedding can later be
        exactly reproduced (see FaceExtractor.reencode_missing_faces())
        without re-detecting from scratch."""
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)

        self.extractor.find_and_encode_faces()

        faces = list(Face.objects.filter(source_image_file=img_obj))
        self.assertGreaterEqual(len(faces), 1)
        for f in faces:
            self.assertIsNotNone(f.kps, f"Face {f.id} has no kps after fresh detection")
            self.assertEqual(len(f.kps), 10)

    def test_rematched_existing_face_gains_kps_from_update_path(self):
        """update_existing_face_to_insightface() (the IOU-rematch branch,
        hit when find_and_encode_faces() runs again against an image that
        already has Face rows) should also populate kps, not just
        add_new_face()'s fresh-detection branch."""
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)

        self.extractor.find_and_encode_faces()
        first_pass_ids = set(Face.objects.filter(source_image_file=img_obj).values_list('id', flat=True))
        self.assertGreaterEqual(len(first_pass_ids), 1)

        # Clear kps to simulate a face that predates this field, then
        # re-run detection against the same (already-processed) image by
        # resetting isProcessed -- this should hit the existing-face
        # IOU-rematch path, not add_new_face(), since Face rows already
        # exist for this image.
        Face.objects.filter(id__in=first_pass_ids).update(kps=None)
        img_obj.isProcessed = False
        img_obj.save()

        self.extractor.find_and_encode_faces()

        second_pass_faces = Face.objects.filter(source_image_file=img_obj)
        self.assertEqual(
            set(second_pass_faces.values_list('id', flat=True)), first_pass_ids,
            "Re-running detection against an already-processed image should rematch "
            "existing faces, not create new ones."
        )
        for f in second_pass_faces:
            self.assertIsNotNone(f.kps, f"Face {f.id} has no kps after IOU-rematch")
            self.assertEqual(len(f.kps), 10)

    def test_failure_on_one_image_does_not_block_others(self):
        """Regression test: the IOU-matching logic in
        find_and_encode_faces() (everything after image load/detection --
        computing IOU, matching existing faces to detections, and the
        NotImplementedError/ValueError/bare-assert paths the original
        author left for cases believed unreachable) used to have no
        exception handling of its own. Any failure there -- one of those
        specific cases, or any future unforeseen edge case -- propagated
        out of the entire per-image loop and up into face_manager.tasks.
        process_faces()'s outer bare except, silently aborting the
        *entire* scheduled run: every other already-queued image would
        never get attempted either. Now wrapped so a failure on one image
        just skips it (left isProcessed=False to retry later) instead of
        blocking every other image in the batch."""
        path_a = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        path_b = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_same_faces.jpg"
        create_image_file(path_a)
        create_image_file(path_b)
        img_a = ImageFile.objects.get(filename=path_a)
        img_b = ImageFile.objects.get(filename=path_b)

        original_add_new_face = FaceExtractor.add_new_face

        def flaky_add_new_face(self, insight_detected_face, img_obj, img_numpy):
            if img_obj.pk == img_a.pk:
                raise RuntimeError("simulated IOU-matching crash")
            return original_add_new_face(self, insight_detected_face, img_obj, img_numpy)

        with patch.object(FaceExtractor, "add_new_face", flaky_add_new_face):
            self.extractor.find_and_encode_faces()

        img_a.refresh_from_db()
        img_b.refresh_from_db()
        self.assertFalse(img_a.isProcessed)
        self.assertTrue(img_b.isProcessed)

    def test_skips_entirely_when_advisory_lock_already_held(self):
        # Regression test for a real production bug: find_and_encode_faces()
        # used to have no locking of its own at all -- only the Celery task
        # wrapper checked celery_app.control.inspect().active(), a
        # check-then-act race that also didn't cover a direct call like
        # this one. Two concurrent runs against the same never-before-
        # processed image each saw n_existing=0 and both inserted a Face
        # row for the same detected face -- confirmed against real
        # production data (~72% of found same-image duplicate-Face-row
        # pairs had adjacent/near-adjacent ids). Now find_and_encode_faces()
        # holds a Postgres advisory lock (common/advisory_lock.py) for its
        # whole run; while another session holds it, this call must do
        # nothing at all, not even mark the image processed.
        #
        # Advisory locks are reentrant PER SESSION, so the "other holder"
        # has to be a genuinely separate connection, not another lock
        # acquired on Django's own (shared, single) test connection.
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)

        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            key = zlib.crc32(b'face_manager.find_and_encode_faces')
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
                self.assertTrue(cur.fetchone()[0])

                self.extractor.find_and_encode_faces()

                img_obj.refresh_from_db()
                self.assertFalse(img_obj.isProcessed)
                self.assertEqual(Face.objects.filter(source_image_file=img_obj).count(), 0)

                cur.execute("SELECT pg_advisory_unlock(%s)", [key])
        finally:
            other.close()

        # Once free again, a normal run still works.
        self.extractor.find_and_encode_faces()
        img_obj.refresh_from_db()
        self.assertTrue(img_obj.isProcessed)


class UpdateListOfNoMatchingDetectsTests(TestCase):
    """Regression test for a fixed bug: update_list_of_no_matching_detects()
    only clamped each box coordinate on one side (box_left/box_top floored
    at 0, box_right/box_bottom capped at the image's current width/height),
    so a face whose stored box came from a different coordinate space than
    the image's current dimensions -- e.g. a stale detection from before
    this session's EXIF-orientation-consolidation fix, when
    face_extract_encode's decode path could disagree with filepopulator's
    about a rotated image's true width/height -- could still end up with
    box_right <= box_left (or box_bottom <= box_top) after "clamping",
    which Face.save() correctly rejects via ValidationError. That
    propagated all the way up through find_and_encode_faces() into
    face_manager.tasks.process_faces()'s bare except:, silently aborting
    the *entire* scheduled run, not just this one face -- confirmed
    against a real production case (FastFoto_0248.jpg) before fixing."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extractor = FaceExtractor()

    def test_geometrically_invalid_face_is_deleted_not_crashed(self):
        img = make_image()
        stale_face = make_face(
            img,
            box_left=img.width + 500,
            box_top=1,
            box_right=img.width + 700,
            box_bottom=40,
        )

        self.extractor.update_list_of_no_matching_detects([stale_face])

        self.assertFalse(Face.objects.filter(pk=stale_face.pk).exists())

    def test_recoverable_out_of_bounds_face_is_clamped_and_kept(self):
        img = make_image()
        face = make_face(
            img,
            box_left=1,
            box_top=1,
            box_right=img.width + 50,  # slightly over -- should just clamp
            box_bottom=40,
        )

        self.extractor.update_list_of_no_matching_detects([face])

        face.refresh_from_db()
        self.assertEqual(face.box_right, img.width)
        self.assertTrue(face.reencoded)

    def test_box_that_only_fits_swapped_dimensions_is_deleted(self):
        """A box that doesn't fit the image's current (w, h) at all, but
        does fit the swapped (h, w) -- the signature of a face whose
        coordinates were computed against the wrong (unrotated) dimensions
        by the pre-fix decode-path disagreement -- must be deleted even
        though clamping alone wouldn't necessarily make it degenerate."""
        img = make_image()  # width=800, height=286
        self.assertGreater(img.width, img.height)
        face = make_face(
            img,
            box_left=1, box_top=1,
            box_right=200,  # fits both width interpretations
            box_bottom=500,  # > img.height (286), but < img.width (800): only fits swapped
        )

        self.extractor.update_list_of_no_matching_detects([face])

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())

    def test_unreencoded_face_on_rotated_image_is_deleted_even_if_in_bounds(self):
        """An orientation 6/8 face that has never been successfully
        re-matched since creation (reencoded=False) has never had its box
        verified under the corrected decode path -- delete it outright,
        even if its box happens to still fit the image's current
        dimensions by coincidence."""
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8)
        img.refresh_from_db()
        face = make_face(img, box_left=1, box_top=1, box_right=40, box_bottom=40)
        self.assertFalse(face.reencoded)

        self.extractor.update_list_of_no_matching_detects([face])

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())

    def test_unreencoded_face_on_unrotated_image_is_kept(self):
        """Regression guard: the orientation 6/8 + reencoded=False deletion
        must not apply to orientations where no width/height swap is
        possible -- an in-bounds, unreencoded face on an orientation-1
        image is just legitimately not yet re-matched, not stale data, and
        should be kept (clamped/marked reencoded like any other kept
        face)."""
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=1)
        img.refresh_from_db()
        face = make_face(img, box_left=1, box_top=1, box_right=40, box_bottom=40)
        self.assertFalse(face.reencoded)

        self.extractor.update_list_of_no_matching_detects([face])

        face.refresh_from_db()
        self.assertTrue(Face.objects.filter(pk=face.pk).exists())
        self.assertTrue(face.reencoded)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class CleanupChronicallyUnmatchedFacesTests(TestCase):
    """A face on an orientation 6/8 image that's reencoded=True but still
    carries the NON_DETECTED_FACE_ENCODING sentinel has never been
    geometrically verified under the corrected decode path -- since a
    wrongly-positioned box can never match a real detection, it would
    otherwise persist forever. This command deletes those outright and
    marks their images unprocessed for redetection."""

    def test_deletes_chronically_unmatched_faces_and_marks_images_unprocessed(self):
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8, isProcessed=True)
        stale_face = make_face(img, reencoded=True, face_encoding_512=list(settings.NON_DETECTED_FACE_ENCODING))

        call_command("cleanup_chronically_unmatched_faces", "--yes")

        self.assertFalse(Face.objects.filter(pk=stale_face.pk).exists())
        img.refresh_from_db()
        self.assertFalse(img.isProcessed)

    def test_leaves_verified_faces_alone(self):
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8, isProcessed=True)
        good_face = make_face(img, reencoded=True, face_encoding_512=[0.1] * 512)

        call_command("cleanup_chronically_unmatched_faces", "--yes")

        self.assertTrue(Face.objects.filter(pk=good_face.pk).exists())
        img.refresh_from_db()
        self.assertTrue(img.isProcessed)

    def test_dry_run_writes_nothing(self):
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8, isProcessed=True)
        stale_face = make_face(img, reencoded=True, face_encoding_512=list(settings.NON_DETECTED_FACE_ENCODING))

        call_command("cleanup_chronically_unmatched_faces", "--dry-run")

        self.assertTrue(Face.objects.filter(pk=stale_face.pk).exists())


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClearConfirmedIgnoreEncodingsTests(TestCase):
    """clear_confirmed_ignore_encodings clears face_encoding_512 only for
    faces CONFIRMED (declared_name) to .ignore/.realignore -- never faces
    merely SUGGESTED as ignore (poss_ident1 set, declared_name still the
    blank sentinel), and never faces declared to a real person. Face.kps
    must survive untouched either way, since it's what lets a later
    reencode_missing_faces() pass exactly recover the embedding."""

    def setUp(self):
        self.image = make_image()
        self.ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.realignore_person = Person.objects.get(person_name='.realignore')
        self.blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)

    def test_confirmed_ignore_face_gets_encoding_cleared_kps_kept(self):
        kps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        face = make_face(
            self.image, declared_name=self.ignore_person,
            face_encoding_512=[0.5] * 512, kps=kps,
        )

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)
        self.assertEqual(face.kps, kps)

    def test_confirmed_realignore_face_gets_encoding_cleared(self):
        face = make_face(self.image, declared_name=self.realignore_person, face_encoding_512=[0.5] * 512)

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)

    def test_suggested_ignore_face_is_left_alone(self):
        """poss_ident1 == ignore_person but declared_name is still the
        blank sentinel -- this is an unreviewed classifier suggestion,
        not a human confirmation. Its embedding is exactly what a human
        still needs to review that suggestion against, so it must not be
        touched."""
        face = make_face(
            self.image, declared_name=self.blank_person,
            poss_ident1=self.ignore_person, weight_1=0.9,
            face_encoding_512=[0.5] * 512,
        )

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)

    def test_face_declared_to_real_person_is_left_alone(self):
        person = make_person("Real Person")
        face = make_face(self.image, declared_name=person, face_encoding_512=[0.5] * 512)

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)

    def test_already_cleared_face_is_a_noop(self):
        face = make_face(self.image, declared_name=self.ignore_person)
        face.face_encoding_512 = None
        face.save()

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)

    def test_dry_run_writes_nothing(self):
        face = make_face(self.image, declared_name=self.ignore_person, face_encoding_512=[0.5] * 512)

        call_command("clear_confirmed_ignore_encodings", "--dry-run")

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)

    def test_shared_function_matches_command_behavior(self):
        """clear_confirmed_ignore_face_encodings() (face_manager/models.py)
        is the single source of truth the management command AND the
        ongoing face_manager.clear_ignored_encodings Celery task both call
        for the actual write -- covered directly here so a regression in
        either caller's own logic can't hide the underlying function
        breaking."""
        confirmed = make_face(self.image, declared_name=self.ignore_person, face_encoding_512=[0.5] * 512)
        suggested = make_face(
            self.image, declared_name=self.blank_person,
            poss_ident1=self.ignore_person, weight_1=0.9,
            face_encoding_512=[0.5] * 512,
        )

        updated_count = clear_confirmed_ignore_face_encodings()

        self.assertEqual(updated_count, 1)
        confirmed.refresh_from_db()
        suggested.refresh_from_db()
        self.assertIsNone(confirmed.face_encoding_512)
        self.assertEqual(suggested.face_encoding_512, [0.5] * 512)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class MergeAnotherIgnoreIntoIgnoreTests(TestCase):
    """'.another_ignore' used to be a separate sentinel Person from '.ignore'
    (see settings.SOFT_IGNORE_NAME) -- created by the assign_faces
    classifier for low-confidence auto-suggestions, but never recognized by
    api/views.py's close_ignored bulk action, which only checked for
    '.ignore'/'.realignore'. This command folds any Face rows still
    pointing at '.another_ignore' over to '.ignore' and removes it."""

    def setUp(self):
        # The --keepdb test DB can already have a permanent '.ignore'/
        # '.another_ignore' Person row left over from before IGNORED_NAMES
        # was trimmed (they aren't test-created, so TestCase's rollback
        # doesn't remove them) -- reuse whatever's there instead of always
        # creating a fresh one, or Person.objects.get() in the command
        # finds two rows with the same name.
        self.ignore_person = Person.objects.filter(person_name=".ignore").first() or make_person(".ignore")
        self.another_ignore_person = Person.objects.filter(person_name=".another_ignore").first() or make_person(".another_ignore")
        self.img = make_image()

    def test_dry_run_reports_counts_and_writes_nothing(self):
        declared_face = make_face(self.img, declared_name=self.another_ignore_person)
        poss_face = make_face(self.img, declared_name=self.ignore_person, poss_ident1=self.another_ignore_person)

        call_command("merge_another_ignore_into_ignore", "--dry-run")

        declared_face.refresh_from_db()
        poss_face.refresh_from_db()
        self.assertEqual(declared_face.declared_name, self.another_ignore_person)
        self.assertEqual(poss_face.poss_ident1, self.another_ignore_person)
        self.assertTrue(Person.objects.filter(person_name=".another_ignore").exists())

    def test_merge_reassigns_declared_name_and_poss_idents_then_deletes_sentinel(self):
        declared_face = make_face(self.img, declared_name=self.another_ignore_person)
        poss_face = make_face(self.img, declared_name=self.ignore_person, poss_ident1=self.another_ignore_person)

        call_command("merge_another_ignore_into_ignore", "--yes")

        declared_face.refresh_from_db()
        poss_face.refresh_from_db()
        self.assertEqual(declared_face.declared_name, self.ignore_person)
        self.assertEqual(poss_face.poss_ident1, self.ignore_person)
        self.assertFalse(Person.objects.filter(person_name=".another_ignore").exists())

    def test_merge_with_no_another_ignore_person_is_a_safe_noop(self):
        self.another_ignore_person.delete()
        call_command("merge_another_ignore_into_ignore", "--yes")
        self.assertTrue(Person.objects.filter(person_name=".ignore").exists())


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClassifyUnassignedArraySizingTests(TestCase):
    """Regression test for a fixed bug: classify_unassigned()'s
    metrics_array used to be sized to self.num_likely_people (the full
    candidate roster) rather than len(candidate_ids) (this face's roster
    minus whatever's already been rejected for it), leaving stale
    np.zeros() padding in the tail rows whenever a rejection had shrunk the
    candidate list. A real (negative) similarity could lose to that padded
    0 in np.argmax(), which then indexed candidate_id_arr -- sized to the
    *shrunk* list -- out of bounds, raising IndexError. Since execute()'s
    per-face try/except was commented out, that IndexError aborted the
    entire scheduled assign_faces run, not just this one face."""

    def setUp(self):
        self.assigner = faceAssigner()
        self.person_a = make_person("Candidate A")
        self.person_b = make_person("Candidate B")
        self.assigner.likely_people_ids = [self.person_a.id, self.person_b.id]
        self.assigner.num_likely_people = 2

        self.query = np.zeros(512)
        self.query[0] = 1.0

        # Person A's only reference encoding points the opposite
        # direction -- a real, negative cosine similarity, needed to make
        # the stale zero padding (0.0) look like a better match than any
        # real candidate under the old sizing bug.
        ref_a = -self.query.copy()
        self.assigner.embedding_dict = {self.person_a.id: ref_a.reshape(512, 1)}
        self.assigner.norm_dict = {self.person_a.id: np.array([1.0])}
        self.assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        self.face = make_face(img, declared_name=blank_person)
        self.face.face_encoding_512 = self.query.tolist()
        self.face.save()

    def test_shrunk_candidate_list_does_not_raise_indexerror(self):
        # Reject person_b and the ignore person -- only person_a is left
        # as a real candidate, but the array used to still be sized for
        # both likely people (self.num_likely_people == 2).
        self.face.rejected_fields = [self.person_b.id, self.assigner.ignore_person_id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, self.person_a.id)

    def test_all_candidates_rejected_assigns_to_ignore_without_crashing(self):
        # Every likely person has been rejected for this face -- candidate_ids
        # is empty. Must not attempt any similarity math (which would raise
        # ValueError on a size-0 array reduction) and must not crash.
        self.face.rejected_fields = [self.person_a.id, self.person_b.id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        mock_set.assert_called_once_with(self.assigner.ignore_person_id, 1, 1.0)


class P99ThresholdLookupTests(TestCase):
    """_p99_threshold_for_gallery_size() implements the gallery-size-
    adaptive accept/reject gate (see CLAUDE.md's "Face-classification
    outlier-rejection" section for the experiment this is calibrated
    from) -- checks the boundary edges land in the bucket they should."""

    def setUp(self):
        self.assigner = faceAssigner()

    def test_boundary_edges(self):
        t = self.assigner.BUCKET_THRESHOLDS
        cases = [
            (1, t[0]), (49, t[0]),
            (50, t[1]), (199, t[1]),
            (200, t[2]), (499, t[2]),
            (500, t[3]), (10_000, t[3]),
        ]
        for gallery_size, expected in cases:
            with self.subTest(gallery_size=gallery_size):
                self.assertEqual(self.assigner._p99_threshold_for_gallery_size(gallery_size), expected)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClassifyUnassignedBucketGateTests(TestCase):
    """Confirms the accept/reject gate actually varies by candidate
    gallery size, not just that the lookup table itself is correct --
    a candidate whose sim_99th clears its OWN bucket's (looser, large-
    gallery) threshold but not a stricter small-gallery threshold must
    still be accepted, and vice versa."""

    def setUp(self):
        self.assigner = faceAssigner()
        self.small_person = make_person("Small Gallery Person")
        self.large_person = make_person("Large Gallery Person")
        self.assigner.likely_people_ids = [self.small_person.id, self.large_person.id]
        self.assigner.num_likely_people = 2

        self.query = np.zeros(512)
        self.query[0] = 1.0

        # A similarity strictly between the smallest bucket's threshold
        # (0.558) and the largest bucket's threshold (0.394) -- accepted
        # for a large gallery, rejected for a small one.
        self.mid_similarity = 0.50
        ref = self.query.copy()
        ref[0] = self.mid_similarity
        ref[1] = np.sqrt(1 - self.mid_similarity ** 2)  # unit vector at exactly mid_similarity cosine to query

        small_gallery_size = 10   # falls in the [10,50) bucket -> threshold 0.558
        large_gallery_size = 600  # falls in the [500+) bucket -> threshold 0.394

        self.assigner.embedding_dict = {
            self.small_person.id: np.tile(ref.reshape(512, 1), (1, small_gallery_size)),
            self.large_person.id: np.tile(ref.reshape(512, 1), (1, large_gallery_size)),
        }
        self.assigner.norm_dict = {
            self.small_person.id: np.ones(small_gallery_size),
            self.large_person.id: np.ones(large_gallery_size),
        }
        self.assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        self.face = make_face(img, declared_name=blank_person)
        self.face.face_encoding_512 = self.query.tolist()
        self.face.save()

    def test_same_similarity_accepted_for_large_gallery_rejected_for_small(self):
        # Reject the small-gallery candidate and the ignore person up
        # front, isolating whether the large-gallery candidate alone
        # gets proposed -- this only happens if its looser threshold is
        # actually being applied.
        self.face.rejected_fields = [self.small_person.id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, self.large_person.id)

    def test_same_similarity_rejects_small_gallery_when_it_is_the_only_candidate(self):
        self.face.rejected_fields = [self.large_person.id, self.assigner.ignore_person_id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        # Falls through to the "no match cleared threshold" branch, which
        # (since ignore_person_id is itself rejected here) proposes the
        # best-scoring candidate anyway -- confirms it took the reject
        # path, not a confident accept, for the identical similarity
        # value that DID pass for the large-gallery candidate above.
        mock_set.assert_called_once_with(self.small_person.id, 1, self.mid_similarity)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class IgnoreWeightMarginTests(TestCase):
    """The ignore-branch weight is deliberately the OPPOSITE sense of a
    real-match weight: frontend sorts by weight descending, and a face
    that's confidently far from every candidate (a safe, obvious ignore)
    should surface before a face that nearly cleared someone's bar (a
    genuine near-miss worth a human's eyes). Uses each candidate's own
    margin below ITS bucket threshold, not a raw score, since raw scores
    aren't comparable across differently-thresholded buckets."""

    def _make_face_with_single_candidate(self, similarity, gallery_size):
        assigner = faceAssigner()
        person = make_person(f"Candidate {similarity}-{gallery_size}")
        assigner.likely_people_ids = [person.id]

        query = np.zeros(512)
        query[0] = 1.0
        ref = query.copy()
        ref[0] = similarity
        ref[1] = np.sqrt(max(0.0, 1 - similarity ** 2))

        assigner.embedding_dict = {person.id: np.tile(ref.reshape(512, 1), (1, gallery_size))}
        assigner.norm_dict = {person.id: np.ones(gallery_size)}
        assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        face = make_face(img, declared_name=blank_person)
        face.face_encoding_512 = query.tolist()
        face.save()
        return assigner, face

    def test_near_miss_gets_low_ignore_weight(self):
        # Large-gallery bucket threshold is 0.394; scoring 0.39 is a
        # near-miss margin of 0.004 -- should sort near the back.
        assigner, face = self._make_face_with_single_candidate(similarity=0.39, gallery_size=600)

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)

        mock_set.assert_called_once()
        _, precedence, weight = mock_set.call_args[0]
        expected = 0.004 / assigner.IGNORE_WEIGHT_MARGIN_CLAMP
        self.assertAlmostEqual(weight, expected, places=3)
        self.assertLess(weight, 0.1)

    def test_far_from_everyone_gets_high_ignore_weight(self):
        # Same bucket/threshold, but scoring far below it (margin well
        # past the clamp) -- should sort to the very front.
        assigner, face = self._make_face_with_single_candidate(similarity=0.0, gallery_size=600)

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)

        mock_set.assert_called_once()
        _, precedence, weight = mock_set.call_args[0]
        self.assertEqual(weight, 1.0)

    def test_far_beats_near_miss_in_sort_order(self):
        near_miss_assigner, near_miss_face = self._make_face_with_single_candidate(similarity=0.39, gallery_size=600)
        far_assigner, far_face = self._make_face_with_single_candidate(similarity=0.0, gallery_size=600)

        with patch.object(Face, "set_possible_person") as mock_set:
            near_miss_assigner.classify_unassigned(near_miss_face)
            near_miss_weight = mock_set.call_args[0][2]

            far_assigner.classify_unassigned(far_face)
            far_weight = mock_set.call_args[0][2]

        self.assertGreater(far_weight, near_miss_weight)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class VerificationClusterGroupTests(TestCase):
    """Face.verification_cluster_group: nightly per-person complete-linkage
    clustering of confirmed-but-unverified faces, so a human reviewer can
    spot-check a whole visually-coherent group at once. See
    verification_clustering.py / CLAUDE.md for the full investigation
    behind why complete linkage, per-person-only, was chosen."""

    def setUp(self):
        self.person = make_person("Cluster Person")
        self.other_person = make_person("Other Cluster Person")
        self.img = make_image()

    def _make_unverified_face(self, person, embedding, **overrides):
        overrides.setdefault('validated', False)
        return make_face(
            self.img, declared_name=person,
            face_encoding_512=embedding.tolist(), **overrides,
        )

    def test_two_tight_clusters_get_distinct_group_ids_and_singleton_stays_null(self):
        cluster_a = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(4)]
        cluster_b = [self._make_unverified_face(self.person, _embedding(1, seed=100 + i)) for i in range(3)]
        singleton = self._make_unverified_face(self.person, _embedding(2, seed=200))

        cluster_all_unverified_faces()

        groups_a = {Face.objects.get(pk=f.pk).verification_cluster_group for f in cluster_a}
        groups_b = {Face.objects.get(pk=f.pk).verification_cluster_group for f in cluster_b}
        self.assertEqual(len(groups_a), 1)
        self.assertEqual(len(groups_b), 1)
        self.assertNotIn(None, groups_a)
        self.assertNotIn(None, groups_b)
        self.assertNotEqual(groups_a, groups_b)
        self.assertIsNone(Face.objects.get(pk=singleton.pk).verification_cluster_group)

    def test_group_ids_are_independent_per_person(self):
        # Both people's tight clusters should each land on group id 0 --
        # ids are 0-indexed PER PERSON, not globally unique.
        cluster_a = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_b = [self._make_unverified_face(self.other_person, _embedding(0, seed=50 + i)) for i in range(3)]

        cluster_all_unverified_faces()

        self.assertEqual(Face.objects.get(pk=cluster_a[0].pk).verification_cluster_group, 0)
        self.assertEqual(Face.objects.get(pk=cluster_b[0].pk).verification_cluster_group, 0)

    def test_validated_faces_are_excluded(self):
        already_verified = self._make_unverified_face(self.person, _embedding(0, seed=1), validated=True)
        unverified = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(2, 4)]

        cluster_all_unverified_faces()

        self.assertIsNone(Face.objects.get(pk=already_verified.pk).verification_cluster_group)
        self.assertIsNotNone(Face.objects.get(pk=unverified[0].pk).verification_cluster_group)

    def test_ignored_sentinel_person_faces_are_excluded(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        faces = [self._make_unverified_face(ignore_person, _embedding(0, seed=i)) for i in range(3)]

        cluster_all_unverified_faces()

        for f in faces:
            self.assertIsNone(Face.objects.get(pk=f.pk).verification_cluster_group)

    def test_null_and_sentinel_encodings_are_excluded(self):
        f_null = self._make_unverified_face(self.person, _embedding(0, seed=1))
        f_null.face_encoding_512 = None
        f_null.save()

        f_sentinel = self._make_unverified_face(self.person, _embedding(0, seed=2))
        f_sentinel.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        f_sentinel.save()

        f_real_a = self._make_unverified_face(self.person, _embedding(0, seed=3))
        f_real_b = self._make_unverified_face(self.person, _embedding(0, seed=4))

        cluster_all_unverified_faces()

        self.assertIsNone(Face.objects.get(pk=f_null.pk).verification_cluster_group)
        self.assertIsNone(Face.objects.get(pk=f_sentinel.pk).verification_cluster_group)
        self.assertEqual(
            Face.objects.get(pk=f_real_a.pk).verification_cluster_group,
            Face.objects.get(pk=f_real_b.pk).verification_cluster_group,
        )
        self.assertIsNotNone(Face.objects.get(pk=f_real_a.pk).verification_cluster_group)

    def test_nightly_rebuild_clears_stale_groups_not_reproduced_this_run(self):
        # A face that was grouped by a previous run but is no longer
        # eligible (e.g. since verified) must not be left with a stale
        # group id -- the nightly rebuild clears ALL faces first.
        stale = self._make_unverified_face(self.person, _embedding(0, seed=1))
        stale.verification_cluster_group = 7
        stale.validated = True
        stale.save()

        cluster_all_unverified_faces()

        self.assertIsNone(Face.objects.get(pk=stale.pk).verification_cluster_group)

    def test_associate_person_clears_group_immediately(self):
        faces = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_all_unverified_faces()
        self.assertIsNotNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)

        other = make_person("Reassign Target")
        face = Face.objects.get(pk=faces[0].pk)
        face.associate_person(other.id)

        self.assertIsNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)

    def test_verify_person_in_image_clears_group_immediately(self):
        faces = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_all_unverified_faces()

        face = Face.objects.get(pk=faces[0].pk)
        face.verify_person_in_image()

        self.assertIsNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)

    def test_reset_to_pool_clears_group_immediately(self):
        faces = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_all_unverified_faces()

        face = Face.objects.get(pk=faces[0].pk)
        face.reset_to_pool()

        self.assertIsNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class LoadEncodingsCachingTests(TestCase):
    """load_encodings()'s persistent cache: reused as-is within the same
    day, reused across a day boundary if nothing changed, and fully
    rebuilt across a day boundary if something did -- see the method's
    own docstring for the full rationale."""

    def setUp(self):
        self.cache_path = "/tmp/face_manager_test_media/test_encodings_cache.pkl"
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

        self.assigner = faceAssigner()
        self.assigner.ENCODINGS_PKL_FILE = self.cache_path
        self.person = make_person("Cached Person")
        self.assigner.likely_people_ids = [self.person.id]

        img = make_image()
        self.face = make_face(img, declared_name=self.person)
        self.face.face_encoding_512 = ([0.1] * 512)
        self.face.save()

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def test_builds_and_persists_cache_on_first_call(self):
        self.assigner.load_encodings()
        self.assertTrue(os.path.exists(self.cache_path))
        self.assertIn(self.person.id, self.assigner.embedding_dict)

    def test_same_day_reuses_cache_without_requerying(self):
        self.assigner.load_encodings()

        with patch.object(faceAssigner, "_current_face_data_signature") as mock_sig:
            fresh_assigner = faceAssigner()
            fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
            fresh_assigner.likely_people_ids = [self.person.id]
            fresh_assigner.load_encodings()

        mock_sig.assert_not_called()
        self.assertIn(self.person.id, fresh_assigner.embedding_dict)

    def test_next_day_no_changes_keeps_cache(self):
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(days=1)
        with open(self.cache_path, "wb") as fh:
            pickle.dump(cached, fh)

        fresh_assigner = faceAssigner()
        fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
        fresh_assigner.likely_people_ids = [self.person.id]
        with patch.object(Face.objects, "filter", wraps=Face.objects.filter) as mock_filter:
            fresh_assigner.load_encodings()
        # The only DB access should be the cheap signature count query --
        # not a full re-fetch of this person's face rows.
        self.assertEqual(mock_filter.call_count, 1)
        self.assertIn(self.person.id, fresh_assigner.embedding_dict)

    def test_next_day_with_changes_rebuilds_cache(self):
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(days=1)
        cached["signature"] = -1  # force a mismatch against the real current count
        with open(self.cache_path, "wb") as fh:
            pickle.dump(cached, fh)

        # Add a second confirmed face for this person -- a real change
        # the stale cache doesn't know about.
        img2 = make_image()
        second_face = make_face(img2, declared_name=self.person)
        second_face.face_encoding_512 = [0.2] * 512
        second_face.save()

        fresh_assigner = faceAssigner()
        fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
        fresh_assigner.likely_people_ids = [self.person.id]
        fresh_assigner.load_encodings()

        self.assertEqual(len(fresh_assigner.candidate_dict[self.person.id]), 2)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ExecuteTrueingUpTests(TestCase):
    """Regression coverage for execute()'s "Verifying face counts"
    trueing-up pass (Person.objects.all(), recomputing num_faces/
    num_possibilities/num_unverified_faces from scratch): during a
    since-reverted multi-threading experiment, this briefly ended up
    inside the per-face helper instead of execute() itself, so it ran
    once PER FACE instead of once per execute() call -- a real
    production incident where a 140k-face reprocess's ETA jumped from
    ~10 hours to ~92 (140k redundant full-Person-table passes). Worth
    keeping this test even though the threading experiment itself was
    reverted (threading didn't actually speed things up here -- likely
    BLAS thread oversubscription from numpy already parallelizing the
    matmul internally -- and wasn't worth the added complexity)."""

    def test_trueing_up_pass_runs_exactly_once_not_per_face(self):
        img = make_image()
        prolific = make_person("Trueing Up Person")
        base = np.zeros(512)
        base[0] = 1.0
        for _ in range(12):
            f = make_face(img, declared_name=prolific)
            f.face_encoding_512 = base.tolist()
            f.save()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        for _ in range(4):
            f = make_face(img, declared_name=blank_person)
            f.face_encoding_512 = base.tolist()
            f.save()

        assigner = faceAssigner()
        assigner.ENCODINGS_PKL_FILE = "/tmp/face_manager_test_media/exec_trueing_up_cache.pkl"

        with patch.object(Person.objects, 'all', wraps=Person.objects.all) as mock_all:
            assigner.execute(redo_all=True)
        self.assertEqual(mock_all.call_count, 1)


class FlattenKpsTests(unittest.TestCase):
    """Unit tests for FaceExtractor._flatten_kps(), the helper that turns
    InsightFace's (5, 2) landmark array into the flat 10-float list stored
    on Face.kps. No DB/model access needed."""

    def test_none_input_returns_none(self):
        self.assertIsNone(FaceExtractor._flatten_kps(None))

    def test_flattens_in_row_major_order(self):
        kps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        flat = FaceExtractor._flatten_kps(kps)
        self.assertEqual(flat, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_accepts_plain_nested_list(self):
        kps = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        flat = FaceExtractor._flatten_kps(kps)
        self.assertEqual(len(flat), 10)
        self.assertTrue(all(isinstance(v, float) for v in flat))


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ReencodeMissingFacesTests(TestCase):
    """FaceExtractor.reencode_missing_faces() re-encodes faces that have
    no face_encoding_512 but are no longer .ignore/.realignore. These
    tests mock the underlying InsightFace models entirely (constructing
    the extractor via __new__ rather than FaceExtractor(), which would
    load the real ~100MB ONNX models) so they stay fast -- no real
    inference happens, only the selection/branching logic in
    reencode_missing_faces() itself is exercised."""

    def setUp(self):
        self.image = make_image()
        self.extractor = FaceExtractor.__new__(FaceExtractor)
        self.extractor.reencode_crop_margin_mult = 0.6
        # extractor.app is a PyramidalDetector in real code; .app.app is
        # the underlying FaceAnalysis. Mock both layers directly rather
        # than constructing a real PyramidalDetector/FaceAnalysis.
        self.mock_face_analysis = MagicMock()
        self.mock_pyramidal = MagicMock()
        self.mock_pyramidal.app = self.mock_face_analysis
        self.extractor.app = self.mock_pyramidal

    def _face_missing_encoding(self, declared_name=None, **overrides):
        face = make_face(self.image, declared_name=declared_name, **overrides)
        face.face_encoding_512 = None
        face.save()
        return face

    def test_ignore_and_realignore_faces_are_never_selected(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        realignore_person = Person.objects.get(person_name='.realignore')
        ignored_face = self._face_missing_encoding(declared_name=ignore_person)
        realignored_face = self._face_missing_encoding(declared_name=realignore_person)

        self.extractor.reencode_missing_faces()

        ignored_face.refresh_from_db()
        realignored_face.refresh_from_db()
        self.assertIsNone(ignored_face.face_encoding_512)
        self.assertIsNone(realignored_face.face_encoding_512)
        self.mock_face_analysis.get.assert_not_called()

    def test_non_detected_sentinel_on_a_real_person_is_selected_for_reencode(self):
        # Regression test: update_list_of_no_matching_detects() stamps
        # settings.NON_DETECTED_FACE_ENCODING ([-999]*512) onto a face
        # whose box wasn't matched to any detection during a full-image
        # reprocessing pass. Found 2026-09-03 via a clustering
        # investigation: 1,207 faces database-wide carried this sentinel
        # while still being declared to a REAL person -- not NULL, so
        # invisible to the original face_encoding_512__isnull=True filter,
        # and not .ignore/.realignore, so this isn't a "confirmed-ignore"
        # case either. Just a real face silently stuck with fake data.
        person = make_person("Has Sentinel Encoding")
        face = make_face(
            self.image, declared_name=person,
            box_left=100, box_top=100, box_right=140, box_bottom=140,
        )
        face.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        face.save()

        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 40, 40], 'kps': np.zeros((5, 2)), 'embedding': np.full(512, 0.4)},
        ]

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.4] * 512)
        self.assertTrue(face.reencoded)

    def test_non_detected_sentinel_on_ignore_face_is_still_excluded(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        face = make_face(self.image, declared_name=ignore_person)
        face.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        face.save()

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, list(settings.NON_DETECTED_FACE_ENCODING))
        self.mock_face_analysis.get.assert_not_called()

    def test_faces_with_an_existing_encoding_are_left_alone(self):
        person = make_person("Already Encoded")
        face = make_face(self.image, declared_name=person)
        # 0.5 is exactly representable in float32 -- avoids the precision
        # trap in test_face_encoding_512_stores_at_float32_precision's
        # docstring (a value like 0.1 would round on save, so comparing
        # against the original float64 literal after refresh_from_db()
        # would spuriously fail here).
        face.face_encoding_512 = [0.5] * 512
        face.save()

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)
        self.assertFalse(face.reencoded)
        self.mock_face_analysis.get.assert_not_called()

    def test_kps_path_reproduces_embedding_without_redetecting(self):
        """When a face already has stored kps, reencode_missing_faces()
        should call the recognition model directly (no detection pass at
        all) -- this is the exact-reproduction path validated against
        real production data."""
        person = make_person("Has Kps")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        face.save()

        expected_embedding = np.arange(512, dtype=np.float64)
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=expected_embedding)),
        }

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, expected_embedding.tolist())
        self.assertTrue(face.reencoded)
        self.mock_face_analysis.get.assert_not_called()

        # The recognition model should have been called with a Face-like
        # object carrying the ORIGINAL stored kps, reshaped to (5, 2).
        call_args = self.mock_face_analysis.models['recognition'].get.call_args
        passed_face_obj = call_args[0][1]
        np.testing.assert_array_equal(
            np.asarray(passed_face_obj.kps), np.array(face.kps).reshape(5, 2)
        )

    def test_no_kps_falls_back_to_crop_detection_and_picks_center_face(self):
        """Without stored kps, the crop-based path should run, and among
        multiple detections found in the crop, the one closest to the
        crop's center (i.e. closest to the known face) should win."""
        person = make_person("No Kps")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        self.assertIsNone(face.kps)

        near_center_embedding = np.full(512, 0.5)
        far_embedding = np.full(512, 0.9)
        # Crop center will be at roughly (crop_w/2, crop_h/2) since the
        # margin is symmetric -- a detection near (0, 0) is far off-center,
        # one near the crop's actual center should be picked instead.
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 4, 4], 'kps': np.zeros((5, 2)), 'embedding': far_embedding},
            {'bbox': [30, 30, 34, 34], 'kps': np.ones((5, 2)), 'embedding': near_center_embedding},
        ]
        self.mock_face_analysis.models = {'recognition': MagicMock()}

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, near_center_embedding.tolist())
        self.assertTrue(face.reencoded)
        self.mock_face_analysis.models['recognition'].get.assert_not_called()

    def test_crop_detection_kps_saved_in_absolute_image_coordinates(self):
        """Freshly-detected kps from the crop-based path are in
        crop-local coordinates -- they must be translated back to the
        source image's absolute coordinate space (adding the crop's own
        offset) before being stored, matching how add_new_face()/
        update_existing_face_to_insightface() store kps, so a future
        re-encode of this same face can use the exact-replay path."""
        person = make_person("Needs Translation")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        box_w, box_h = 40, 40
        margin_x = int(box_w * self.extractor.reencode_crop_margin_mult)
        margin_y = int(box_h * self.extractor.reencode_crop_margin_mult)
        crop_left, crop_top = 100 - margin_x, 100 - margin_y

        crop_local_kps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 40, 40], 'kps': crop_local_kps, 'embedding': np.full(512, 0.3)},
        ]

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        expected_abs_kps = (crop_local_kps + np.array([crop_left, crop_top])).reshape(-1).tolist()
        self.assertEqual(face.kps, expected_abs_kps)

    def test_no_detection_found_uses_default_encoding_without_setting_reencoded(self):
        """A face too degraded to redetect even in a tight crop gets the
        neutral default vector rather than being left NULL forever (NULL
        is exactly what this method selects on) -- but since nothing
        about it actually came from a fresh detection, reencoded should
        NOT be flipped to True."""
        person = make_person("Undetectable")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        self.mock_face_analysis.get.return_value = []

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        # face_encoding_512 is stored at float32 precision (see
        # test_face_encoding_512_stores_at_float32_precision) -- compare
        # with a tolerance rather than exact equality against the
        # float64-computed settings constant, both to be correct about
        # what the DB actually guarantees and to avoid unittest's
        # list-diff (difflib recursion crashes on a 512-element mismatch
        # rather than reporting a clean failure).
        np.testing.assert_allclose(
            np.array(face.face_encoding_512), np.array(settings.REENCODE_DEFAULT_ENCODING), rtol=1e-6, atol=1e-6
        )
        self.assertFalse(face.reencoded)
        self.assertIsNone(face.kps)

    def test_reencode_default_encoding_is_a_unit_vector(self):
        vec = np.array(settings.REENCODE_DEFAULT_ENCODING)
        self.assertEqual(len(vec), 512)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)
        self.assertEqual(len(set(vec.tolist())), 1, "Every component should be identical/uniform")


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class DedupeOverlappingFacesTests(TestCase):
    """dedupe_overlapping_faces management command -- cleans up the
    duplicate Face rows left behind by the (now-fixed, see common/
    advisory_lock.py) find_and_encode_faces() race condition: two
    concurrent runs on the same never-before-processed image could each
    insert a Face row for the same real face, pixel-identical box."""

    def setUp(self):
        self.person = make_person("Dup Person")
        self.other_person = make_person("Other Dup Person")
        self.img = make_image()

    def _duplicate_pair(self, **overrides_a):
        overrides_b = dict(overrides_a)
        a = make_face(self.img, declared_name=self.person, box_left=1, box_top=1,
                       box_right=30, box_bottom=30, **overrides_a)
        b = make_face(self.img, declared_name=self.person, box_left=1, box_top=1,
                       box_right=30, box_bottom=30, **overrides_b)
        return a, b

    def test_dry_run_reports_but_changes_nothing(self):
        a, b = self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--dry-run')
        self.assertEqual(Face.objects.filter(pk__in=[a.pk, b.pk]).count(), 2)

    def test_identical_pair_collapses_to_one_survivor(self):
        a, b = self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertEqual(Face.objects.filter(source_image_file=self.img).count(), 1)

    def test_validated_face_is_kept_over_unvalidated(self):
        a, b = self._duplicate_pair()
        b.validated = True
        b.save()
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, b.pk)

    def test_labeled_face_is_kept_over_unlabeled_when_neither_validated(self):
        # Common real scenario: a human tagged one copy of a duplicate
        # pair without knowing the other copy existed, leaving it at the
        # blank sentinel forever. The labeled copy should survive.
        blank = get_default_blank_person()
        a = make_face(self.img, declared_name=blank, box_left=1, box_top=1, box_right=30, box_bottom=30)
        b = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, b.pk)

    def test_validated_takes_priority_over_label_when_the_validated_one_is_unlabeled(self):
        # Validated (real completed human verification) still outranks
        # "has a label" -- shouldn't actually diverge in practice (a
        # validated face should already be labeled), but the ordering
        # itself should hold regardless.
        blank = get_default_blank_person()
        a = make_face(self.img, declared_name=blank, box_left=1, box_top=1, box_right=30, box_bottom=30,
                      validated=True)
        b = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, a.pk)

    def test_face_with_kps_is_kept_over_one_without_when_neither_validated(self):
        a, b = self._duplicate_pair()
        b.kps = [1.0] * 10
        b.save()
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, b.pk)

    def test_lowest_id_is_kept_as_final_tiebreaker(self):
        a, b = self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, min(a.pk, b.pk))

    def test_deleted_faces_thumbnail_file_is_removed(self):
        a, b = self._duplicate_pair()
        loser_path = a.face_thumbnail.path if a.pk != min(a.pk, b.pk) else b.face_thumbnail.path
        self.assertTrue(os.path.exists(loser_path))
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertFalse(os.path.exists(loser_path))

    def test_two_separate_duplicate_pairs_on_same_image_both_collapse(self):
        a1 = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        a2 = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        b1 = make_face(self.img, declared_name=self.other_person, box_left=50, box_top=50, box_right=80, box_bottom=80)
        b2 = make_face(self.img, declared_name=self.other_person, box_left=50, box_top=50, box_right=80, box_bottom=80)
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertEqual(Face.objects.filter(source_image_file=self.img).count(), 2)

    def test_non_overlapping_faces_on_same_image_are_left_alone(self):
        a = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        b = make_face(self.img, declared_name=self.other_person, box_left=50, box_top=50, box_right=80, box_bottom=80)
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertEqual(Face.objects.filter(pk__in=[a.pk, b.pk]).count(), 2)

    def test_person_face_counts_are_recomputed_after_dedupe(self):
        a, b = self._duplicate_pair()
        self.person.num_faces = 999  # deliberately stale
        self.person.save()
        call_command('dedupe_overlapping_faces', '--yes')
        self.person.refresh_from_db()
        self.assertEqual(self.person.num_faces, 1)

    def test_rerunning_after_cleanup_finds_nothing(self):
        self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--yes')
        call_command('dedupe_overlapping_faces', '--dry-run')
        self.assertEqual(Face.objects.filter(source_image_file=self.img).count(), 1)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class MergeDuplicateImageFilesTests(TestCase):
    """merge_duplicate_imagefiles management command -- cleans up the
    duplicate ImageFile rows left behind by the (now-fixed, see
    filepopulator/scripts.py's create_image_file()) missing-return bug:
    a genuine duplicate file used to get BOTH a correct DuplicateFile
    record AND a redundant ImageFile row of its own, which could carry
    real independent human work (labels, validations). Test data seeds
    the contaminated state directly via bulk_create, mirroring
    filepopulator.tests's own approach for the same reason: the normal
    create_image_file() ingestion path no longer produces this state at
    all now that it's fixed."""

    def setUp(self):
        self.person = make_person("Merge Person")
        self.primary_img = make_image()

    def _make_contaminated_duplicate(self, primary):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(primary.filename))
        dup_filename = primary.filename + '.dup.jpg'
        ImageFile.objects.bulk_create([ImageFile(
            filename=dup_filename, directory=directory,
            pixel_hash=primary.pixel_hash, file_hash=primary.file_hash,
            width=primary.width, height=primary.height, isProcessed=False,
            thumbnail_big='', thumbnail_medium='', thumbnail_small='',
        )])
        dup = ImageFile.objects.get(filename=dup_filename)
        DuplicateFile.objects.create(filename=dup_filename)
        return dup

    def test_face_is_transferred_from_duplicate_to_primary(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        face = make_face(dup, declared_name=self.person, validated=True,
                          box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')

        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, self.primary_img.id)
        self.assertFalse(ImageFile.objects.filter(pk=dup.pk).exists())

    def test_duplicate_face_pairs_after_transfer_collapse_preferring_validated(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        blank = get_default_blank_person()
        # A face already exists on the primary at this box, unlabeled/unvalidated.
        make_face(self.primary_img, declared_name=blank,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)
        # The duplicate's copy of the SAME face was validated + labeled --
        # a human tagged it without knowing it was on a duplicate photo.
        dup_face = make_face(dup, declared_name=self.person, validated=True,
                              box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')

        remaining = Face.objects.get(source_image_file=self.primary_img)
        self.assertEqual(remaining.pk, dup_face.pk)
        self.assertTrue(remaining.validated)

    def test_unresolved_duplicate_is_left_alone_when_no_primary_found(self):
        # A DuplicateFile record with no OTHER ImageFile sharing its
        # pixel_hash -- e.g. the primary was itself separately removed.
        # Must be left alone, not guessed at or deleted.
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp')
        ImageFile.objects.bulk_create([ImageFile(
            filename='/tmp/orphan_dup.jpg', directory=directory,
            pixel_hash='no_matching_hash_at_all', file_hash='x',
            width=10, height=10, isProcessed=False,
            thumbnail_big='', thumbnail_medium='', thumbnail_small='',
        )])
        DuplicateFile.objects.create(filename='/tmp/orphan_dup.jpg')

        call_command('merge_duplicate_imagefiles', '--yes')

        self.assertTrue(ImageFile.objects.filter(filename='/tmp/orphan_dup.jpg').exists())

    def test_dry_run_changes_nothing(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        face = make_face(dup, declared_name=self.person, validated=True,
                          box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--dry-run')

        self.assertTrue(ImageFile.objects.filter(pk=dup.pk).exists())
        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, dup.pk)

    def test_person_face_counts_recomputed_for_collapsed_away_person(self):
        # A plain transfer with no collision doesn't change any person's
        # face count (declared_name is untouched by moving source_image_
        # file) -- recompute only matters when the post-transfer collapse
        # actually DELETES a face, which is what this exercises: the
        # losing side's stale cached count should get corrected.
        dup = self._make_contaminated_duplicate(self.primary_img)
        losing_person = make_person("Losing Person")
        make_face(self.primary_img, declared_name=losing_person,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)
        make_face(dup, declared_name=self.person, validated=True,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)
        losing_person.num_faces = 999
        losing_person.save()

        call_command('merge_duplicate_imagefiles', '--yes')

        losing_person.refresh_from_db()
        self.assertEqual(losing_person.num_faces, 0)

    def test_rerunning_after_merge_finds_nothing_more_to_do(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        make_face(dup, declared_name=self.person,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')
        call_command('merge_duplicate_imagefiles', '--dry-run')

        dup_filenames = set(DuplicateFile.objects.values_list('filename', flat=True))
        self.assertEqual(ImageFile.objects.filter(filename__in=dup_filenames).count(), 0)
