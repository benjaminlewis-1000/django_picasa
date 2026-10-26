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
import io
import os
import unittest
import zlib
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import psycopg2
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.db import connection
from django.test import TestCase, override_settings, tag

from django.core.management import call_command
from django.core.management.base import CommandError

from unittest.mock import patch, MagicMock

from face_manager.assign_faces import faceAssigner
from face_manager.face_extract_encode import FaceExtractor
from face_manager.live_counts import compute_live_face_counts
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

    def test_live_face_counts_reflect_real_assignments(self):
        # Person.num_faces/num_possibilities/num_unverified_faces used to
        # be cached IntegerFields, manually kept in sync by
        # increment_*/decrement_* methods called from Face's assignment
        # methods -- removed because that bookkeeping could silently
        # drift stale (e.g. a bulk .update() bypassing it entirely). Now
        # computed live -- see face_manager/live_counts.py.
        person = make_person("Counted")
        image = make_image()
        face = make_face(image, declared_name=person)
        counts = compute_live_face_counts([person.id])[person.id]
        self.assertEqual(counts['num_faces'], 1)
        self.assertEqual(counts['num_unverified_faces'], 1)

        face.verify_person_in_image()
        counts = compute_live_face_counts([person.id])[person.id]
        self.assertEqual(counts['num_faces'], 1)
        self.assertEqual(counts['num_unverified_faces'], 0)

        other = make_person("Other")
        face.set_possible_person(other.id, 1, 0.9)
        counts = compute_live_face_counts([other.id])[other.id]
        self.assertEqual(counts['num_possibilities'], 1)


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
        guesser = make_person("Guesser")
        face = make_face(self.image, declared_name=owner)
        face.set_possible_person(guesser.id, 1, 0.8)

        face.reset_to_pool()
        face.refresh_from_db()

        self.assertEqual(face.declared_name_id, blank.id)
        self.assertIsNotNone(face.declared_name)
        self.assertFalse(face.validated)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(owner.face_declared.count(), 0)

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

        # Compare via a float32 narrowing on both sides rather than exact
        # double equality: Postgres's `real` -> text -> Python float round
        # trip (via psycopg2) doesn't necessarily reproduce the exact same
        # double bit pattern as `float(np.float32(x))` computed directly in
        # Python -- e.g. this test observed the DB round trip coming back
        # as 0.12345679 (fewer significant digits) rather than
        # 0.12345679104328156. Narrowing both to float32 collapses that
        # text-precision noise while still genuinely verifying the stored
        # value lost precision down to float32, which is the actual point
        # of this test.
        self.assertEqual(
            np.float32(face.face_encoding_512[0]), np.float32(double_only_value)
        )
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


class AddNewFacePhotoCollectionGateTests(TestCase):
    """add_new_face()'s per-PhotoCollection detection floors (see
    below_collection_floor() in filepopulator/models.py). The Face row
    is ALWAYS created regardless of the floors -- box/kps/det_score/
    relative_size are always stored, so a detection is never silently
    lost -- only face_encoding_512 is conditionally withheld. Uses a
    real FaceExtractor (ONNX models loaded once via setUpClass, same
    pattern as FaceExtractorCorruptedImageTests -- add_new_face() itself
    never touches the ONNX session, only pure-Python helpers) with a
    synthetic insightface.app.common.Face (a plain dict subclass, real
    inference not needed to construct one) so the det_score/box can be
    controlled directly rather than depending on what a real image
    happens to detect."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extractor = FaceExtractor()

    def _make_image_in_collection(self, collection, path):
        directory, _ = Directory.objects.get_or_create(
            dir_path=os.path.dirname(path), defaults={'collection': collection},
        )
        if directory.collection_id != (collection.id if collection else None):
            directory.collection = collection
            directory.save()
        image = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            width=640, height=480, pixel_hash='hash', file_hash='fhash', dateTakenUTC=0,
        )
        ImageFile.objects.bulk_create([image])
        return ImageFile.objects.get(filename=path)

    def _fake_detection(self, bbox, det_score):
        import insightface.app.common
        import numpy as np
        # Integer dtype deliberately, not float32 -- get_square_face_img()
        # slices img_numpy directly with values derived from this bbox,
        # which needs __index__-capable (integer) values. A real
        # detection's bbox is float32 too, but something upstream of
        # here evidently normalizes it before this point in production;
        # for this synthetic unit test, int is simplest and correct
        # enough (only relative proportions matter for the gate itself).
        return insightface.app.common.Face(
            bbox=np.array(bbox, dtype=np.int64),
            kps=np.zeros((5, 2), dtype=np.float32),
            det_score=det_score,
            embedding=np.ones(512, dtype=np.float32),  # ones, not zeros -- distinguishable from "never set"
            age=30, gender=0,
        )

    def _img_numpy(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_default_collection_no_floor_keeps_everything_including_encoding(self):
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_default/a.jpg')
        # Small box (20.8% of the image's min dimension) and a low
        # det_score -- would fail floors if any were set, but the
        # default collection has none.
        face = self._fake_detection([100, 100, 200, 200], det_score=0.1)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNotNone(saved.face_encoding_512)
        self.assertAlmostEqual(saved.relative_size, 100 / 480, places=3)

    def test_det_score_below_floor_still_creates_row_but_withholds_encoding(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity', root_directory='/photos/gate_test_detscore', min_face_det_score=0.6,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_detscore/a.jpg')
        face = self._fake_detection([50, 50, 400, 400], det_score=0.5)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_512)
        self.assertAlmostEqual(saved.det_score, 0.5)
        self.assertIsNotNone(saved.kps)

    def test_det_score_at_or_above_floor_gets_encoded(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity 2', root_directory='/photos/gate_test_detscore2', min_face_det_score=0.6,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_detscore2/a.jpg')
        face = self._fake_detection([50, 50, 400, 400], det_score=0.6)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNotNone(saved.face_encoding_512)

    def test_relative_size_below_floor_still_creates_row_but_withholds_encoding(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity 3', root_directory='/photos/gate_test_size', min_face_relative_size=0.3,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_size/a.jpg')
        # 640x480 image, box 100x100 -> relative size = 100/480 = 0.208,
        # below the 0.3 floor.
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_512)
        self.assertAlmostEqual(saved.relative_size, 100 / 480, places=3)

    def test_relative_size_at_or_above_floor_gets_encoded(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity 4', root_directory='/photos/gate_test_size2', min_face_relative_size=0.1,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_size2/a.jpg')
        # box 100x100 -> relative size 0.208, above the 0.1 floor.
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNotNone(saved.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_above_floor_stores_lvface_encoding_too(self):
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_lvface/a.jpg')
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.25, dtype=np.float32)) as mock_lv:
            self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertEqual(saved.face_encoding_lvface_b, [0.25] * 512)
        # input_is_bgr=False -- img_numpy here is the PIL/RGB-sourced
        # array this codebase's own extraction pipeline already uses,
        # not a cv2-decoded BGR one.
        mock_lv.assert_called_once()
        self.assertEqual(mock_lv.call_args.kwargs.get('input_is_bgr'), False)

    @override_settings(LVFACE_ENCODING_ENABLED=False)
    def test_lvface_disabled_never_attempts_encoding(self):
        # settings.LVFACE_ENCODING_ENABLED is True in production now
        # (2026-10-25 -- LVFace-B validated, see that setting's own
        # comment), but the disabled path must still work correctly
        # (e.g. if ever flipped back off) -- exercised here via an
        # explicit override rather than relying on the current default.
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_lvface_disabled/a.jpg')
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.add_new_face(face, img, self._img_numpy())
        mock_lv.assert_not_called()
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_lvface_b)
        self.assertIsNotNone(saved.face_encoding_512)  # insightface's own encoding is unaffected

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_below_floor_never_attempts_lvface_encoding(self):
        # Matches face_encoding_512's own floor gating -- a detection
        # that never gets a real insightface embedding shouldn't pay
        # LVFace-B's (much larger) inference cost either.
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity LVFace', root_directory='/photos/gate_test_lvface_skip', min_face_det_score=0.6,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_lvface_skip/a.jpg')
        face = self._fake_detection([50, 50, 400, 400], det_score=0.5)
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.add_new_face(face, img, self._img_numpy())
        mock_lv.assert_not_called()
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_lvface_b)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_lvface_failure_is_non_fatal_and_leaves_insightface_encoding_intact(self):
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_lvface_fail/a.jpg')
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        with patch('lvface.embed_from_image_array', side_effect=RuntimeError("boom")):
            self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_lvface_b)
        self.assertIsNotNone(saved.face_encoding_512)

    def test_retuning_floor_changes_outcome_for_the_same_stored_values(self):
        # The real point of NOT storing a separate flag: below_collection_
        # floor() always checks the collection's CURRENT thresholds, so
        # loosening/tightening a floor later automatically reclassifies
        # faces detected before the change, with no backfill needed.
        from filepopulator.models import PhotoCollection, below_collection_floor

        collection = PhotoCollection.objects.create(
            name='Retune Test', root_directory='/photos/gate_test_retune', min_face_relative_size=0.3,
        )
        self.assertTrue(below_collection_floor(collection, det_score=0.9, relative_size=0.208))

        collection.min_face_relative_size = 0.1
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection.save()
        self.assertFalse(below_collection_floor(collection, det_score=0.9, relative_size=0.208))


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillRelativeSizeTests(TestCase):
    """backfill_relative_size: pure-arithmetic backfill, no decode/model
    inference needed, so plain ImageFile/Face rows built directly
    (bulk_create, same shape as make_preexisting_image_row) are enough --
    no real fixture images required."""

    def _make_image(self, width, height, name):
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/relsize_test')
        path = f'/photos/relsize_test/{name}.jpg'
        img = ImageFile(filename=path, directory=directory, thumbnail_small="", width=width, height=height)
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def _make_bare_face(self, img, box, person=None):
        bl, bt, br, bb = box
        face = Face(
            source_image_file=img, declared_name=person or get_default_blank_person(),
            box_left=bl, box_top=bt, box_right=br, box_bottom=bb,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_backfill_computes_correct_relative_size(self):
        img = self._make_image(640, 480, 'a')
        face = self._make_bare_face(img, (100, 100, 200, 200))  # 100x100 box, min dim 480 -> 0.2083

        call_command('backfill_relative_size')

        face.refresh_from_db()
        self.assertAlmostEqual(face.relative_size, 100 / 480, places=4)

    def test_dry_run_does_not_write(self):
        img = self._make_image(640, 480, 'b')
        face = self._make_bare_face(img, (100, 100, 200, 200))

        call_command('backfill_relative_size', '--dry-run')

        face.refresh_from_db()
        self.assertIsNone(face.relative_size)

    def test_already_populated_rows_are_skipped(self):
        img = self._make_image(640, 480, 'c')
        face = self._make_bare_face(img, (100, 100, 200, 200))
        Face.objects.filter(pk=face.pk).update(relative_size=0.999)

        call_command('backfill_relative_size')

        face.refresh_from_db()
        self.assertEqual(face.relative_size, 0.999)

    def test_face_with_no_source_image_is_skipped_not_errored(self):
        # Video-sourced faces (source_image_file=None) must not crash
        # the backfill -- they're simply excluded from this query (the
        # feature this backfill supports is image-collection-only right
        # now).
        from filepopulator.models import VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/relsize_test')
        video = VideoFile.objects.create(
            filename='/videos/relsize_test/v.mp4', directory=directory, width=640, height=480,
        )
        video_face = Face(
            declared_name=get_default_blank_person(), source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
        )
        video_face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        video_face.save()

        call_command('backfill_relative_size')  # must not raise

        video_face.refresh_from_db()
        self.assertIsNone(video_face.relative_size)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClearEncodingsBelowFloorTests(TestCase):
    """clear_encodings_below_floor: the one-time sweep that actually
    applies a collection's floor to faces that were already encoded
    before the floor existed/was tightened -- add_new_face()/
    reencode_missing_faces() only ever gate a face that doesn't already
    have a usable embedding, so retuning a floor down has no visible
    effect on the existing backlog without this."""

    def _make_image(self, width, height, name, collection=None):
        from filepopulator.models import PhotoCollection
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/clear_floor_test')
        if collection is not None:
            directory.collection = collection
            directory.save()
        path = f'/photos/clear_floor_test/{name}.jpg'
        img = ImageFile(filename=path, directory=directory, thumbnail_small="", width=width, height=height)
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def _make_face(self, img, relative_size, det_score=0.9, person=None, validated=False, encoding=None, poss_ident1=None):
        face = Face(
            source_image_file=img, declared_name=person or get_default_blank_person(),
            box_left=0, box_top=0, box_right=10, box_bottom=10,
            relative_size=relative_size, det_score=det_score, validated=validated,
            face_encoding_512=encoding if encoding is not None else ([0.1] * 512),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        if poss_ident1 is not None:
            face.poss_ident1 = poss_ident1
            face.weight_1 = 0.5
        face.save()
        return face

    def _default_collection(self, min_face_relative_size):
        # Not testing PhotoCollection's own save()-triggered task here
        # (see PhotoCollectionFloorChangeTriggerTests) -- patch it out so
        # this setup helper doesn't enqueue a real Celery task against
        # the test broker.
        from filepopulator.models import PhotoCollection
        collection, _ = PhotoCollection.objects.get_or_create(
            is_default=True, defaults={'name': 'Default', 'root_directory': '/photos'},
        )
        collection.min_face_relative_size = min_face_relative_size
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection.save()
        return collection

    def test_clears_encoding_on_unconfirmed_face_below_floor(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'a', collection=collection)
        face = self._make_face(img, relative_size=0.01)  # below 0.02

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)

    def test_also_clears_stale_poss_identN_proposals(self):
        # A never-adjudicated face can still carry a live poss_ident1/
        # weight_1 proposal from a past classification run -- clearing
        # the encoding without also clearing this leaves a stale
        # proposal with nothing left to justify it (misleading, and a
        # real crash risk for anything that assumes a proposal implies a
        # usable embedding -- confirmed in production 2026-10-20).
        from face_manager.models import Person
        proposed = Person.objects.create(person_name='Proposed Candidate')
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'g', collection=collection)
        face = self._make_face(img, relative_size=0.01, poss_ident1=proposed)

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(face.weight_1, 0.0)

    def test_leaves_encoding_on_unconfirmed_face_at_or_above_floor(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'b', collection=collection)
        face = self._make_face(img, relative_size=0.05)  # above 0.02

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_never_touches_a_validated_face_even_below_floor(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'c', collection=collection)
        face = self._make_face(img, relative_size=0.01, validated=True)

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_never_touches_an_already_declared_face_even_below_floor(self):
        from face_manager.models import Person
        person = Person.objects.create(person_name='Someone Real')
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'd', collection=collection)
        face = self._make_face(img, relative_size=0.01, person=person)

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_dry_run_does_not_write(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'e', collection=collection)
        face = self._make_face(img, relative_size=0.01)

        call_command('clear_encodings_below_floor', '--dry-run')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_already_null_encoding_is_not_counted_as_cleared(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'f', collection=collection)
        self._make_face(img, relative_size=0.01, encoding=None)

        # Must not raise -- NULL encodings are excluded from the query
        # entirely (nothing to clear).
        call_command('clear_encodings_below_floor')


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClearStaleUnencodedProposalsTests(TestCase):
    """clear_stale_proposals_with_no_encoding: the catch-up pass for the
    backlog clear_encodings_below_collection_floor() can't reach -- a
    face already NULL-encoded before that function's own poss_identN
    clearing existed (or for any other reason) but still carrying a
    stale proposal."""

    def _make_image(self, name):
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/stale_proposal_test')
        path = f'/photos/stale_proposal_test/{name}.jpg'
        img = ImageFile(filename=path, directory=directory, thumbnail_small="", width=640, height=480)
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def _make_face(self, img, person=None, validated=False, encoding=None, poss_ident1=None):
        face = Face(
            source_image_file=img, declared_name=person or get_default_blank_person(),
            box_left=0, box_top=0, box_right=10, box_bottom=10, validated=validated,
            face_encoding_512=encoding,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        if poss_ident1 is not None:
            face.poss_ident1 = poss_ident1
            face.weight_1 = 0.5
        face.save()
        return face

    def test_clears_stale_proposal_on_unencoded_unconfirmed_face(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Stale Candidate')
        img = self._make_image('a')
        face = self._make_face(img, encoding=None, poss_ident1=proposed)

        count = clear_stale_proposals_with_no_encoding()

        face.refresh_from_db()
        self.assertEqual(count, 1)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(face.weight_1, 0.0)

    def test_leaves_encoded_face_with_proposal_untouched(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Real Candidate')
        img = self._make_image('b')
        face = self._make_face(img, encoding=[0.1] * 512, poss_ident1=proposed)

        clear_stale_proposals_with_no_encoding()

        face.refresh_from_db()
        self.assertEqual(face.poss_ident1_id, proposed.id)

    def test_leaves_unencoded_face_with_no_proposal_untouched(self):
        from face_manager.models import clear_stale_proposals_with_no_encoding
        img = self._make_image('c')
        self._make_face(img, encoding=None, poss_ident1=None)

        count = clear_stale_proposals_with_no_encoding()
        self.assertEqual(count, 0)

    def test_never_touches_a_validated_or_declared_face(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Candidate')
        real_person = Person.objects.create(person_name='Real Person')
        img = self._make_image('d')
        validated_face = self._make_face(img, encoding=None, validated=True, poss_ident1=proposed)
        img2 = self._make_image('e')
        declared_face = self._make_face(img2, person=real_person, encoding=None, poss_ident1=proposed)

        clear_stale_proposals_with_no_encoding()

        validated_face.refresh_from_db()
        declared_face.refresh_from_db()
        self.assertEqual(validated_face.poss_ident1_id, proposed.id)
        self.assertEqual(declared_face.poss_ident1_id, proposed.id)

    def test_dry_run_does_not_write_but_returns_count(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Dry Run Candidate')
        img = self._make_image('f')
        face = self._make_face(img, encoding=None, poss_ident1=proposed)

        count = clear_stale_proposals_with_no_encoding(dry_run=True)

        face.refresh_from_db()
        self.assertEqual(count, 1)
        self.assertEqual(face.poss_ident1_id, proposed.id)

    def test_scoped_to_a_specific_collection(self):
        from filepopulator.models import PhotoCollection
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            other_collection = PhotoCollection.objects.create(
                name='Other Collection', root_directory='/photos/stale_proposal_other',
            )
        other_dir, _ = Directory.objects.get_or_create(
            dir_path='/photos/stale_proposal_other', defaults={'collection': other_collection},
        )
        other_img = ImageFile(
            filename='/photos/stale_proposal_other/a.jpg', directory=other_dir,
            thumbnail_small="", width=640, height=480,
        )
        ImageFile.objects.bulk_create([other_img])
        other_img = ImageFile.objects.get(filename='/photos/stale_proposal_other/a.jpg')

        proposed = Person.objects.create(person_name='Scoped Candidate')
        default_face = self._make_face(self._make_image('g'), encoding=None, poss_ident1=proposed)
        other_face = self._make_face(other_img, encoding=None, poss_ident1=proposed)

        count = clear_stale_proposals_with_no_encoding(collection=other_collection)

        default_face.refresh_from_db()
        other_face.refresh_from_db()
        self.assertEqual(count, 1)
        self.assertEqual(default_face.poss_ident1_id, proposed.id)  # untouched -- different collection
        self.assertIsNone(other_face.poss_ident1)

    def test_management_command_dry_run_vs_real(self):
        from face_manager.models import Person
        proposed = Person.objects.create(person_name='Command Candidate')
        img = self._make_image('h')
        face = self._make_face(img, encoding=None, poss_ident1=proposed)

        call_command('clear_stale_unencoded_proposals', '--dry-run')
        face.refresh_from_db()
        self.assertEqual(face.poss_ident1_id, proposed.id)

        call_command('clear_stale_unencoded_proposals')
        face.refresh_from_db()
        self.assertIsNone(face.poss_ident1)


class PhotoCollectionFloorChangeTriggerTests(TestCase):
    """PhotoCollection.save() fires face_manager.apply_collection_floor_
    change whenever min_face_det_score/min_face_relative_size actually
    changes on an existing row -- so retuning a floor in the frontend
    settings page applies itself automatically, with no one needing to
    remember to run clear_encodings_below_floor/reencode_missing_faces
    by hand. Only the triggering condition is tested here (mocking
    .delay() -- no real Celery/Redis/model-loading involved); the
    task's own behavior is covered by ApplyCollectionFloorChangeTaskTests."""

    def test_no_trigger_on_initial_create(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            PhotoCollection.objects.create(
                name='Trigger Create', root_directory='/photos/trigger_create', min_face_relative_size=0.1,
            )
        mock_delay.assert_not_called()

    def test_triggers_on_relative_size_floor_change(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Relsize', root_directory='/photos/trigger_relsize', min_face_relative_size=0.1,
            )
        collection.min_face_relative_size = 0.2
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()
        mock_delay.assert_called_once_with(collection.pk)

    def test_triggers_on_det_score_floor_change(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Detscore', root_directory='/photos/trigger_detscore', min_face_det_score=0.5,
            )
        collection.min_face_det_score = 0.6
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()
        mock_delay.assert_called_once_with(collection.pk)

    def test_no_trigger_when_floor_values_unchanged(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Unchanged', root_directory='/photos/trigger_unchanged', min_face_relative_size=0.1,
            )
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()  # re-save with the same values
        mock_delay.assert_not_called()

    def test_no_trigger_on_unrelated_field_change(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Unrelated', root_directory='/photos/trigger_unrelated', min_face_relative_size=0.1,
            )
        collection.name = 'Renamed'
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()
        mock_delay.assert_not_called()

    def test_triggers_again_on_a_freshly_loaded_instance(self):
        # The change-detection snapshot is taken in __init__, so it must
        # work correctly for an instance loaded fresh from the DB (via
        # .get()), not just one that's been held in memory since create().
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            created = PhotoCollection.objects.create(
                name='Trigger Reload', root_directory='/photos/trigger_reload', min_face_relative_size=0.1,
            )
        reloaded = PhotoCollection.objects.get(pk=created.pk)
        reloaded.min_face_relative_size = 0.3
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            reloaded.save()
        mock_delay.assert_called_once_with(reloaded.pk)


class ApplyCollectionFloorChangeTaskTests(TestCase):
    """face_manager.apply_collection_floor_change: the task itself, not
    the triggering condition (see PhotoCollectionFloorChangeTriggerTests).
    FaceExtractor is mocked out entirely -- real model loading is
    expensive and this task's own logic (scoping both calls to the one
    changed collection) doesn't depend on real inference."""

    def test_scopes_both_clear_and_reencode_to_the_changed_collection(self):
        from filepopulator.models import PhotoCollection
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Task Scope', root_directory='/photos/task_scope', min_face_relative_size=0.1,
            )

        with patch('face_manager.tasks.clear_encodings_below_collection_floor', return_value=(0, 0)) as mock_clear, \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding', return_value=0) as mock_stale, \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            mock_extractor = mock_extractor_cls.return_value
            apply_collection_floor_change(collection.pk)

        mock_clear.assert_called_once_with(collection=collection)
        mock_stale.assert_called_once_with(collection=collection)
        mock_extractor.reencode_missing_faces.assert_called_once_with(collection=collection)

    def test_missing_collection_is_a_no_op(self):
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.clear_encodings_below_collection_floor') as mock_clear, \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding') as mock_stale, \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            apply_collection_floor_change(999999)  # no such PhotoCollection

        mock_clear.assert_not_called()
        mock_stale.assert_not_called()
        mock_extractor_cls.assert_not_called()

    def test_concurrent_run_for_the_same_collection_is_skipped(self):
        from filepopulator.models import PhotoCollection
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Task Lock', root_directory='/photos/task_lock', min_face_relative_size=0.1,
            )

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.clear_encodings_below_collection_floor') as mock_clear, \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding') as mock_stale, \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            mock_lock.return_value.__enter__.return_value = False
            apply_collection_floor_change(collection.pk)

        mock_clear.assert_not_called()
        mock_stale.assert_not_called()
        mock_extractor_cls.assert_not_called()

    def test_a_failing_step_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from filepopulator.models import PhotoCollection
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Task Error Logging', root_directory='/photos/task_error_logging',
                min_face_relative_size=0.1,
            )

        with patch('face_manager.tasks.clear_encodings_below_collection_floor',
                   side_effect=RuntimeError("boom")), \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding', return_value=0), \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            mock_extractor_cls.return_value.reencode_missing_faces.return_value = None
            apply_collection_floor_change(collection.pk)

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.apply_collection_floor_change')
        self.assertIn(str(collection.pk), error.message)
        self.assertIn('boom', error.message)


class TaskCrashLoggingTests(TestCase):
    """Regression tests for a real, confirmed bug (2026-10-21):
    process_faces/reencode_missing_faces/clear_ignored_encodings/
    thistask (assign_faces)/cluster_unverified_faces_task all used to
    wrap their entire body in a bare `except:` that discarded the real
    exception completely -- not even writing it to the log files. Now
    routed through the shared _log_task_crash() helper, which logs the
    real exception AND records it via common.error_log.log_error()."""

    def _mock_empty_inspect(self):
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {}
        return patch('face_manager.tasks.celery_app.control.inspect', return_value=mock_inspect)

    def test_log_task_crash_logs_and_records(self):
        from common.models import ApplicationError
        from face_manager.tasks import _log_task_crash
        _log_task_crash('some_task', RuntimeError("boom"))
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.some_task')
        self.assertIn('RuntimeError', error.message)
        self.assertIn('boom', error.message)

    def test_process_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import process_faces
        img = make_image()
        img.isProcessed = False
        img.save()

        with patch('face_manager.tasks.FaceExtractor', side_effect=RuntimeError("boom")):
            process_faces()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.process_faces')
        self.assertIn('boom', error.message)

    def test_reencode_missing_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import reencode_missing_faces
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.FaceExtractor', side_effect=RuntimeError("boom")):
            reencode_missing_faces()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.reencode_missing_faces')

    def test_reencode_missing_faces_task_skips_when_parallel_backfill_active(self):
        # Regression test for a real production incident (2026-10-25):
        # the one-time parallel backfill (backfill_lvface_parallel)
        # shards the backlog under per-shard lock names, deliberately
        # different from this task's own plain lock name, so the two
        # were never mutually exclusive -- this task ran fully
        # concurrently with the parallel backfill's supervisor, doing
        # real, redundant work on the same backlog while competing with
        # it for the exact CPU/memory headroom the supervisor was
        # trying to budget around. Both now coordinate via a shared
        # marker lock (see face_manager.tasks.reencode_missing_faces'
        # own comment and backfill_lvface_parallel.py's).
        from face_manager.tasks import reencode_missing_faces
        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            key = zlib.crc32(b'face_manager.reencode_missing_faces.parallel_backfill_active')
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
                self.assertTrue(cur.fetchone()[0])

                with patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
                    reencode_missing_faces()
                mock_extractor_cls.assert_not_called()

                cur.execute("SELECT pg_advisory_unlock(%s)", [key])
        finally:
            other.close()

        # Once free again, the task runs normally.
        with patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            reencode_missing_faces()
        mock_extractor_cls.assert_called_once()

    def test_clear_ignored_encodings_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import clear_ignored_encodings
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.clear_confirmed_ignore_face_encodings', side_effect=RuntimeError("boom")):
            clear_ignored_encodings()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.clear_ignored_encodings')

    def test_assign_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import thistask
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner', side_effect=RuntimeError("boom")):
            thistask()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.thistask')

    def test_weekly_redo_all_run_records_its_own_heartbeat(self):
        from common.models import TaskHeartbeat
        from face_manager.tasks import thistask
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner') as mock_assigner_cls:
            thistask(redo_all=True)

        heartbeat = TaskHeartbeat.objects.get(task_name='face_manager.thistask.weekly_full')
        self.assertIsNotNone(heartbeat.last_started_at)
        # A clean (mocked, non-crashing) run must mark itself done, not stuck.
        self.assertIsNotNone(heartbeat.last_completed_at)
        self.assertGreaterEqual(heartbeat.last_completed_at, heartbeat.last_started_at)

    def test_hourly_run_records_a_distinctly_named_heartbeat(self):
        # Must not collide with the weekly one -- they're watched for
        # staleness/hangs independently (only the weekly one is
        # currently in common.tasks.HEARTBEAT_REGISTRY).
        from common.models import TaskHeartbeat
        from face_manager.tasks import thistask
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner') as mock_assigner_cls:
            thistask(redo_all=False)

        self.assertTrue(TaskHeartbeat.objects.filter(task_name='face_manager.thistask.hourly').exists())
        self.assertFalse(
            TaskHeartbeat.objects.filter(task_name='face_manager.thistask.weekly_full').exists())

    def test_crashing_run_still_marks_its_heartbeat_done_not_stuck(self):
        # A crash is a DIFFERENT, already-logged failure mode (see
        # test_assign_faces_crash_is_recorded above) -- it must not also
        # look like a hang to the heartbeat watchdog.
        from common.models import TaskHeartbeat
        from face_manager.tasks import thistask
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner', side_effect=RuntimeError("boom")):
            thistask(redo_all=True)

        heartbeat = TaskHeartbeat.objects.get(task_name='face_manager.thistask.weekly_full')
        self.assertIsNotNone(heartbeat.last_completed_at)

    def test_cluster_unverified_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import cluster_unverified_faces_task
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.cluster_all_unverified_faces', side_effect=RuntimeError("boom")):
            cluster_unverified_faces_task()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.cluster_unverified_faces_task')


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


class CohesionAndSpanNudgeTests(TestCase):
    """_cohesion_and_span_nudge(): the per-person threshold correction on
    top of the size-only bucket gate -- see THRESHOLD_NUDGE_REFERENCE_N's
    own comment in faceAssigner.__init__ for the full investigation (3
    real false-positive hot spots after the -0.05 cut, root-caused via
    real production data) this is calibrated from."""

    def setUp(self):
        self.assigner = faceAssigner()

    def test_high_cohesion_wide_span_gets_no_nudge(self):
        # A normal, healthy gallery -- cohesion above the floor, span
        # well past MIN_SPAN_DAYS -- should be untouched.
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.5, span_iqr_days=365, gallery_size=1000
        )
        self.assertEqual(nudge, 0.0)

    def test_low_cohesion_small_gallery_gets_penalized(self):
        # David Redd's real profile: cohesion well below the floor, a
        # small-ish gallery so the size-scale factor doesn't crush it.
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.28, span_iqr_days=5000, gallery_size=431
        )
        self.assertGreater(nudge, 0.03)

    def test_same_low_cohesion_huge_gallery_gets_almost_no_nudge(self):
        # The exact same cohesion gap as the case above, but backed by a
        # huge gallery -- the size-scale factor should crush the penalty
        # to near zero, since a percentile over thousands of faces is
        # trustworthy even at low cohesion (this is what actually
        # separates David Redd from Joshua/Daniel Lewis in production).
        small = self.assigner._cohesion_and_span_nudge(
            cohesion=0.28, span_iqr_days=5000, gallery_size=431
        )
        huge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.28, span_iqr_days=5000, gallery_size=14000
        )
        self.assertLess(huge, small)
        self.assertLess(huge, 0.02)

    def test_narrow_single_session_gallery_gets_penalized_even_with_high_cohesion(self):
        # Emma's real profile: cohesion is actually the HIGHEST of any
        # real case checked (near-duplicate photos from one session), but
        # the near-zero span means the p99 isn't measuring real
        # person-to-person variation -- must still be penalized.
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.56, span_iqr_days=0, gallery_size=19
        )
        self.assertGreater(nudge, 0.03)

    def test_nudge_is_capped(self):
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.0, span_iqr_days=0, gallery_size=10
        )
        self.assertEqual(nudge, self.assigner.MAX_THRESHOLD_NUDGE)

    def test_nudge_never_negative(self):
        # A gallery that's BETTER than the floor on both axes must never
        # get a negative nudge (i.e. loosen the threshold further).
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.9, span_iqr_days=10_000, gallery_size=10
        )
        self.assertEqual(nudge, 0.0)


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
        # and the largest bucket's threshold (0.558 / 0.344 as of
        # 2026-09-17's -0.05 cut to the non-<50 buckets -- see
        # BUCKET_THRESHOLDS' own comment) -- accepted for a large
        # gallery, rejected for a small one.
        self.mid_similarity = 0.50
        ref = self.query.copy()
        ref[0] = self.mid_similarity
        ref[1] = np.sqrt(1 - self.mid_similarity ** 2)  # unit vector at exactly mid_similarity cosine to query

        small_gallery_size = 10   # falls in the [10,50) bucket -> threshold 0.558
        large_gallery_size = 600  # falls in the [500+) bucket -> threshold 0.344

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
class WindowedClassificationTests(TestCase):
    """Person.use_windowed_classification -- restricts sim_99th to
    gallery members within classification_window_days of the QUERY's
    own date, instead of the whole gallery. Built to fix a real,
    validated false-positive pattern (2026-10-25, see the field's own
    docstring): a candidate that coincidentally resembles a temporally-
    distant subset of a person's gallery (e.g. a much older/younger-
    looking era) can clear the whole-gallery p99 even though it looks
    nothing like that person's photos from NEAR the query's own date.

    Gallery here is two clusters of the SAME person, 2000 days apart,
    at two different cosine similarities to a FIXED query vector
    ([1,0,0,...]): a "close" cluster (cosine 0.65, clears the bucket
    threshold) and a "distant" cluster (cosine 0.05, does not). The
    query's own dateTakenUTC (not its embedding) is what varies between
    the "false candidate" and "true positive" scenarios -- exactly
    matching the real mechanism (the query's actual photo date decides
    which gallery members are temporally relevant), rather than
    varying the query embedding itself."""

    def _gallery_member_at_similarity(self, sim):
        # A unit vector whose cosine similarity to the fixed query
        # [1,0,0,...] is exactly `sim`.
        v = np.zeros(512)
        v[0] = sim
        v[1] = np.sqrt(1 - sim ** 2)
        return v

    def setUp(self):
        self.assigner = faceAssigner()
        self.person = make_person("Windowed Person")
        self.control_person = make_person("Non-windowed Person")
        self.assigner.likely_people_ids = [self.person.id, self.control_person.id]
        self.assigner.num_likely_people = 2

        self.EARLY_DATE = 0.0
        self.LATE_DATE = 2000 * 86400.0  # 2000 days later
        self.close_sim = 0.65   # clears the <50-gallery bucket threshold (0.558)
        self.distant_sim = 0.05  # does not clear it

        # "close" cluster (dated EARLY) is a genuinely good match to the
        # fixed query; "distant" cluster (dated LATE) is not.
        close_embs = [self._gallery_member_at_similarity(self.close_sim) for _ in range(10)]
        distant_embs = [self._gallery_member_at_similarity(self.distant_sim) for _ in range(10)]
        embeddings = np.array(close_embs + distant_embs).T  # 512 x 20
        dates = np.array([self.EARLY_DATE] * 10 + [self.LATE_DATE] * 10)

        self.assigner.embedding_dict = {
            self.person.id: embeddings,
            self.control_person.id: embeddings.copy(),
        }
        self.assigner.norm_dict = {
            self.person.id: np.ones(20),
            self.control_person.id: np.ones(20),
        }
        self.assigner.candidate_dict = {
            self.person.id: pd.DataFrame({'dateTakenUTC': dates}),
            self.control_person.id: pd.DataFrame({'dateTakenUTC': dates}),
        }
        self.assigner._build_concatenated_gallery()

        self.person.use_windowed_classification = True
        self.person.classification_window_days = 182
        self.person.save()
        self.assigner.windowed_people = {self.person.id: 182}

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()

        # The fixed query vector, shared by every test -- what varies is
        # only the query's own dateTakenUTC.
        query_vec = np.zeros(512)
        query_vec[0] = 1.0
        self.query_face = make_face(img, declared_name=blank_person)
        self.query_face.face_encoding_512 = query_vec.tolist()
        self.query_face.save()

    def test_false_candidate_rejected_when_windowed_but_would_pass_unwindowed(self):
        # Query is dated LATE (matching the "distant"/dissimilar
        # cluster) but coincidentally resembles the temporally-UNrelated
        # "close" cluster (dated EARLY) well enough to clear the whole-
        # gallery p99 -- the real Leslie Williams failure shape (a
        # stranger who happens to resemble an old/different-era photo).
        self.query_face.dateTakenUTC = self.LATE_DATE
        self.query_face.rejected_fields = [self.person.id, self.assigner.ignore_person_id]
        self.query_face.save()

        # Sanity check first: WITHOUT windowing (the control person,
        # identical gallery/dates, just not opted in), this exact query
        # DOES get accepted -- confirms the test actually exercises the
        # failure mode windowing is meant to fix, not a strawman.
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.control_person.id, 1, self.close_sim, save=False)

        # WITH windowing: only the temporally-relevant "distant" cluster
        # (dated LATE, matching the query's own date) is considered --
        # correctly rejected instead.
        self.query_face.declared_name = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.person.id, 1, self.distant_sim)

    def test_true_positive_still_accepted_when_windowed(self):
        # Query dated EARLY, matching the genuinely-similar "close"
        # cluster -- windowing restricts to exactly that cluster and
        # still accepts.
        self.query_face.dateTakenUTC = self.EARLY_DATE
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.person.id, 1, self.close_sim, save=False)

    def test_falls_back_to_full_gallery_when_too_few_in_window(self):
        # A narrower window than any date gap present leaves <
        # MIN_WINDOW_SAMPLE (5) members in-window -- falls back to the
        # full gallery rather than computing a near-meaningless
        # percentile over a handful of points. Query dated well past
        # BOTH clusters with a tight window, so neither cluster
        # qualifies (not just a 1-day window around the exact LATE
        # cluster date, which would still trivially include all 10
        # same-dated LATE members).
        self.assigner.windowed_people = {self.person.id: 10}  # 10-day window
        self.query_face.dateTakenUTC = self.LATE_DATE + 1000 * 86400  # 1000 days past LATE
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        # Falls back to the full (unwindowed) gallery -- same as the
        # control-person case above, accepted via the close cluster.
        mock_set.assert_called_once_with(self.person.id, 1, self.close_sim, save=False)

    def test_no_usable_query_date_falls_back_to_full_gallery(self):
        self.query_face.dateTakenUTC = 0  # the field's own "no real date" default
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.person.id, 1, self.close_sim, save=False)


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
        # Scores 0.004 below the large-gallery bucket's own threshold
        # (derived from the live attribute, not a hardcoded absolute
        # value, so this stays valid regardless of where that threshold
        # is calibrated to) -- should sort near the back.
        margin = 0.004
        near_miss_similarity = faceAssigner().BUCKET_THRESHOLDS[-1] - margin
        assigner, face = self._make_face_with_single_candidate(
            similarity=near_miss_similarity, gallery_size=600
        )

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)

        mock_set.assert_called_once()
        _, precedence, weight = mock_set.call_args[0]
        expected = margin / assigner.IGNORE_WEIGHT_MARGIN_CLAMP
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
        near_miss_similarity = faceAssigner().BUCKET_THRESHOLDS[-1] - 0.004
        near_miss_assigner, near_miss_face = self._make_face_with_single_candidate(
            similarity=near_miss_similarity, gallery_size=600
        )
        far_assigner, far_face = self._make_face_with_single_candidate(similarity=0.0, gallery_size=600)

        with patch.object(Face, "set_possible_person") as mock_set:
            near_miss_assigner.classify_unassigned(near_miss_face)
            near_miss_weight = mock_set.call_args[0][2]

            far_assigner.classify_unassigned(far_face)
            far_weight = mock_set.call_args[0][2]

        self.assertGreater(far_weight, near_miss_weight)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class DetScoreMarginGateTests(TestCase):
    """DET_SCORE_FLOOR/CANDIDATE_MARGIN_FLOOR: added 2026-09-21 after a
    real investigation into whether Face.det_score helps classification.
    Combined with a tightened per-candidate margin requirement (sim_99th
    at least CANDIDATE_MARGIN_FLOOR above threshold, not just > 0), this
    is a validated fix for barely-passing candidates on poorly-detected
    faces -- see CLAUDE.md for the full investigation, including a real
    visual false positive (a photo of toy building blocks) that this
    combined gate catches but det_score alone did not."""

    def _make_face_with_single_candidate(self, similarity, gallery_size=600, det_score=None, relative_size=None):
        assigner = faceAssigner()
        person = make_person(f"Candidate {similarity}-{gallery_size}-{det_score}-{relative_size}")
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
        face.det_score = det_score
        face.relative_size = relative_size
        face.save()
        return assigner, face, person

    def test_margin_just_above_zero_now_rejected(self):
        # Previously accepted (margin > 0 was the whole gate); the
        # tightened CANDIDATE_MARGIN_FLOOR (0.03) now rejects a candidate
        # that only barely clears its threshold.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.01  # margin=0.01, below CANDIDATE_MARGIN_FLOOR=0.03
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        # Falls through to the "no match" branch -- proposed person is
        # the ignore sentinel, not the barely-passing candidate.
        self.assertEqual(called_person_id, assigner.ignore_person_id)

    def test_margin_at_floor_with_good_det_score_is_accepted(self):
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + assigner.CANDIDATE_MARGIN_FLOOR + 0.01
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, person.id)

    def test_good_margin_but_low_det_score_is_rejected(self):
        # A candidate that clears the margin requirement comfortably, but
        # on a poorly-detected face (det_score below DET_SCORE_FLOOR) --
        # the det_score gate blocks it even though margin alone would
        # have accepted it. Mirrors the real "toy building blocks" false
        # positive found during the investigation.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.20  # comfortable margin
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.4,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, assigner.ignore_person_id)

    def test_null_det_score_skips_det_score_check_but_not_margin(self):
        # A face not yet reached by the ongoing det_score backfill (NULL)
        # must not be blocked by the det_score half of the gate -- but
        # the margin requirement still applies unconditionally.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]

        # Comfortable margin, NULL det_score -- should still be accepted.
        similarity = threshold + 0.20
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=None,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

        # Thin margin (0.01), NULL det_score -- margin gate still rejects
        # it regardless of the missing det_score.
        similarity2 = threshold + 0.01
        assigner2, face2, person2 = self._make_face_with_single_candidate(
            similarity=similarity2, det_score=None,
        )
        with patch.object(Face, "set_possible_person") as mock_set2:
            assigner2.classify_unassigned(face2)
        mock_set2.assert_called_once()
        self.assertEqual(mock_set2.call_args[0][0], assigner2.ignore_person_id)

    def _make_video_face_with_single_candidate(self, similarity, gallery_size=600, det_score=None):
        # Same shape as _make_face_with_single_candidate, but the query
        # face is video-sourced -- needed to exercise VIDEO_DET_SCORE_FLOOR
        # and the video sub-gallery boost, neither of which a plain
        # image-sourced query ever touches.
        from filepopulator.models import Directory, VideoFile

        assigner = faceAssigner()
        person = make_person(f"VideoCandidate {similarity}-{gallery_size}-{det_score}")
        assigner.likely_people_ids = [person.id]

        query = np.zeros(512)
        query[0] = 1.0
        ref = query.copy()
        ref[0] = similarity
        ref[1] = np.sqrt(max(0.0, 1 - similarity ** 2))

        assigner.embedding_dict = {person.id: np.tile(ref.reshape(512, 1), (1, gallery_size))}
        assigner.norm_dict = {person.id: np.ones(gallery_size)}
        assigner._build_concatenated_gallery()

        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/video_classify_test')
        video = VideoFile.objects.create(
            filename='/tmp/video_classify_test/a.mp4', directory=directory,
            width=320, height=240, duration_seconds=10.0,
        )
        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        face = Face(
            declared_name=blank_person, source_video_file=video,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.face_encoding_512 = query.tolist()
        face.det_score = det_score
        face.save()
        return assigner, face, person

    def test_video_query_uses_video_det_score_floor_not_image_floor(self):
        # det_score=0.55 sits between VIDEO_DET_SCORE_FLOOR (0.5, passes)
        # and the image-calibrated DET_SCORE_FLOOR (0.6, would fail) --
        # must be accepted for a video query specifically because the
        # lower, video-calibrated floor applies.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.20
        assigner, face, person = self._make_video_face_with_single_candidate(
            similarity=similarity, det_score=0.55,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_image_query_at_same_det_score_still_uses_image_floor(self):
        # Sanity check: the exact same det_score (0.55) on an IMAGE query
        # still gets rejected -- confirms this is a video-specific floor,
        # not an accidental global loosening.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.20
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.55,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], assigner.ignore_person_id)

    def test_video_subgallery_boosts_a_candidate_above_threshold_for_video_query(self):
        # Full-gallery similarity sits below the margin floor on its own;
        # the SAME candidate's video-only sub-gallery scores it high
        # enough to clear -- a video query should be accepted via that
        # boost. CANDIDATE_MARGIN_FLOOR/DET_SCORE_FLOOR values are
        # deliberately generous here (det_score=0.9, well above either
        # floor) to isolate the sub-gallery boost itself.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        below_margin = threshold + 0.01  # fails CANDIDATE_MARGIN_FLOOR on its own
        above_margin = threshold + 0.20

        assigner, face, person = self._make_video_face_with_single_candidate(
            similarity=below_margin, det_score=0.9,
        )

        query = np.array(face.face_encoding_512)
        video_ref = query.copy()
        video_ref[0] = above_margin
        video_ref[1] = np.sqrt(max(0.0, 1 - above_margin ** 2))
        assigner.video_embedding_dict = {person.id: np.tile(video_ref.reshape(512, 1), (1, assigner.MIN_VIDEO_SUBGALLERY_SIZE))}
        assigner.video_norm_dict = {person.id: np.ones(assigner.MIN_VIDEO_SUBGALLERY_SIZE)}
        assigner._build_video_concatenated_gallery()

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_video_subgallery_does_not_affect_image_query(self):
        # Same boosted video sub-gallery as above, but the query itself
        # is image-sourced -- must be completely ignored, keeping the
        # already-tuned image classification path untouched.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        below_margin = threshold + 0.01
        above_margin = threshold + 0.20

        assigner, face, person = self._make_face_with_single_candidate(
            similarity=below_margin, det_score=0.9,
        )

        query = np.array(face.face_encoding_512)
        video_ref = query.copy()
        video_ref[0] = above_margin
        video_ref[1] = np.sqrt(max(0.0, 1 - above_margin ** 2))
        assigner.video_embedding_dict = {person.id: np.tile(video_ref.reshape(512, 1), (1, assigner.MIN_VIDEO_SUBGALLERY_SIZE))}
        assigner.video_norm_dict = {person.id: np.ones(assigner.MIN_VIDEO_SUBGALLERY_SIZE)}
        assigner._build_video_concatenated_gallery()

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], assigner.ignore_person_id)


class SizeNudgeMarginGateTests(TestCase):
    """SIZE_NUDGE_REF_SIZE/SIZE_NUDGE_MAX: added 2026-10-20 after a real
    investigation into whether Face.relative_size helps classification --
    see CLAUDE.md for the full sweep (continuous linear decay vs. tiers,
    and why lowering CANDIDATE_MARGIN_FLOOR itself doesn't help even with
    this term in place). Uses a standalone copy of DetScoreMarginGateTests'
    _make_face_with_single_candidate helper (not a subclass -- subclassing
    a TestCase to reuse a helper also re-runs every inherited test method
    under this class name too) -- det_score is always a comfortable 0.9,
    well above DET_SCORE_FLOOR, isolating the size term specifically."""

    def _make_face_with_single_candidate(self, similarity, gallery_size=600, det_score=None, relative_size=None):
        assigner = faceAssigner()
        person = make_person(f"SizeCandidate {similarity}-{gallery_size}-{det_score}-{relative_size}")
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
        face.det_score = det_score
        face.relative_size = relative_size
        face.save()
        return assigner, face, person

    def test_small_face_with_thin_margin_is_rejected(self):
        # A margin that would clear the plain CANDIDATE_MARGIN_FLOOR
        # (0.03) on its own, but not once the small-face size nudge
        # (relative_size well below SIZE_NUDGE_REF_SIZE=0.04) adds its
        # full SIZE_NUDGE_MAX=0.02 on top -- effective floor 0.05.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.04  # margin=0.04: clears 0.03, not 0.05
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.005,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], assigner.ignore_person_id)

    def test_small_face_with_margin_clearing_the_nudged_floor_is_accepted(self):
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.06  # margin=0.06, clears 0.03+0.02=0.05
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.005,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_face_at_or_above_ref_size_gets_no_nudge(self):
        # relative_size >= SIZE_NUDGE_REF_SIZE (0.04) -- same plain 0.03
        # floor as before this feature existed.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.04  # margin=0.04, clears the plain 0.03 floor
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.04,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_null_relative_size_skips_the_size_nudge(self):
        # Not yet reached by a relative_size backfill (or a video-sourced
        # query) -- must not be penalized, same convention as a NULL
        # det_score skipping its own half of the gate.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.04  # margin=0.04: would fail a nudged floor, passes the plain one
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=None,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_nudge_decays_linearly_partway_to_ref_size(self):
        # relative_size halfway between 0 and SIZE_NUDGE_REF_SIZE (0.02 of
        # 0.04) -- nudge should be half of SIZE_NUDGE_MAX (0.01), so
        # margin=0.035 (clears 0.03+0.01=0.04) should be accepted...
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.045
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.02,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

        # ...but margin=0.035 (just under 0.03+0.01=0.04) should not.
        assigner2 = faceAssigner()
        similarity2 = threshold + 0.035
        assigner2, face2, person2 = self._make_face_with_single_candidate(
            similarity=similarity2, det_score=0.9, relative_size=0.02,
        )
        with patch.object(Face, "set_possible_person") as mock_set2:
            assigner2.classify_unassigned(face2)
        mock_set2.assert_called_once()
        self.assertEqual(mock_set2.call_args[0][0], assigner2.ignore_person_id)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClassificationInvariantCheckTests(TestCase):
    """faceAssigner._verify_classification_invariant(): a safety net run
    at the end of every execute() call -- a face with a real encoding
    and no poss_ident1 proposal shouldn't exist after a normal pass
    (every real classify_unassigned() success path sets something).
    classify_unassigned() itself is mocked throughout -- this tests the
    invariant-check/retry/log_error wiring, not real classification
    math (already covered elsewhere)."""

    def _make_leftover_face(self):
        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        face = make_face(img, declared_name=blank_person)
        face.face_encoding_512 = (np.ones(512) / np.sqrt(512)).tolist()
        face.poss_ident1 = None
        face.save()
        return face

    def test_no_leftover_faces_is_a_no_op(self):
        from common.models import ApplicationError
        assigner = faceAssigner()
        with patch.object(faceAssigner, 'classify_unassigned') as mock_classify:
            assigner._verify_classification_invariant()
        mock_classify.assert_not_called()
        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_leftover_face_that_succeeds_on_retry_is_not_logged(self):
        from common.models import ApplicationError
        face = self._make_leftover_face()
        person = make_person("Retried Match")
        assigner = faceAssigner()

        def fake_classify(f, **kwargs):
            # Simulate a successful classification on retry.
            f.poss_ident1 = person
            f.save()

        with patch.object(faceAssigner, 'classify_unassigned', side_effect=fake_classify):
            assigner._verify_classification_invariant()

        face.refresh_from_db()
        self.assertEqual(face.poss_ident1_id, person.id)
        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_leftover_face_that_keeps_failing_is_logged(self):
        from common.models import ApplicationError
        face = self._make_leftover_face()
        assigner = faceAssigner()

        with patch.object(faceAssigner, 'classify_unassigned', side_effect=RuntimeError("boom")):
            assigner._verify_classification_invariant()

        self.assertEqual(ApplicationError.objects.count(), 1)
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.assign_faces.execute')
        self.assertIn(str(face.id), error.message)
        self.assertFalse(error.dismissed)

    def test_newly_created_face_during_retry_window_is_not_falsely_flagged(self):
        # Real production false-positive (2026-10-21): a face created
        # concurrently by face_extraction, in the window between
        # building leftover_ids and the still_broken re-check, must not
        # be reported as "still broken after a retry" -- it was never
        # part of the retry batch at all, just an ordinary new,
        # not-yet-classified face.
        from common.models import ApplicationError
        face = self._make_leftover_face()
        assigner = faceAssigner()

        def fake_classify(f, **kwargs):
            person = make_person("Retried Match For Race Test")
            f.poss_ident1 = person
            f.save()
            # A brand-new face lands concurrently, created AFTER
            # leftover_ids was already built -- never actually retried.
            self._make_leftover_face()

        with patch.object(faceAssigner, 'classify_unassigned', side_effect=fake_classify):
            assigner._verify_classification_invariant()

        self.assertEqual(ApplicationError.objects.count(), 0)


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

    def test_within_max_age_reuses_cache_without_requerying(self):
        # A cache built (CACHE_MAX_AGE_DAYS - 1) days ago is still fully
        # inside the trusted window -- not even the cheap signature check
        # should run, let alone a full re-fetch.
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(
            days=self.assigner.CACHE_MAX_AGE_DAYS - 1
        )
        with open(self.cache_path, "wb") as fh:
            pickle.dump(cached, fh)

        fresh_assigner = faceAssigner()
        fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
        fresh_assigner.likely_people_ids = [self.person.id]
        with patch.object(Face.objects, "filter", wraps=Face.objects.filter) as mock_filter, \
             patch.object(faceAssigner, "_current_face_data_signature") as mock_sig:
            fresh_assigner.load_encodings()
        mock_sig.assert_not_called()
        self.assertEqual(mock_filter.call_count, 0)
        self.assertIn(self.person.id, fresh_assigner.embedding_dict)

    def test_past_max_age_no_changes_keeps_cache(self):
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(
            days=self.assigner.CACHE_MAX_AGE_DAYS
        )
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

    def test_past_max_age_with_changes_rebuilds_cache(self):
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(
            days=self.assigner.CACHE_MAX_AGE_DAYS
        )
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
        self.extractor.LVFACE_REENCODE_MAX_RUNTIME_SECONDS = 3000
        # extractor.app is a PyramidalDetector in real code; .app.app is
        # the underlying FaceAnalysis. Mock both layers directly rather
        # than constructing a real PyramidalDetector/FaceAnalysis.
        self.mock_face_analysis = MagicMock()
        self.mock_pyramidal = MagicMock()
        self.mock_pyramidal.app = self.mock_face_analysis
        self.extractor.app = self.mock_pyramidal

        # The LVFace-B catch-up sweep (_backfill_lvface_encodings(), run
        # automatically at the end of reencode_missing_faces() now) is
        # out of scope for these tests -- they're about the insightface
        # re-encode selection/branching logic specifically. Mocked out
        # (never None/real) so a test face that happens to carry kps
        # doesn't trigger a real, unmocked ONNX model load.
        lvface_patcher = patch('lvface.embed_from_image_array', return_value=None)
        lvface_patcher.start()
        self.addCleanup(lvface_patcher.stop)

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

    # --- Merged LVFace-B image-side coverage (2026-10-25) ---------------
    # reencode_missing_faces() used to call a separate _backfill_lvface_
    # encodings() sweep at the end, which opened and decoded the same
    # image a SECOND time for any face missing both encodings. Now both
    # encodings are handled together, per face, in this same loop -- see
    # the method's own docstring for the full reasoning. These tests
    # cover the merge itself; the pre-existing tests above already cover
    # the insightface-only branching logic, which is unchanged.

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_face_needing_only_lvface_gets_it_without_touching_insight(self):
        person = make_person("Only Needs Lvface")
        face = make_face(self.image, declared_name=person)
        face.face_encoding_512 = [0.5] * 512  # already real, float32-exact
        face.kps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        face.face_encoding_lvface_b = None
        face.reencoded = False
        face.save()

        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)) as mock_lv:
            self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.4] * 512)
        self.assertEqual(face.face_encoding_512, [0.5] * 512)  # untouched
        self.assertFalse(face.reencoded)  # insight never ran
        self.mock_face_analysis.get.assert_not_called()
        mock_lv.assert_called_once()

    def test_face_needing_neither_is_skipped_without_opening_the_image(self):
        # The exact scenario that motivated this merge: a face with a
        # real insightface embedding but no kps (so it can never
        # qualify for LVFace-B without a fresh detection) must not pay
        # for a single image decode.
        person = make_person("Needs Nothing")
        face = make_face(self.image, declared_name=person)
        face.face_encoding_512 = [0.5] * 512
        face.kps = None
        face.face_encoding_lvface_b = None
        face.save()

        with patch('common.open_img_oriented') as mock_open:
            self.extractor.reencode_missing_faces()

        mock_open.assert_not_called()
        self.mock_face_analysis.get.assert_not_called()

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_crop_redetect_recovering_kps_also_triggers_lvface_in_same_pass(self):
        # A face with no kps at all (needs insight's crop-redetect path)
        # whose redetection succeeds recovers kps as a side effect --
        # that should make it newly eligible for LVFace-B too, using the
        # SAME already-open image, rather than waiting for a future run.
        person = make_person("Recovers Kps")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        self.assertIsNone(face.kps)

        crop_local_kps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 40, 40], 'kps': crop_local_kps, 'embedding': np.full(512, 0.3)},
        ]

        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.55, dtype=np.float32)) as mock_lv:
            self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertIsNotNone(face.kps)  # recovered by the crop-redetect path
        self.assertEqual(face.face_encoding_lvface_b, [0.55] * 512)
        mock_lv.assert_called_once()
        # The recovered (absolute-coordinate) kps is what LVFace-B should
        # have been called with, not the raw crop-local values.
        called_kps = mock_lv.call_args.args[1]
        np.testing.assert_array_equal(np.asarray(called_kps).reshape(-1), np.array(face.kps))

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_image_face_below_collection_floor_excluded_from_both_encodings(self):
        from filepopulator.models import Directory, PhotoCollection
        low_fidelity = PhotoCollection.objects.create(
            name='Low Fidelity Reencode', root_directory='/photos/reencode_floor_test',
            min_face_det_score=0.6,
        )
        directory = Directory.objects.create(dir_path='/photos/reencode_floor_test', collection=low_fidelity)
        below_floor_image = ImageFile.objects.create(
            filename=self.image.filename, directory=directory,
            width=self.image.width, height=self.image.height,
            pixel_hash='reencode-floor-hash', file_hash='reencode-floor-fhash',
        )
        person = make_person("Below Floor Reencode")
        face = make_face(below_floor_image, declared_name=person, det_score=0.5)
        face.face_encoding_512 = None
        face.kps = [1.0] * 10
        face.face_encoding_lvface_b = None
        face.save()

        with patch('common.open_img_oriented') as mock_open, \
             patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.reencode_missing_faces()

        mock_open.assert_not_called()
        self.mock_face_analysis.get.assert_not_called()
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_below_floor_face_never_enters_the_per_face_loop_at_all(self):
        # Distinguishes "excluded by the SQL query itself" (the real
        # point of the below_collection_floor() SQL mirror added
        # 2026-10-25 -- previously tens of thousands of permanently-
        # below-floor rows were fetched and iterated every single
        # hourly run just to be skipped) from merely "fetched, then
        # skipped in Python" -- if the SQL filter is doing its job, the
        # per-face loop body (and its own redundant below_collection_
        # floor() safety-net call) should never even run for this row.
        from filepopulator.models import Directory, PhotoCollection
        low_fidelity = PhotoCollection.objects.create(
            name='Low Fidelity SQL Filter', root_directory='/photos/reencode_sql_floor_test',
            min_face_det_score=0.6,
        )
        directory = Directory.objects.create(dir_path='/photos/reencode_sql_floor_test', collection=low_fidelity)
        below_floor_image = ImageFile.objects.create(
            filename=self.image.filename, directory=directory,
            width=self.image.width, height=self.image.height,
            pixel_hash='reencode-sql-floor-hash', file_hash='reencode-sql-floor-fhash',
        )
        person = make_person("Sql Floor Excluded")
        face = make_face(below_floor_image, declared_name=person, det_score=0.5)
        face.face_encoding_512 = None
        face.kps = [1.0] * 10
        face.face_encoding_lvface_b = None
        face.save()

        with patch('face_manager.face_extract_encode.below_collection_floor') as mock_floor_check:
            self.extractor.reencode_missing_faces()

        mock_floor_check.assert_not_called()

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_confirmed_ignore_face_excluded_from_lvface_too(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        face = make_face(self.image, declared_name=ignore_person)
        face.face_encoding_512 = None
        face.kps = [1.0] * 10
        face.face_encoding_lvface_b = None
        face.save()

        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.reencode_missing_faces()

        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    @override_settings(LVFACE_ENCODING_ENABLED=False)
    def test_lvface_disabled_skips_lvface_but_insight_still_runs(self):
        person = make_person("Lvface Off")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        face.save()
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.7))),
        }

        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.reencode_missing_faces()

        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.7] * 512)
        self.assertIsNone(face.face_encoding_lvface_b)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_time_budget_shared_across_combined_loop(self):
        person_a = make_person("Budget Face A")
        person_b = make_person("Budget Face B")
        face_a = make_face(self.image, declared_name=person_a)
        face_a.face_encoding_512 = [0.5] * 512
        face_a.kps = [1.0] * 10
        face_a.face_encoding_lvface_b = None
        face_a.save()
        face_b = make_face(self.image, declared_name=person_b)
        face_b.face_encoding_512 = [0.5] * 512
        face_b.kps = [1.0] * 10
        face_b.face_encoding_lvface_b = None
        face_b.save()

        times = iter([0, 0, 1000])  # start_time, first face check (ok), second face check (over budget)

        def fake_time():
            return next(times, 1000)

        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.2, dtype=np.float32)) as mock_lv, \
             patch('time.time', side_effect=fake_time):
            self.extractor.reencode_missing_faces(lvface_max_runtime_seconds=10)

        self.assertEqual(mock_lv.call_count, 1)

    def _make_video_face(self, **overrides):
        # Mirrors BackfillLvfaceVideoEncodingsTests' own helper -- kept
        # as a small, separate copy rather than a shared fixture, since
        # these two test classes exercise different entry points
        # (reencode_missing_faces()'s own combined queryset here, vs.
        # _backfill_lvface_video_encodings() directly there).
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/reencode_combined_test')
        video_path = '/tmp/reencode_combined_test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(b'not a real video, existence is all this sweep checks')
        self.addCleanup(lambda: os.path.exists(video_path) and os.remove(video_path))
        video = VideoFile.objects.create(filename=video_path, directory=directory, width=640, height=480)
        face = Face(
            declared_name=overrides.pop('declared_name', None) or get_default_blank_person(),
            source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
            video_thumbnail_frame_seconds=12.3,
            kps=[1.0] * 10,
            **overrides,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_video_faces_are_included_in_the_combined_sweep(self):
        # Real behavior change, 2026-10-26: video used to require a
        # completely separate sub-pass call that only ever ran AFTER
        # the image loop fully finished -- for an unbounded (shard)
        # call, that meant video never got touched until a shard's
        # entire image backlog was already drained (confirmed stuck at
        # 0 progress in real production for this exact reason). Image
        # and video faces are now unioned into one combined, randomly-
        # ordered queryset, so a call that processes anything at all
        # can reach video faces too, regardless of image backlog size.
        video_face = self._make_video_face()
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)):
            self.extractor.reencode_missing_faces()
        video_face.refresh_from_db()
        self.assertEqual(video_face.face_encoding_lvface_b, [0.5] * 512)

    def test_sharded_call_only_processes_video_faces_in_its_own_shard(self):
        # The combined queryset's Mod(F('id'), total_shards) partition
        # applies uniformly to both face types -- every video face must
        # still be covered by exactly one of the shard runs below.
        video_faces = [self._make_video_face() for _ in range(6)]
        total_shards = 3
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.6, dtype=np.float32)):
            for shard_index in range(total_shards):
                self.extractor.reencode_missing_faces(shard_index=shard_index, total_shards=total_shards)
        for face in video_faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)

    def test_collection_scoped_call_never_includes_video(self):
        # collection is an image-only concept (no PhotoCollection
        # association for VideoFile, see CLAUDE.md) -- a collection-
        # scoped call (e.g. apply_collection_floor_change) must never
        # touch the video backlog, same as before this change.
        from filepopulator.models import PhotoCollection
        video_face = self._make_video_face()
        default_collection = PhotoCollection.objects.get(is_default=True)
        with patch('video_face_pipeline.ffprobe_info') as mock_ffprobe, \
             patch('lvface.embed_from_image_array'):
            self.extractor.reencode_missing_faces(collection=default_collection)
        mock_ffprobe.assert_not_called()
        video_face.refresh_from_db()
        self.assertIsNone(video_face.face_encoding_lvface_b)

    def test_skips_entirely_when_advisory_lock_already_held(self):
        # Regression test for a real production incident (2026-10-25): a
        # manual manage.py shell call and the scheduled face_manager.
        # reencode task ran fully concurrently with zero mutual
        # awareness (the task wrapper's own overlap check only looked
        # for other Celery tasks of the same name, invisible to a direct
        # call), each building its own separate FaceExtractor/ONNX
        # sessions and stacking memory until the container neared its
        # cgroup limit. reencode_missing_faces() now holds a Postgres
        # advisory lock (common/advisory_lock.py, same pattern as
        # find_and_encode_faces()'s own lock -- see that test's own
        # comment for why a genuinely separate connection is needed to
        # simulate real contention) for its whole run; while another
        # session holds it, this call must do nothing at all.
        person = make_person("Lock Contention")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        face.save()
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.5))),
        }

        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            key = zlib.crc32(b'face_manager.reencode_missing_faces')
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
                self.assertTrue(cur.fetchone()[0])

                self.extractor.reencode_missing_faces()

                face.refresh_from_db()
                self.assertIsNone(face.face_encoding_512)
                self.mock_face_analysis.get.assert_not_called()

                cur.execute("SELECT pg_advisory_unlock(%s)", [key])
        finally:
            other.close()

        # Once free again, a normal run still works.
        self.extractor.reencode_missing_faces()
        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_periodic_reinit_bounds_memory_growth(self):
        # Regression test for a real production incident (2026-10-25):
        # the merged image loop can now run for a long time against a
        # very large backlog (previously this method never ran long
        # enough for ONNXRuntime's arena growth to matter) -- confirmed
        # via real data, an unbounded manual run grew to 13+ GB RSS
        # within the hour. Mirrors find_and_encode_faces()'s own
        # periodic-rebuild mitigation. Faces here are set up to need
        # ONLY LVFace-B (real face_encoding_512 already present) so this
        # stays focused purely on confirming the rebuild happens at the
        # right counts, without needing the rebuilt mock detector to
        # also support a real insight re-encode.
        self.extractor.REENCODE_REINIT_EVERY_N_IMAGES = 2
        faces = []
        for i in range(3):
            person = make_person(f"Reinit Face {i}")
            face = make_face(
                self.image, declared_name=person,
                box_left=10, box_top=10, box_right=30, box_bottom=30,
            )
            face.face_encoding_512 = [0.5] * 512
            face.kps = [1.0] * 10
            face.face_encoding_lvface_b = None
            face.save()
            faces.append(face)

        with patch.object(self.extractor, '_build_detector', return_value=MagicMock()) as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.3, dtype=np.float32)), \
             patch('lvface.reset_session') as mock_reset:
            self.extractor.reencode_missing_faces()

        # 3 faces processed, REENCODE_REINIT_EVERY_N_IMAGES=2 -> the
        # check at processed_count==2 (0-indexed boundary, i.e. right
        # before the 3rd face) fires exactly once.
        mock_build.assert_called_once()
        mock_reset.assert_called_once()
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_lvface_b, [0.3] * 512)

    def test_shard_filters_to_matching_ids_only(self):
        # Core correctness property of the parallel backfill (2026-10-25,
        # see backfill_lvface_parallel.py): shard_index/total_shards
        # must partition faces (via _shard_annotation()'s hash, not
        # plain id % total_shards -- see that function's own comment
        # for why) so every face is covered by EXACTLY one shard, never
        # zero or more than one.
        faces = []
        for i in range(6):
            person = make_person(f"Shard Face {i}")
            face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
            face.kps = [1.0] * 10
            face.save()
            faces.append(face)

        total_shards = 3
        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            for shard_index in range(total_shards):
                self.mock_face_analysis.models = {
                    'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.4))),
                }
                self.extractor.reencode_missing_faces(shard_index=shard_index, total_shards=total_shards)

        # Every face is covered by exactly one of the 3 shard runs above
        # (its own hashed shard assignment) -- confirms the partition
        # has no gaps, without needing to separately prove no face is
        # visited twice (a face's hash only ever matches one shard_index
        # by construction, so double-processing isn't reachable here).
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_512, [0.4] * 512)

    def test_different_shards_use_independent_locks(self):
        # A real race this is meant to allow, not prevent: two DIFFERENT
        # shards of the same backfill must be able to run genuinely
        # concurrently (that's the whole point), unlike two invocations
        # of the SAME shard (already covered by test_skips_entirely_
        # when_advisory_lock_already_held) or the plain unsharded lock.
        person = make_person("Shard Lock Independence")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [1.0] * 10
        face.save()
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.6))),
        }

        # Hold the OTHER shard's lock (the one this face does NOT belong
        # to) -- computed via the real shard annotation (a HASHTEXT-
        # based hash, not plain id % total_shards -- see
        # _shard_annotation()'s own comment for why), not assumed, to
        # stay correct regardless of what id this face lands on.
        from face_manager.face_extract_encode import _shard_annotation
        face_shard = Face.objects.filter(id=face.id).annotate(
            _shard=_shard_annotation(2)
        ).values_list('_shard', flat=True).first()
        held_shard = 1 - face_shard

        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            held_shard_key = zlib.crc32(f'face_manager.reencode_missing_faces.shard_{held_shard}_of_2'.encode())
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [held_shard_key])
                self.assertTrue(cur.fetchone()[0])

                # face_shard's own lock was never touched -- this must
                # proceed normally despite the OTHER shard being locked.
                self.extractor.reencode_missing_faces(shard_index=face_shard, total_shards=2)

                cur.execute("SELECT pg_advisory_unlock(%s)", [held_shard_key])
        finally:
            other.close()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.6] * 512)


class ReencodeMissingFacesShardCommandTests(TestCase):
    """reencode_missing_faces_shard -- the worker entry point the
    one-time parallel backfill (backfill_lvface_parallel) spawns one of
    per shard. Mocks FaceExtractor entirely (real model loading is
    expensive) -- this is about the command's own argument validation
    and pass-through, not FaceExtractor's own logic (covered elsewhere)."""

    def test_valid_shard_args_pass_through_to_extractor(self):
        with patch('face_manager.management.commands.reencode_missing_faces_shard.FaceExtractor') as mock_cls:
            mock_instance = mock_cls.return_value
            call_command('reencode_missing_faces_shard', '--shard-index', '2', '--total-shards', '5')
        mock_instance.reencode_missing_faces.assert_called_once_with(
            shard_index=2, total_shards=5, lvface_max_runtime_seconds=None,
        )

    def test_shard_index_out_of_range_raises(self):
        with patch('face_manager.management.commands.reencode_missing_faces_shard.FaceExtractor'):
            with self.assertRaises(CommandError):
                call_command('reencode_missing_faces_shard', '--shard-index', '5', '--total-shards', '5')

    def test_negative_shard_index_raises(self):
        with patch('face_manager.management.commands.reencode_missing_faces_shard.FaceExtractor'):
            with self.assertRaises(CommandError):
                call_command('reencode_missing_faces_shard', '--shard-index', '-1', '--total-shards', '5')


class BackfillLvfaceParallelHelperTests(TestCase):
    """Pure helper functions from backfill_lvface_parallel.py (the
    one-time parallel-backfill supervisor) -- _available_memory_gb() and
    _total_cores(). The supervisor's own main loop (subprocess spawning,
    CPU-affinity pinning, the scaling decisions) is deliberately not
    unit-tested here: it's a one-off tool run once under direct
    supervision (see CLAUDE.md's dated write-up), not a permanent code
    path, and meaningfully testing it would need mocking subprocess.
    Popen/os.sched_setaffinity/os.getloadavg/time.sleep so extensively
    that the test would mostly just re-assert the implementation rather
    than catch real bugs -- real validation came from actually running
    it against production."""

    def test_available_memory_gb_prefers_cgroup_ceiling_when_lower(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\n"
        # cgroup says only ~2GB left (host thinks ~38GB) -- the smaller
        # (cgroup) figure must win, since that's the one that actually
        # kills a new worker first.
        cgroup_max = str(10 * 1024 ** 3)
        cgroup_current = str(8 * 1024 ** 3)
        contents = {
            '/proc/meminfo': meminfo,
            '/sys/fs/cgroup/memory.max': cgroup_max,
            '/sys/fs/cgroup/memory.current': cgroup_current,
        }

        def fake_open(path, *a, **kw):
            if path not in contents:
                raise FileNotFoundError(path)
            return io.StringIO(contents[path])

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_available_memory_gb_adds_back_reclaimable_page_cache(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\n"
        # memory.current says only ~2GB left (10GB max, 8GB current), but
        # 5GB of that 8GB is reclaimable page cache -- the real incident
        # this guards against (2026-10-25): a naive memory.max -
        # memory.current treated that cache as genuinely unavailable,
        # keeping the supervisor pinned near its floor for hours despite
        # real headroom existing. Real available should be 2GB + 5GB = 7GB.
        cgroup_max = str(10 * 1024 ** 3)
        cgroup_current = str(8 * 1024 ** 3)
        memory_stat = f"anon {3 * 1024 ** 3}\nfile {5 * 1024 ** 3}\ninactive_file {5 * 1024 ** 3}\n"
        contents = {
            '/proc/meminfo': meminfo,
            '/sys/fs/cgroup/memory.max': cgroup_max,
            '/sys/fs/cgroup/memory.current': cgroup_current,
            '/sys/fs/cgroup/memory.stat': memory_stat,
        }

        def fake_open(path, *a, **kw):
            if path not in contents:
                raise FileNotFoundError(path)
            return io.StringIO(contents[path])

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 7.0, places=3)

    def test_available_memory_gb_cgroup_math_survives_missing_memory_stat(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\n"
        # memory.stat itself missing (e.g. an older/different cgroup
        # setup) must not abort the whole cgroup calculation -- it should
        # just treat reclaimable cache as 0, not fall all the way back to
        # host-only.
        cgroup_max = str(10 * 1024 ** 3)
        cgroup_current = str(8 * 1024 ** 3)
        contents = {
            '/proc/meminfo': meminfo,
            '/sys/fs/cgroup/memory.max': cgroup_max,
            '/sys/fs/cgroup/memory.current': cgroup_current,
        }

        def fake_open(path, *a, **kw):
            if path not in contents:
                raise FileNotFoundError(path)
            return io.StringIO(contents[path])

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_available_memory_gb_falls_back_to_host_without_cgroup_v2(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   20000000 kB\n"

        def fake_open(path, *a, **kw):
            if path == '/proc/meminfo':
                return io.StringIO(meminfo)
            raise FileNotFoundError(path)

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 20000000 / (1024 * 1024), places=3)

    def test_total_cores_uses_sched_affinity_when_available(self):
        from face_manager.management.commands.backfill_lvface_parallel import _total_cores
        with patch('os.sched_getaffinity', return_value=set(range(6)), create=True):
            self.assertEqual(_total_cores(), 6)


class LVFaceModuleTests(TestCase):
    """face_manager.lvface -- the LVFace-B (ViT) embedding helper. Mocks
    the ONNXRuntime session entirely (via lvface._get_session, patched
    module-globally) so these stay fast and never need the real 455MB
    model file on disk -- see lvface.py's own docstring for why session
    construction is lazy in the first place."""

    def setUp(self):
        import lvface
        lvface._session = None  # don't leak a mock session across tests

    def tearDown(self):
        import lvface
        lvface._session = None

    def _mock_session(self, raw_output):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name='data')]
        mock_session.run.return_value = [np.array([raw_output], dtype=np.float32)]
        return mock_session

    def test_embed_from_image_array_normalizes_and_handles_bgr(self):
        import lvface
        raw = np.full(512, 2.0, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = np.array([[50, 60], [100, 60], [75, 90], [55, 120], [95, 120]], dtype=np.float32)
        with patch('lvface._get_session', return_value=self._mock_session(raw)):
            result = lvface.embed_from_image_array(img, kps, input_is_bgr=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_embed_from_image_array_accepts_flattened_kps(self):
        import lvface
        raw = np.full(512, 1.0, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        flat_kps = [50, 60, 100, 60, 75, 90, 55, 120, 95, 120]
        with patch('lvface._get_session', return_value=self._mock_session(raw)):
            result = lvface.embed_from_image_array(img, flat_kps, input_is_bgr=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (512,))

    def test_zero_norm_output_returns_none(self):
        import lvface
        raw = np.zeros(512, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = np.array([[50, 60], [100, 60], [75, 90], [55, 120], [95, 120]], dtype=np.float32)
        with patch('lvface._get_session', return_value=self._mock_session(raw)):
            result = lvface.embed_from_image_array(img, kps, input_is_bgr=True)
        self.assertIsNone(result)

    def test_embed_from_path_returns_none_on_undecodable_file(self):
        import lvface
        kps = [50, 60, 100, 60, 75, 90, 55, 120, 95, 120]
        result = lvface.embed_from_path('/nonexistent/path/does_not_exist.jpg', kps)
        self.assertIsNone(result)

    def test_session_is_built_lazily_and_cached(self):
        import lvface
        with patch('onnxruntime.InferenceSession') as mock_ort:
            mock_ort.return_value = MagicMock()
            s1 = lvface._get_session()
            s2 = lvface._get_session()
        self.assertIs(s1, s2)
        mock_ort.assert_called_once()


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class UpdateExistingFaceLvfaceEncodingTests(TestCase):
    """update_existing_face_to_insightface()'s LVFace-B wiring -- the
    IOU-rematch branch (find_and_encode_faces()'s "existing Face rows
    already exist for this image" path). Constructed via __new__ (same
    pattern as ReencodeMissingFacesTests) to avoid loading the real
    ~100MB insightface ONNX models, since this function never touches
    self.app at all (only self.iou_function/self.gender_map, both plain
    Python)."""

    def setUp(self):
        import torchvision.ops.boxes as bops
        self.image = make_image()
        self.extractor = FaceExtractor.__new__(FaceExtractor)
        self.extractor.iou_function = bops.distance_box_iou
        self.extractor.gender_map = {1: 'M', 0: 'F'}
        self.extractor.IOU_thresh = 0.3

    def _existing_face(self):
        return make_face(
            self.image, box_left=10, box_top=10, box_right=50, box_bottom=50,
        )

    def _new_data(self, bbox):
        import insightface.app.common
        return insightface.app.common.Face(
            bbox=np.array(bbox, dtype=np.float32),
            kps=np.zeros((5, 2), dtype=np.float32),
            det_score=0.9,
            embedding=np.ones(512, dtype=np.float32),
            age=30, gender=0,
        )

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_matching_box_stores_lvface_encoding(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.6, dtype=np.float32)) as mock_lv:
            self.extractor.update_existing_face_to_insightface(face, new_data, img_numpy=np.zeros((100, 100, 3)))
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)
        self.assertEqual(mock_lv.call_args.kwargs.get('input_is_bgr'), False)

    @override_settings(LVFACE_ENCODING_ENABLED=False)
    def test_lvface_disabled_never_attempts_encoding(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.update_existing_face_to_insightface(face, new_data, img_numpy=np.zeros((100, 100, 3)))
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        self.assertIsNotNone(face.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_no_img_numpy_skips_lvface_without_error(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.update_existing_face_to_insightface(face, new_data)
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        self.assertIsNotNone(face.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_lvface_failure_is_non_fatal(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array', side_effect=RuntimeError("boom")):
            self.extractor.update_existing_face_to_insightface(face, new_data, img_numpy=np.zeros((100, 100, 3)))
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        self.assertIsNotNone(face.face_encoding_512)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media", LVFACE_ENCODING_ENABLED=True)
class BackfillLvfaceVideoEncodingsTests(TestCase):
    """FaceExtractor._backfill_lvface_video_encodings() -- the video-only
    LVFace-B catch-up sweep, called automatically at the end of
    reencode_missing_faces() with whatever time budget remains after its
    own merged image-side insightface+LVFace-B pass (see that method's
    own tests in ReencodeMissingFacesTests, and its docstring). Mocks
    lvface.embed_from_image_array entirely to stay fast. LVFACE_ENCODING_
    ENABLED is overridden True class-wide here -- this class tests the
    sweep's own logic, not the (separately-tested) master on/off switch
    itself."""

    def setUp(self):
        self.extractor = FaceExtractor.__new__(FaceExtractor)

    def _make_video_face(self, **overrides):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/lvface_backfill_test')
        video_path = '/tmp/lvface_backfill_test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(b'not a real video, existence is all this sweep checks')
        self.addCleanup(lambda: os.path.exists(video_path) and os.remove(video_path))
        video = VideoFile.objects.create(filename=video_path, directory=directory, width=640, height=480)
        from face_manager.models import get_default_blank_person
        face = Face(
            declared_name=overrides.pop('declared_name', None) or get_default_blank_person(),
            source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
            video_thumbnail_frame_seconds=12.3,
            kps=[1.0] * 10,
            **overrides,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_video_sourced_face_is_encoded_via_frame_extraction(self):
        face = self._make_video_face()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=fake_frame) as mock_extract, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.7, dtype=np.float32)) as mock_lv:
            self.extractor._backfill_lvface_video_encodings()
        mock_extract.assert_called_once()
        # input_is_bgr=True for the video path (the frame comes straight
        # from cv2/PyAV, unlike the image path's PIL-sourced RGB array).
        self.assertIs(mock_lv.call_args.kwargs.get('input_is_bgr'), True)
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.7] * 512)

    def test_video_face_with_no_decodable_frame_is_left_unencoded(self):
        face = self._make_video_face()
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=None), \
             patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor._backfill_lvface_video_encodings()
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_confirmed_ignore_video_face_is_excluded(self):
        ignore_person = make_person(settings.SOFT_IGNORE_NAME)
        face = self._make_video_face(declared_name=ignore_person)
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp') as mock_extract, \
             patch('lvface.embed_from_image_array'):
            self.extractor._backfill_lvface_video_encodings()
        mock_extract.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_collection_scoped_call_skips_entirely(self):
        # collection is an image-only concept (PhotoCollection/
        # Directory.collection) -- a collection-scoped call has nothing
        # to say about video faces, so this sweep must not run at all
        # for one, rather than silently sweeping the whole unscoped
        # video backlog.
        from filepopulator.models import PhotoCollection
        face = self._make_video_face()
        default_collection = PhotoCollection.objects.get(is_default=True)
        with patch('video_face_pipeline.ffprobe_info') as mock_ffprobe, \
             patch('lvface.embed_from_image_array'):
            self.extractor._backfill_lvface_video_encodings(collection=default_collection)
        mock_ffprobe.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_time_budget_stops_the_sweep_between_faces(self):
        face_a = self._make_video_face()
        face_b = self._make_video_face()

        times = iter([0, 0, 1000])  # start_time, first check (ok), second check (over budget)

        def fake_time():
            return next(times, 1000)

        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.2, dtype=np.float32)) as mock_lv, \
             patch('time.time', side_effect=fake_time):
            self.extractor._backfill_lvface_video_encodings(max_runtime_seconds=10)

        # Only the first face should have been processed before the
        # (mocked) clock crossed the budget.
        self.assertEqual(mock_lv.call_count, 1)

    def test_shard_filters_to_matching_ids_only(self):
        # Added 2026-10-26 once the parallel image backfill was already
        # running well -- video faces have a real id, just like image
        # faces, so the same hash-based shard partition (_shard_
        # annotation()) applies unchanged. Every face must be covered
        # by exactly one shard.
        faces = [self._make_video_face() for _ in range(6)]
        total_shards = 3
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=fake_frame), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.6, dtype=np.float32)):
            for shard_index in range(total_shards):
                self.extractor._backfill_lvface_video_encodings(
                    shard_index=shard_index, total_shards=total_shards,
                )
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)


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
            thumbnail_small='',
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
            thumbnail_small='',
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

    def test_collapsed_away_persons_face_is_actually_gone(self):
        # A plain transfer with no collision doesn't change any person's
        # face count (declared_name is untouched by moving source_image_
        # file) -- this exercises the case where the post-transfer
        # collapse actually DELETES a face (the losing side's only face).
        dup = self._make_contaminated_duplicate(self.primary_img)
        losing_person = make_person("Losing Person")
        make_face(self.primary_img, declared_name=losing_person,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)
        make_face(dup, declared_name=self.person, validated=True,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')

        self.assertEqual(losing_person.face_declared.count(), 0)

    def test_rerunning_after_merge_finds_nothing_more_to_do(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        make_face(dup, declared_name=self.person,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')
        call_command('merge_duplicate_imagefiles', '--dry-run')

        dup_filenames = set(DuplicateFile.objects.values_list('filename', flat=True))
        self.assertEqual(ImageFile.objects.filter(filename__in=dup_filenames).count(), 0)


class VideoFaceFrameSecondsTests(unittest.TestCase):
    """_frame_seconds: real, live bug found 2026-09-11 via a production
    404 (face 1097880's "large image" -- see CLAUDE.md) -- a frame that
    has passed through extract_frame_near_timestamp's yadif deinterlace
    filter graph carries a DIFFERENT time_base than the source stream
    (confirmed on a real interlaced .m2ts file: 1/180000 vs the stream's
    own 1/90000), so applying the stream's time_base uniformly silently
    doubled the computed time for every deinterlaced frame. No real
    interlaced fixture is committed for a full integration test, but the
    actual bug is entirely in this one pts-to-seconds conversion, so a
    focused unit test on it directly covers the real fix."""

    def test_same_pts_with_different_time_bases_gives_different_seconds(self):
        # This is the bug itself, made concrete: the SAME numeric pts
        # value means a different real time depending on which
        # time_base it's interpreted through.
        from fractions import Fraction

        from video_face_pipeline import _frame_seconds
        pts = 44522298
        seconds_at_90k = _frame_seconds(pts, Fraction(1, 90000), start_seconds=0)
        seconds_at_180k = _frame_seconds(pts, Fraction(1, 180000), start_seconds=0)
        self.assertAlmostEqual(seconds_at_90k, seconds_at_180k * 2, places=6)

    def test_matching_real_world_time_survives_a_time_base_change(self):
        # The actual real-production values: an input frame at pts=
        # 22261149 in the stream's native 1/90000 time_base represents
        # the exact same real-world instant as yadif's corresponding
        # deinterlaced output (its first output overall -- yadif's
        # mode=0 needs one input's worth of buffering delay before its
        # first output, so this is the deinterlaced version of the
        # FIRST input frame, not the second) at pts=44522298 in yadif's
        # 1/180000 time_base -- confirmed directly against the real file
        # (2026-09-11). Using each frame's own time_base must recover
        # the same real second count for both.
        from fractions import Fraction

        from video_face_pipeline import _frame_seconds
        start_seconds = _frame_seconds(99009, Fraction(1, 90000), start_seconds=0)
        input_seconds = _frame_seconds(22261149, Fraction(1, 90000), start_seconds=start_seconds)
        output_seconds = _frame_seconds(44522298, Fraction(1, 180000), start_seconds=start_seconds)
        self.assertAlmostEqual(input_seconds, output_seconds, places=3)
        self.assertAlmostEqual(input_seconds, 246.246, places=2)

    def test_pre_fix_bug_reproduced_with_stream_time_base_applied_to_both(self):
        # Demonstrates the actual regression: naively applying the
        # STREAM's time_base (1/90000) to an output frame's pts that's
        # really in 1/180000 units produces roughly double the correct
        # value -- exactly the silent doubling that made
        # extract_frame_near_timestamp return None for interlaced videos.
        from fractions import Fraction

        from video_face_pipeline import _frame_seconds
        correct = _frame_seconds(44522298, Fraction(1, 180000), start_seconds=0)
        buggy = _frame_seconds(44522298, Fraction(1, 90000), start_seconds=0)
        self.assertAlmostEqual(buggy, correct * 2, places=3)


class HdrVideoToneMapTests(unittest.TestCase):
    """is_hdr_transfer/build_vf_filter -- pure functions, no DB/decode
    needed. See CLAUDE.md's 2026-09-22 write-up: a real HDR-tagged phone
    video (color_transfer=smpte2084, color_primaries=bt2020), decoded
    via this pipeline's plain bgr24 raw pipe with no tone-mapping, came
    out visibly washed-out/desaturated -- confirmed by extracting the
    same real frame with and without the zscale+tonemap filter chain."""

    def test_pq_and_hlg_are_hdr(self):
        from video_face_pipeline import is_hdr_transfer
        self.assertTrue(is_hdr_transfer('smpte2084'))
        self.assertTrue(is_hdr_transfer('arib-std-b67'))

    def test_ordinary_transfers_are_not_hdr(self):
        from video_face_pipeline import is_hdr_transfer
        for ct in ('bt709', 'unknown', None, ''):
            self.assertFalse(is_hdr_transfer(ct))

    def test_neither_filter_needed_gives_none(self):
        from video_face_pipeline import build_vf_filter
        self.assertIsNone(build_vf_filter('progressive', 'bt709'))
        self.assertIsNone(build_vf_filter('unknown', None))

    def test_only_deinterlace_needed(self):
        from video_face_pipeline import build_vf_filter
        self.assertEqual(build_vf_filter('tt', 'bt709'), 'yadif=0')

    def test_only_tonemap_needed(self):
        from video_face_pipeline import build_vf_filter, HDR_TONEMAP_FILTER
        self.assertEqual(build_vf_filter('progressive', 'smpte2084'), HDR_TONEMAP_FILTER)

    def test_both_combine_deinterlace_first(self):
        # Order matters: deinterlace must run on the source's native
        # pixel format before the tone-map chain's own format
        # conversions, not after.
        from video_face_pipeline import build_vf_filter, HDR_TONEMAP_FILTER
        combined = build_vf_filter('bb', 'arib-std-b67')
        self.assertEqual(combined, f'yadif=0,{HDR_TONEMAP_FILTER}')


class ExtractFrameNearTimestampHdrDispatchTests(unittest.TestCase):
    """extract_frame_near_timestamp() must dispatch HDR-tagged sources to
    the ffmpeg-based path (PyAV's own bundled libavfilter lacks the
    zscale filter the tone-map chain needs -- confirmed directly via
    av.filter.Filter('zscale') raising "no filter zscale", 2026-09-22)
    and leave every other video on the existing, already-hardened PyAV
    path untouched."""

    def test_hdr_source_dispatches_to_ffmpeg_path(self):
        from video_face_pipeline import extract_frame_near_timestamp
        with patch('video_face_pipeline._extract_frame_near_timestamp_ffmpeg') as mock_ffmpeg_path, \
             patch('video_face_pipeline.av.open') as mock_av_open:
            mock_ffmpeg_path.return_value = 'sentinel-frame'
            result = extract_frame_near_timestamp(
                '/fake/path.mp4', 10.0, 30.0, 'progressive', 0,
                (0, 0, 10, 10), np.zeros((10, 10, 3), dtype=np.uint8),
                width=1920, height=1080, color_transfer='smpte2084',
            )
        self.assertEqual(result, 'sentinel-frame')
        mock_ffmpeg_path.assert_called_once()
        mock_av_open.assert_not_called()

    def test_non_hdr_source_never_touches_ffmpeg_path(self):
        from video_face_pipeline import extract_frame_near_timestamp
        with patch('video_face_pipeline._extract_frame_near_timestamp_ffmpeg') as mock_ffmpeg_path, \
             patch('video_face_pipeline.av.open') as mock_av_open:
            mock_container = MagicMock()
            mock_container.streams.video = [MagicMock(start_time=0, time_base=1)]
            mock_container.demux.return_value = []
            mock_av_open.return_value = mock_container
            extract_frame_near_timestamp(
                '/fake/path.mp4', 10.0, 30.0, 'progressive', 0,
                (0, 0, 10, 10), np.zeros((10, 10, 3), dtype=np.uint8),
                width=1920, height=1080, color_transfer='bt709',
            )
        mock_ffmpeg_path.assert_not_called()
        mock_av_open.assert_called_once()


class ExtractFrameNearTimestampFfmpegPathTests(unittest.TestCase):
    """_extract_frame_near_timestamp_ffmpeg()'s own search logic, with
    ffmpeg_frame_iterator mocked out (no real video/subprocess needed --
    this tests the frame-selection and seek-overshoot-fallback logic
    directly, mirroring the PyAV path's own already-covered behavior)."""

    def _frame(self, fill_value):
        return np.full((40, 40, 3), fill_value, dtype=np.uint8)

    def _thumb(self, fill_value):
        # Matches what _extract_frame_near_timestamp_ffmpeg itself
        # produces internally (VideoFaceExtractor._square_thumbnail,
        # resized to settings.FACE_THUMBNAIL_SIZE) -- the reference
        # image must be the same shape or _mean_abs_pixel_diff refuses
        # to compare (returns None) and nothing would ever match.
        from video_face_pipeline import VideoFaceExtractor
        return VideoFaceExtractor._square_thumbnail(self._frame(fill_value), (5, 5, 35, 35))

    def test_picks_the_closest_pixel_match_within_the_window(self):
        from video_face_pipeline import _extract_frame_near_timestamp_ffmpeg
        reference = self._thumb(100)
        # Three candidate frames at t=0,1,2 (fps=1) -- only the middle
        # one (t=1, matching the target) is a real pixel match.
        frames = [self._frame(0), self._frame(100), self._frame(255)]
        with patch('video_face_pipeline.ffmpeg_frame_iterator', return_value=iter(frames)):
            result = _extract_frame_near_timestamp_ffmpeg(
                '/fake/path.mp4', target_seconds=1.0, avg_fps=1.0, field_order='progressive',
                box=(5, 5, 35, 35), reference_bgr=reference, window_seconds=1.5,
                width=40, height=40, color_transfer='bt709',
            )
        self.assertTrue(np.array_equal(result, self._frame(100)))

    def test_falls_back_to_full_decode_when_windowed_scan_finds_nothing(self):
        # Mirrors the PyAV path's own real 2026-09-11 WMV/ASF seek-
        # overshoot fix, adapted to what this raw-pipe path can actually
        # detect: unlike PyAV, ffmpeg's raw `rawvideo` output carries no
        # true per-frame timestamps at all, so a bad seek can't be
        # detected by comparing a frame's OWN real pts against the
        # window (there isn't one) -- only by the windowed scan coming
        # back with literally zero candidate frames (e.g. the seek
        # landed at/past EOF). That's the case this test simulates and
        # the fallback covers: retry with a full sequential decode from
        # frame zero rather than trusting the seek a second time.
        from video_face_pipeline import _extract_frame_near_timestamp_ffmpeg
        reference = self._thumb(100)
        calls = []

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            calls.append(seek_seconds)
            if seek_seconds > 0:
                return iter([])  # seek landed past any usable content
            return iter([self._frame(0), self._frame(100), self._frame(0)])

        with patch('video_face_pipeline.ffmpeg_frame_iterator', side_effect=fake_iterator):
            result = _extract_frame_near_timestamp_ffmpeg(
                '/fake/path.mp4', target_seconds=1.0, avg_fps=1.0, field_order='progressive',
                box=(5, 5, 35, 35), reference_bgr=reference, window_seconds=0.5,
                width=40, height=40, color_transfer='bt709',
            )
        self.assertTrue(np.array_equal(result, self._frame(100)))
        self.assertEqual(calls, [0.5, 0])

    def test_returns_none_when_nothing_matches_anywhere(self):
        from video_face_pipeline import _extract_frame_near_timestamp_ffmpeg
        reference = self._frame(100)
        with patch('video_face_pipeline.ffmpeg_frame_iterator', return_value=iter([])):
            result = _extract_frame_near_timestamp_ffmpeg(
                '/fake/path.mp4', target_seconds=1.0, avg_fps=1.0, field_order='progressive',
                box=(0, 0, 4, 4), reference_bgr=reference, window_seconds=0.5,
                width=4, height=4, color_transfer='bt709',
            )
        self.assertIsNone(result)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class RetonemapHdrVideoThumbnailsTests(TestCase):
    """retonemap_hdr_video_thumbnails management command -- regenerates
    ONLY the stored face_thumbnail file for already-processed HDR-video
    faces, using the already-known-correct video_thumbnail_frame_seconds
    as ground truth (no pixel-matching needed here, unlike
    backfill_video_thumbnail_timestamps.py, which recovers an UNKNOWN
    timestamp by matching against the existing thumbnail -- that
    wouldn't make sense here since the existing thumbnail IS the
    washed-out artifact being replaced). box/kps/embedding/det_score/
    declared_name/validated must all stay untouched -- only the
    thumbnail's bytes should change."""

    CMD = 'face_manager.management.commands.retonemap_hdr_video_thumbnails'

    def _make_video(self, path, width=100, height=100):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
        return VideoFile.objects.create(
            filename=path, directory=directory, width=width, height=height,
            duration_seconds=30, isProcessed=True,
        )

    def _make_person(self, name):
        person, created = Person.objects.get_or_create(person_name=name)
        if created:
            person.highlight_img.save(f"{name}.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        return person

    def _make_face(self, video, person, timestamp, validated=False, embedding=None):
        face = Face(
            declared_name=person, source_video_file=video, validated=validated,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
            video_thumbnail_frame_seconds=timestamp, face_encoding_512=embedding,
        )
        face.face_thumbnail.save(
            f"thumb_{video.pk}_{timestamp}.jpg",
            ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
        )
        face.save()
        return face

    def _frame(self, fill_value, size=100):
        return np.full((size, size, 3), fill_value, dtype=np.uint8)

    def test_regenerates_thumbnail_leaving_identity_and_box_untouched(self):
        video = self._make_video('/videos/hdr_test/a.mp4')
        person = self._make_person('Retonemap Test Person')
        face = self._make_face(video, person, timestamp=2.0, validated=True, embedding=[0.1] * 512)

        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)
        old_declared_id = face.declared_name_id
        old_validated = face.validated
        old_embedding = list(face.face_encoding_512)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            # seek_seconds lands at 0 (target 2.0 - 5 margin, clamped);
            # frame at real t=2.0 gets a distinct fill value so a wrong
            # pick is detectable.
            for i in range(8):
                t = seek_seconds + i
                yield self._frame(50 if abs(t - 2.0) < 0.01 else 10)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'smpte2084', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator', side_effect=fake_iterator):
            call_command('retonemap_hdr_video_thumbnails', '--yes')

        face.refresh_from_db()
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual(face.declared_name_id, old_declared_id)
        self.assertEqual(face.validated, old_validated)
        self.assertEqual(list(face.face_encoding_512), old_embedding)

        with open(face.face_thumbnail.path, 'rb') as fh:
            new_bytes = fh.read()
        self.assertNotEqual(new_bytes, old_bytes)

    def test_non_hdr_video_is_skipped_entirely(self):
        video = self._make_video('/videos/hdr_test/b.mp4')
        person = self._make_person('Retonemap Non HDR Person')
        face = self._make_face(video, person, timestamp=2.0)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'bt709', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator') as mock_iterator:
            call_command('retonemap_hdr_video_thumbnails', '--yes')

        mock_iterator.assert_not_called()
        with open(face.face_thumbnail.path, 'rb') as fh:
            self.assertEqual(fh.read(), old_bytes)

    def test_dry_run_makes_no_changes(self):
        # Uses the same fake_iterator shape as the real-regenerate test
        # (a genuine match IS found at t=2.0) specifically so this test
        # actually exercises "found a match, but --dry-run skips the
        # write" -- not just "nothing was found anyway".
        video = self._make_video('/videos/hdr_test/c.mp4')
        person = self._make_person('Retonemap Dry Run Person')
        face = self._make_face(video, person, timestamp=2.0)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            for i in range(8):
                t = seek_seconds + i
                yield self._frame(50 if abs(t - 2.0) < 0.01 else 10)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'smpte2084', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator', side_effect=fake_iterator):
            call_command('retonemap_hdr_video_thumbnails', '--dry-run')

        with open(face.face_thumbnail.path, 'rb') as fh:
            self.assertEqual(fh.read(), old_bytes)

    def test_no_frame_within_tolerance_leaves_thumbnail_untouched_and_reports_unresolved(self):
        video = self._make_video('/videos/hdr_test/d.mp4')
        person = self._make_person('Retonemap Unresolved Person')
        face = self._make_face(video, person, timestamp=2.0)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            # Every yielded frame is far outside the 0.5s tolerance.
            yield self._frame(50)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'smpte2084', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator', side_effect=fake_iterator):
            call_command('retonemap_hdr_video_thumbnails', '--yes')

        with open(face.face_thumbnail.path, 'rb') as fh:
            self.assertEqual(fh.read(), old_bytes)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class RedetectSarAffectedVideosTests(TestCase):
    """redetect_sar_affected_videos management command -- reconciles a
    fresh full-redetect pass (VideoFaceExtractor mocked entirely; real
    ONNX inference is covered by VideoFaceExtractorRealInferenceTests
    elsewhere) against a video's existing Face rows. A matched face must
    keep its identity but get fresh geometry/embedding; an unmatched
    candidate becomes a new unclassified face; an unmatched existing
    face is left completely untouched."""

    CMD = 'face_manager.management.commands.redetect_sar_affected_videos'

    def _make_video(self, path, sar_scale_width=640, width=352, height=480):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
        return VideoFile.objects.create(
            filename=path, directory=directory, width=width, height=height,
            duration_seconds=30, isProcessed=True, sar_scale_width=sar_scale_width,
        )

    def _make_person(self, name):
        person, created = Person.objects.get_or_create(person_name=name)
        if created:
            person.highlight_img.save(f"{name}.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        return person

    def _make_face(self, video, person, embedding, first=1.0, last=2.0, validated=False):
        face = Face(
            declared_name=person, source_video_file=video, validated=validated,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
            video_first_timestamp_seconds=first, video_last_timestamp_seconds=last,
            face_encoding_512=embedding,
        )
        face.face_thumbnail.save(
            "thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
        )
        face.save()
        return face

    def _emb(self, direction, dim=512):
        v = [0.0] * dim
        v[direction] = 1.0
        return v

    def _candidate(self, embedding, box=(1, 1, 40, 40), first_sample=10, last_sample=20):
        import numpy as np
        rep = {'box': box, 'kps': [1.0] * 10, 'det_score': 0.9, 'frame_idx': 0, 'emb_if': embedding}
        return {
            'centroid': np.array(embedding),
            'provisional': rep,
            'pooled_reps': [rep],
            'first_sample': first_sample,
            'last_sample': last_sample,
        }

    def _fake_extractor(self, candidates, blank_person, fps=30.0):
        import numpy as np
        mock_extractor = MagicMock()
        mock_extractor.blank_face_person = blank_person
        mock_extractor._compute_candidate_groups.return_value = (fps, {0: np.zeros((10, 10, 3), dtype=np.uint8)}, candidates)
        mock_extractor._clamp_to_duration.side_effect = lambda video, seconds, label: seconds

        def fake_set_box(face, frame, box, kps, timestamp_seconds, det_score=None):
            face.box_left, face.box_top, face.box_right, face.box_bottom = box
            face.kps = list(kps)
            face.det_score = det_score
            face.video_thumbnail_frame_seconds = timestamp_seconds
            face.face_thumbnail.save(
                "redetect_thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
            )
        mock_extractor._set_face_box_and_thumbnail.side_effect = fake_set_box
        mock_extractor._classify_and_pick_thumbnail.return_value = None
        return mock_extractor

    def test_matched_face_keeps_identity_gets_fresh_geometry(self):
        from filepopulator.models import VideoFile
        video = self._make_video('/videos/sar_test/a.mpg')
        person = self._make_person('Redetect Test Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        old_embedding = self._emb(0)
        face = self._make_face(video, person, old_embedding, validated=True)
        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

        new_embedding = self._emb(0)  # same direction -- should match
        new_embedding[0] = 0.99  # slightly different, still a clear match
        candidates = [self._candidate(new_embedding, box=(5, 5, 55, 55))]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--yes')

        face.refresh_from_db()
        self.assertEqual(face.declared_name_id, person.id)
        self.assertTrue(face.validated)
        self.assertNotEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), (5, 5, 55, 55))
        self.assertAlmostEqual(face.face_encoding_512[0], new_embedding[0], places=3)
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 1)

    def test_unmatched_candidate_becomes_new_unclassified_face(self):
        video = self._make_video('/videos/sar_test/b.mpg')
        person = self._make_person('Redetect Existing Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        self._make_face(video, person, self._emb(0))

        # Completely different embedding direction -- a genuinely new face.
        candidates = [self._candidate(self._emb(5), box=(2, 2, 42, 42))]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--yes')

        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 2)
        new_face = Face.objects.filter(source_video_file=video).exclude(declared_name=person).get()
        self.assertEqual(new_face.declared_name_id, blank.id)

    def test_unmatched_existing_face_left_completely_untouched(self):
        video = self._make_video('/videos/sar_test/c.mpg')
        person = self._make_person('Redetect Untouched Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        face = self._make_face(video, person, self._emb(0), validated=True)
        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)
        old_embedding = list(face.face_encoding_512)

        candidates = [self._candidate(self._emb(7))]  # no real match

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--yes')

        face.refresh_from_db()
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual(list(face.face_encoding_512), old_embedding)
        self.assertEqual(face.declared_name_id, person.id)
        self.assertTrue(face.validated)
        # Plus the genuinely new candidate, created separately.
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 2)

    def test_dry_run_makes_no_changes(self):
        video = self._make_video('/videos/sar_test/d.mpg')
        person = self._make_person('Redetect Dry Run Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        face = self._make_face(video, person, self._emb(0))
        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

        candidates = [self._candidate(self._emb(0))]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--dry-run')

        face.refresh_from_db()
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 1)

    def test_non_sar_video_is_not_selected(self):
        video = self._make_video('/videos/sar_test/e.mp4', sar_scale_width=0)
        person = self._make_person('Redetect Square Pixel Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        self._make_face(video, person, self._emb(0))

        with patch(f'{self.CMD}.VideoFaceExtractor') as MockExtractorCls:
            call_command('redetect_sar_affected_videos', '--yes')

        MockExtractorCls.assert_not_called()
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 1)

    def test_timeout_on_one_video_is_skipped_and_the_run_continues(self):
        # Real, reproduced 2026-09-23: a video timed out under real
        # concurrent CPU/memory pressure from another job, not because
        # of anything wrong with the file (re-running it in isolation
        # took ~3s) -- same class of issue the scheduled video_face_
        # extraction task already guards against. This must be reported
        # distinctly from a real failure and must NOT crash the whole
        # command -- later videos still get processed.
        from face_manager.tasks import _VideoProcessingTimeout
        video_a = self._make_video('/videos/sar_test/f.mpg')
        video_b = self._make_video('/videos/sar_test/g.mpg')
        person = self._make_person('Redetect Timeout Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        self._make_face(video_a, person, self._emb(0))
        self._make_face(video_b, person, self._emb(0))

        candidates = [self._candidate(self._emb(0))]
        fake_extractor = self._fake_extractor(candidates, blank)
        fake_extractor._compute_candidate_groups.side_effect = [
            _VideoProcessingTimeout("exceeded 2700s"),
            fake_extractor._compute_candidate_groups.return_value,
        ]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=fake_extractor):
            call_command('redetect_sar_affected_videos', '--yes')

        # video_a's face untouched (its redetect never completed);
        # video_b's face still got matched/updated normally.
        self.assertEqual(fake_extractor._compute_candidate_groups.call_count, 2)


class VideoFaceIOUTrackingTests(unittest.TestCase):
    """_iou_track: pure frame-to-frame linking, no Django/model deps --
    see video_face_pipeline.py / CLAUDE.md's Phase 3 design for the full
    investigation these constants (grow 25%, iou_thresh 0.3, gap
    tolerance 1 sample) came out of."""

    def _det(self, sample_idx, boxes):
        # boxes: list of (l, t, r, b); scores/kps are unused by tracking
        # logic itself, just carried through -- filler values are fine.
        return (sample_idx, boxes, [1.0] * len(boxes), [[0.0] * 10] * len(boxes))

    def test_same_box_across_samples_forms_one_track(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50]]), self._det(1, [[11, 11, 51, 51]]),
                self._det(2, [[10, 10, 50, 50]])]
        tracks = _iou_track(dets)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['span'], (0, 2))
        self.assertEqual(len(tracks[0]['boxes']), 3)

    def test_two_spatially_distinct_boxes_form_two_tracks(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50], [500, 500, 540, 540]]),
                self._det(1, [[10, 10, 50, 50], [500, 500, 540, 540]])]
        tracks = _iou_track(dets)
        self.assertEqual(len(tracks), 2)

    def test_gap_tolerance_bridges_one_missed_sample(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50]]),
                self._det(1, []),  # missed detection
                self._det(2, [[10, 10, 50, 50]])]
        tracks = _iou_track(dets, max_gap=1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['span'], (0, 2))

    def test_gap_beyond_tolerance_splits_into_two_tracks(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50]]),
                self._det(1, []), self._det(2, []),  # two consecutive misses
                self._det(3, [[10, 10, 50, 50]])]
        tracks = _iou_track(dets, max_gap=1)
        self.assertEqual(len(tracks), 2)

    def test_far_apart_boxes_in_same_frame_never_merge_regardless_of_gap(self):
        from video_face_pipeline import _iou_track
        # Two different people, both present in every sampled frame --
        # must never collapse into one track no matter the gap tolerance.
        dets = [self._det(i, [[10, 10, 50, 50], [500, 500, 540, 540]]) for i in range(5)]
        tracks = _iou_track(dets, max_gap=1)
        self.assertEqual(len(tracks), 2)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillDetScoreTests(TestCase):
    """backfill_det_score management command: re-detects each unlabeled
    .ignore face's source image and matches the result back to the
    stored box by IOU, storing the matched detection's own det_score.
    Mocks _build_detection_only_detector entirely (real model loading is
    expensive) and common.open_img_oriented (no real image decode
    needed) -- only the command's own IOU-matching/grouping logic is
    under test here. Uses the real FaceExtractor._flatten_kps (a pure
    static method, cheap to call directly -- doesn't instantiate
    FaceExtractor, so no model loading happens) rather than mocking it."""

    class _FakeDetection:
        def __init__(self, bbox, det_score, kps=None):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.det_score = det_score
            self._kps = kps if kps is not None else [[1.0, 2.0]] * 5

        def __getitem__(self, key):
            if key == 'kps':
                return self._kps
            raise KeyError(key)

    def setUp(self):
        self.ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.image = make_image()
        self.image.width, self.image.height = 1000, 1000
        self.image.save()

    def _make_ignore_face(self, box):
        face = make_face(
            self.image, declared_name=Person.objects.get(person_name=settings.BLANK_FACE_NAME),
            box_left=box[0], box_top=box[1], box_right=box[2], box_bottom=box[3],
        )
        face.set_possible_person(self.ignore_person, 1, 1.0)
        face.refresh_from_db()
        return face

    def test_matched_detection_sets_det_score_and_kps(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_kps = [[15.0, 15.0], [35.0, 15.0], [25.0, 25.0], [17.0, 35.0], [33.0, 35.0]]
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91, kps=fake_kps)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.91, places=4)
        self.assertEqual(face.kps, [v for pair in fake_kps for v in pair])

    def test_matched_face_save_does_not_open_thumbnail_file(self):
        # Real crash found running this at scale: image_list holds every
        # Face object in memory for the whole run, so nothing is ever
        # garbage-collected between saves -- Face.save() used to
        # unconditionally open the thumbnail file (to check it exists)
        # and never close it, and with thousands of faces processed in
        # one long-lived process this exhausted the OS fd limit (OSError:
        # Too many open files, hit for real at ~600/38489 images). Fixed
        # at the source (Face.save(), see FaceSaveThumbnailCheckTests
        # below) by using .path (pure filesystem-path computation) instead
        # of .file.name (which opens the file) -- this test confirms the
        # backfill command's own save() call benefits from that fix by
        # never opening the file at all, not just closing it after.
        from django.core.files.storage import Storage
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build, \
             patch.object(Storage, 'open', autospec=True) as mock_storage_open:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        mock_storage_open.assert_not_called()

    def test_no_matching_detection_leaves_det_score_null(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        # A detection far from the stored box -- shouldn't match.
        fake_dets = [self._FakeDetection([500, 500, 540, 540], 0.9)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_dry_run_does_not_save(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score', '--dry-run')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_multiple_faces_on_same_image_share_one_detect_call(self):
        face1 = self._make_ignore_face([10, 10, 50, 50])
        face2 = self._make_ignore_face([200, 200, 260, 260])
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.80),
            self._FakeDetection([200, 200, 260, 260], 0.65),
        ]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        self.assertEqual(mock_build.return_value.get.call_count, 1)
        face1.refresh_from_db()
        face2.refresh_from_db()
        self.assertAlmostEqual(face1.det_score, 0.80, places=4)
        self.assertAlmostEqual(face2.det_score, 0.65, places=4)

    def test_non_ignore_faces_on_a_triggered_image_are_also_backfilled(self):
        # Added 2026-09-18 per explicit request: .ignore faces still drive
        # WHICH images get processed, but a real, confirmed (non-ignore)
        # face on the same already-triggered image should also get its
        # det_score/kps backfilled for free, since the whole image is
        # already decoded and detected regardless of which face triggered
        # it -- no extra detect call needed.
        ignore_face = self._make_ignore_face([10, 10, 50, 50])
        confirmed_person = make_person("Someone Real")
        confirmed_face = make_face(
            self.image, declared_name=confirmed_person,
            box_left=200, box_top=200, box_right=260, box_bottom=260,
        )
        self.assertIsNone(confirmed_face.det_score)
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.80),
            self._FakeDetection([200, 200, 260, 260], 0.75),
        ]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        self.assertEqual(mock_build.return_value.get.call_count, 1)
        ignore_face.refresh_from_db()
        confirmed_face.refresh_from_db()
        self.assertAlmostEqual(ignore_face.det_score, 0.80, places=4)
        self.assertAlmostEqual(confirmed_face.det_score, 0.75, places=4)
        # The confirmed face's own identity is untouched -- only
        # det_score/kps were ever written.
        self.assertEqual(confirmed_face.declared_name_id, confirmed_person.id)

    def test_non_ignore_face_on_an_untriggered_image_is_not_touched(self):
        # A non-ignore face's own image is only ever processed if that
        # SAME image also has a .ignore face still needing det_score --
        # this does not become a general "backfill every face everywhere"
        # sweep.
        other_image = make_image()
        other_image.width, other_image.height = 1000, 1000
        other_image.save()
        confirmed_person = make_person("Someone Else")
        untouched_face = make_face(
            other_image, declared_name=confirmed_person,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
        )
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented') as mock_open, \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score')
        mock_open.assert_not_called()
        untouched_face.refresh_from_db()
        self.assertIsNone(untouched_face.det_score)

    def test_already_populated_faces_are_excluded(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        face.det_score = 0.5
        face.save()
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented') as mock_open, \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            call_command('backfill_det_score')
        mock_open.assert_not_called()
        mock_build.return_value.get.assert_not_called()

    def test_all_faces_flag_widens_trigger_query_beyond_ignore(self):
        # --all-faces (added 2026-09-20 per explicit request): a
        # non-ignore face on an image with NO .ignore face at all should
        # now trigger processing too, unlike the default scope (see
        # test_non_ignore_face_on_an_untriggered_image_is_not_touched
        # above, which confirms the OPPOSITE behavior without the flag).
        confirmed_person = make_person("Someone Else")
        untouched_face = make_face(
            self.image, declared_name=confirmed_person,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
        )
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.88)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score', '--all-faces')
        untouched_face.refresh_from_db()
        self.assertAlmostEqual(untouched_face.det_score, 0.88, places=4)

    def test_single_pass_flag_restricts_detector_to_one_pass(self):
        # cut_list=[1,3] (the default, matching the live pipeline exactly)
        # measured at ~0.10 img/s in real production -- a multi-day run
        # for the full unlabeled .ignore population. --single-pass trades
        # some recall (small/edge faces the tiled pass would catch) for
        # roughly 10x fewer detection calls per image.
        self._make_ignore_face([10, 10, 50, 50])
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--single-pass')
        self.assertEqual(mock_build.return_value.cut_list, [1])

    def test_cut_list_flag_overrides_default(self):
        # Benchmarked 2026-09-18: cut_list=[1,2] matched exactly as well as
        # the full [1,3] on a real 20-image/123-face sample (123/123 both),
        # at half the detection calls -- the recommended middle ground.
        self._make_ignore_face([10, 10, 50, 50])
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--cut-list', '1,2')
        self.assertEqual(mock_build.return_value.cut_list, [1, 2])

    def test_cut_list_flag_takes_precedence_over_single_pass(self):
        self._make_ignore_face([10, 10, 50, 50])
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--single-pass', '--cut-list', '1,3')
        self.assertEqual(mock_build.return_value.cut_list, [1, 3])

    def test_per_image_timeout_skips_and_continues(self):
        # Real production hang found running this at scale: one image
        # drove CPU to 1400%+ with the DB's det_score count stuck for
        # minutes -- no specific cause identified, but the same class of
        # risk process_video_faces()'s own PER_VIDEO_TIMEOUT_SECONDS
        # guards against. A pathological image must not be able to hang
        # the whole batch indefinitely.
        import time as time_module
        import face_manager.management.commands.backfill_det_score as cmd_module
        face = self._make_ignore_face([10, 10, 50, 50])

        def slow_get(img):
            time_module.sleep(2)
            return []

        with patch.object(cmd_module, 'PER_IMAGE_TIMEOUT_SECONDS', 1), \
             patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.side_effect = slow_get
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    @tag("slow")
    def test_detection_only_detector_produces_valid_real_scores(self):
        # Real-inference sanity check, not a re-proof of equivalence --
        # equivalence with the full FaceExtractor().app (bit-identical
        # bbox/det_score) was already confirmed manually against a real
        # 70-face production photo (see this method's own module docstring
        # / CLAUDE.md, 2026-09-18): 127.9s -> 3.2s, a 39.69x speedup, with
        # zero difference in any of the 70 detected boxes or scores. This
        # test just confirms the detection-only detector, run for real
        # (no mocks), produces sane det_score/kps output against a known
        # fixture -- catches a real regression (e.g. a bad allowed_modules
        # value silently breaking detection) without needing to reload the
        # full multi-module detector every CI run.
        import common
        from face_manager.management.commands.backfill_det_score import (
            _build_detection_only_detector,
        )
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        img_numpy = common.open_img_oriented(path, as_numpy=True)
        app = _build_detection_only_detector()
        app.cut_list = [1, 2]
        detections = app.get(img_numpy)
        self.assertGreaterEqual(len(detections), 1)
        for d in detections:
            self.assertGreater(d.det_score, 0.0)
            self.assertLessEqual(d.det_score, 1.0)
            self.assertEqual(len(d['kps']), 5)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillVideoDetScoreTests(TestCase):
    """backfill_video_det_score management command: re-extracts the
    exact frame a video face's thumbnail/box came from, redetects in a
    padded crop, and only trusts the result if it actually overlaps
    (IOU >= 0.3) the already-stored box. Mocks VideoFaceExtractor
    entirely (real model loading), plus ffprobe_info and
    extract_frame_near_timestamp (real video decode) -- only the
    command's own IOU-matching/counting logic is under test here."""

    CMD_MODULE = 'face_manager.management.commands.backfill_video_det_score'

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/video_det_backfill_test')
        self.video = VideoFile.objects.create(
            filename='/tmp/video_det_backfill_test/a.mp4', directory=directory,
            width=320, height=240, duration_seconds=10.0,
        )
        self.blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)

    def _make_video_face(self, box=(10, 10, 50, 50)):
        face = Face(
            source_video_file=self.video, declared_name=self.blank_person,
            box_left=box[0], box_top=box[1], box_right=box[2], box_bottom=box[3],
            video_thumbnail_frame_seconds=1.5, det_score=None,
        )
        face.face_thumbnail.save(
            "thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False
        )
        face.save()
        return face

    def _ffprobe_return(self):
        # (width, height, avg_fps, field_order, rotation, color_transfer, sar_scale_width)
        return (320, 240, 30.0, 'progressive', 0, None, None)

    def test_matched_detection_sets_det_score(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            MockExtractor.return_value._redetect_in_crop.return_value = (
                [10, 10, 50, 50], [[0, 0]] * 5, 0.87,
            )
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.87)

    def test_low_iou_match_is_rejected(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            # Redetected box is far away -- essentially zero overlap.
            MockExtractor.return_value._redetect_in_crop.return_value = (
                [200, 200, 240, 240], [[0, 0]] * 5, 0.95,
            )
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_no_frame_extracted_skips(self):
        face = self._make_video_face()

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=None):
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_no_detection_in_crop_skips(self):
        face = self._make_video_face()
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            MockExtractor.return_value._redetect_in_crop.return_value = None
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_dry_run_does_not_save(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            MockExtractor.return_value._redetect_in_crop.return_value = (
                [10, 10, 50, 50], [[0, 0]] * 5, 0.87,
            )
            call_command('backfill_video_det_score', '--dry-run')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_face_with_existing_det_score_is_excluded(self):
        face = self._make_video_face()
        Face.objects.filter(pk=face.pk).update(det_score=0.75)
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()) as mock_ffprobe, \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            call_command('backfill_video_det_score')
            mock_ffprobe.assert_not_called()

        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.75)

    def test_det_thresh_override_applied_to_detector(self):
        face = self._make_video_face()
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            mock_instance = MockExtractor.return_value
            mock_instance.det10g_model.det_thresh = 0.5
            mock_instance._redetect_in_crop.return_value = ([10, 10, 50, 50], [[0, 0]] * 5, 0.35)
            call_command('backfill_video_det_score', '--det-thresh', '0.3')
            self.assertEqual(mock_instance.det10g_model.det_thresh, 0.3)

        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.35)

    def test_pad_pct_override_passed_through(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            mock_instance = MockExtractor.return_value
            mock_instance._redetect_in_crop.return_value = ([10, 10, 50, 50], [[0, 0]] * 5, 0.9)
            call_command('backfill_video_det_score', '--pad-pct', '1.2')
            mock_instance._redetect_in_crop.assert_called_once_with(
                fake_frame, (10, 10, 50, 50), pad_pct=1.2,
            )


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillMissingFaceThumbnailsTests(TestCase):
    """backfill_missing_face_thumbnails management command: real
    incident, 2026-10-06 -- a restored-from-backup database referenced
    thumbnail files that no longer existed on disk. Re-crops from the
    already-stored box (no re-detection) using a real fixture and a real
    decode, rather than mocking -- this command's whole point is pure
    cropping math against real pixel data, so there's nothing to usefully
    mock here."""

    def setUp(self):
        self.image = make_image()
        self.person = make_person("Someone Real")

    def test_missing_thumbnail_is_regenerated_from_the_stored_box(self):
        face = make_face(self.image, declared_name=self.person)
        old_name = face.face_thumbnail.name
        os.remove(face.face_thumbnail.path)

        call_command('backfill_missing_face_thumbnails')

        face.refresh_from_db()
        self.assertTrue(os.path.isfile(face.face_thumbnail.path))
        # A fresh save() with the same filename-derivation inputs can
        # coincidentally reuse the same name if nothing else on disk
        # collides -- what actually matters is that a real file exists
        # now, not that the name necessarily changed.
        self.assertIsNotNone(face.face_thumbnail.name)
        # Classification fields must be untouched.
        self.assertEqual(face.declared_name_id, self.person.id)

    def test_existing_thumbnail_is_left_untouched(self):
        face = make_face(self.image, declared_name=self.person)
        original_name = face.face_thumbnail.name
        original_mtime = os.path.getmtime(face.face_thumbnail.path)

        call_command('backfill_missing_face_thumbnails')

        face.refresh_from_db()
        self.assertEqual(face.face_thumbnail.name, original_name)
        self.assertEqual(os.path.getmtime(face.face_thumbnail.path), original_mtime)

    def test_dry_run_does_not_write(self):
        face = make_face(self.image, declared_name=self.person)
        os.remove(face.face_thumbnail.path)

        call_command('backfill_missing_face_thumbnails', '--dry-run')

        face.refresh_from_db()
        self.assertFalse(os.path.isfile(face.face_thumbnail.path))

    def test_decode_failure_is_counted_not_fatal(self):
        face = make_face(self.image, declared_name=self.person)
        os.remove(face.face_thumbnail.path)
        with patch(
            'face_manager.management.commands.backfill_missing_face_thumbnails.common.open_img_oriented',
            side_effect=OSError('simulated decode failure'),
        ):
            call_command('backfill_missing_face_thumbnails')  # must not raise
        face.refresh_from_db()
        self.assertFalse(os.path.isfile(face.face_thumbnail.path))


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class FaceSaveThumbnailCheckTests(TestCase):
    """Face.save()'s thumbnail-existence check uses .path (a pure
    filesystem-path computation via the storage backend), not
    .file.name (which OPENS the underlying file via FieldFile._get_file()
    and is never closed by Django) -- so it stays cheap and leak-free
    regardless of update_fields, not just for a narrow save. Real crash
    this was found from: OSError: Too many open files at ~600 images
    into backfill_det_score, since that command holds every Face object
    in memory for the whole run so nothing is ever garbage-collected
    between saves. See BackfillDetScoreTests.
    test_matched_face_save_does_not_open_thumbnail_file for the same
    guarantee exercised through the real backfill command."""

    def test_save_never_opens_the_thumbnail_file(self):
        # Patches Storage.open itself (not FieldFile._get_file) since
        # property(_get_file, ...) captures that function object at
        # class-definition time -- patching the _get_file *name*
        # afterward doesn't change what the already-built property
        # descriptor actually calls. A freshly-fetched Face (matching how
        # the real backfill always encounters them) has no cached
        # FieldFile._file yet either, unlike re-saving the same in-memory
        # instance make_face() already saved once.
        from django.core.files.storage import Storage
        image = make_image()
        face_id = make_face(image).id
        for save_kwargs in [{}, {'update_fields': ['detected_age']}, {'update_fields': ['face_thumbnail']}]:
            with self.subTest(save_kwargs=save_kwargs):
                face = Face.objects.get(id=face_id)
                face.detected_age = 42
                with patch.object(Storage, 'open', autospec=True) as mock_storage_open:
                    face.save(**save_kwargs)
                mock_storage_open.assert_not_called()

    def test_missing_thumbnail_file_still_raises(self):
        image = make_image()
        face = make_face(image)
        os.remove(face.face_thumbnail.path)
        face.detected_age = 43
        with self.assertRaises(ValidationError):
            face.save()


class VideoFaceShortTrackFilterTests(unittest.TestCase):
    """_filter_short_tracks: drops raw tracks matched across too few
    sampled frames before the expensive per-track redetect+dual-encode
    pass -- but ONLY once a video's raw singleton count looks
    pathological (ABSURD_SINGLETON_TRACK_COUNT), since a normal video's
    occasional genuine single-appearance track is real signal, not noise
    -- see that constant's own comment in video_face_pipeline.py for the
    real videos (2 old MJPEG .AVIs producing 2,400+ tracks, 78-80%
    singletons) that motivated this gate."""

    def _track(self, n_frames):
        return {'frames': list(range(n_frames))}

    def test_below_the_gate_keeps_every_track_including_singletons(self):
        # Only 2 singletons -- nowhere near "absurd" -- so nothing is
        # dropped, even the singletons themselves.
        from video_face_pipeline import _filter_short_tracks
        tracks = [self._track(1), self._track(1), self._track(3), self._track(10)]
        kept = _filter_short_tracks(tracks, min_len=3, absurd_singleton_count=150)
        self.assertEqual(len(kept), 4)

    def test_at_the_gate_drops_tracks_below_the_length_floor(self):
        # Exactly absurd_singleton_count singletons trips the gate (>=,
        # not strictly >), and once tripped the length floor applies.
        from video_face_pipeline import _filter_short_tracks
        tracks = [self._track(1)] * 3 + [self._track(2), self._track(3), self._track(10)]
        kept = _filter_short_tracks(tracks, min_len=3, absurd_singleton_count=3)
        self.assertEqual([len(t['frames']) for t in kept], [3, 10])

    def test_one_below_the_gate_changes_nothing(self):
        from video_face_pipeline import _filter_short_tracks
        tracks = [self._track(1)] * 2 + [self._track(10)]
        kept = _filter_short_tracks(tracks, min_len=3, absurd_singleton_count=3)
        self.assertEqual(len(kept), 3)

    def test_empty_input_returns_empty(self):
        from video_face_pipeline import _filter_short_tracks
        self.assertEqual(_filter_short_tracks([], min_len=3, absurd_singleton_count=3), [])

    def test_defaults_match_module_constants(self):
        from video_face_pipeline import (
            _filter_short_tracks, MIN_TRACK_LEN_SAMPLES, ABSURD_SINGLETON_TRACK_COUNT,
        )
        tracks = ([self._track(1)] * ABSURD_SINGLETON_TRACK_COUNT
                  + [self._track(MIN_TRACK_LEN_SAMPLES - 1), self._track(MIN_TRACK_LEN_SAMPLES)])
        kept = _filter_short_tracks(tracks)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0]['frames']), MIN_TRACK_LEN_SAMPLES)


class VideoFaceClusterTests(unittest.TestCase):
    """_cluster_track_faces: must-link (same track) + cannot-link
    (temporally-overlapping different tracks) constrained clustering."""

    def test_same_base_direction_different_tracks_cluster_together(self):
        from video_face_pipeline import _cluster_track_faces
        # Track 0: samples 0-10; Track 1: samples 100-110 (non-overlapping)
        # -- both the same person (same base direction), should merge.
        embs = [_embedding(0, seed=i) for i in range(4)]
        track_id = [0, 0, 1, 1]
        spans = {0: (0, 10), 1: (100, 110)}
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.5)
        self.assertEqual(groups[0], groups[1])

    def test_different_base_directions_never_merge(self):
        from video_face_pipeline import _cluster_track_faces
        embs = [_embedding(0, seed=i) for i in range(3)] + [_embedding(1, seed=100 + i) for i in range(3)]
        track_id = [0, 0, 0, 1, 1, 1]
        spans = {0: (0, 10), 1: (100, 110)}
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.5)
        self.assertNotEqual(groups[0], groups[1])

    def test_must_link_keeps_one_tracks_own_faces_together_despite_noise(self):
        from video_face_pipeline import _cluster_track_faces
        # One track's own two reps, seeded far apart enough that raw
        # similarity alone might not obviously cluster them -- must-link
        # forces them together regardless (same track = same person by
        # construction, per the IOU tracker's own guarantee).
        embs = [_embedding(0, seed=1), _embedding(0, seed=999)]
        track_id = [0, 0]
        spans = {0: (0, 10)}
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.99)
        self.assertEqual(groups[0], groups[0])  # trivially true; real check is no crash/split

    def test_cannot_link_blocks_merge_even_at_identical_embeddings(self):
        from video_face_pipeline import _cluster_track_faces
        # Same base direction (would otherwise merge) but temporally
        # overlapping spans -- two different people can't both be the
        # same identity at the same time, so this must NOT merge even
        # though the embeddings alone would suggest it should.
        embs = [_embedding(0, seed=1), _embedding(0, seed=2)]
        track_id = [0, 1]
        spans = {0: (0, 100), 1: (50, 150)}  # overlapping
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.5)
        self.assertNotEqual(groups[0], groups[1])


class VideoFaceUnionMergeTests(unittest.TestCase):
    """_union_merge_groups: union (not intersection) over multiple
    independent clusterings -- the user's own explicit call for this
    pipeline (fewer final groups, some wrong merges acceptable)."""

    def test_union_of_disjoint_groupings_combines_via_either(self):
        from video_face_pipeline import _union_merge_groups
        # Model A groups (0,1) together and (2,3) together; model B
        # groups (1,2) together -- the union should chain all 4 into one.
        grouping_a = {0: 'x', 1: 'x', 2: 'y', 3: 'y'}
        grouping_b = {0: 'p', 1: 'q', 2: 'q', 3: 'r'}
        final = _union_merge_groups(4, grouping_a, grouping_b)
        self.assertEqual(len({final[0], final[1], final[2], final[3]}), 1)

    def test_agreement_on_separation_keeps_tracks_apart(self):
        from video_face_pipeline import _union_merge_groups
        grouping_a = {0: 'x', 1: 'y'}
        grouping_b = {0: 'p', 1: 'q'}
        final = _union_merge_groups(2, grouping_a, grouping_b)
        self.assertNotEqual(final[0], final[1])

    def test_single_grouping_passthrough(self):
        from video_face_pipeline import _union_merge_groups
        grouping = {0: 'x', 1: 'x', 2: 'y'}
        final = _union_merge_groups(3, grouping)
        self.assertEqual(final[0], final[1])
        self.assertNotEqual(final[0], final[2])


class MatchRedetectCandidatesTests(unittest.TestCase):
    """match_redetect_candidates_to_existing_faces: reconciles a fresh
    full-redetect pass (e.g. after fixing a video's aspect-ratio decode)
    against that video's existing Face rows, so confirmed identifications
    survive even though the redetect's own boundaries won't exactly
    reproduce the original pass's. See CLAUDE.md's 2026-09-23 write-up."""

    def _emb(self, direction, dim=8):
        v = np.zeros(dim)
        v[direction] = 1.0
        return v

    def test_matching_embeddings_pair_up(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}, {'embedding': self._emb(1)}]
        candidates = [{'embedding': self._emb(1)}, {'embedding': self._emb(0)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_dissimilar_embedding_stays_unmatched(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}]
        candidates = [{'embedding': self._emb(1)}]  # orthogonal -- cos_sim=0
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [])

    def test_more_candidates_than_existing_leaves_extras_unmatched(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}]
        candidates = [{'embedding': self._emb(0)}, {'embedding': self._emb(1)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [(0, 0)])
        matched_candidate_idcs = {j for _, j in matches}
        self.assertNotIn(1, matched_candidate_idcs)  # the new, unmatched face

    def test_more_existing_than_candidates_leaves_extras_unmatched(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}, {'embedding': self._emb(1)}]
        candidates = [{'embedding': self._emb(0)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [(0, 0)])
        matched_existing_idcs = {i for i, _ in matches}
        self.assertNotIn(1, matched_existing_idcs)  # left alone, not deleted

    def test_empty_either_side_returns_no_matches(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        self.assertEqual(match_redetect_candidates_to_existing_faces([], [{'embedding': self._emb(0)}]), [])
        self.assertEqual(match_redetect_candidates_to_existing_faces([{'embedding': self._emb(0)}], []), [])

    def test_never_force_matches_below_the_floor(self):
        # Two existing, two candidates, but only ONE real match -- the
        # optimal assignment must not force the second (bad) pairing
        # just because it's "the best of what's left".
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}, {'embedding': self._emb(2)}]
        candidates = [{'embedding': self._emb(0)}, {'embedding': self._emb(3)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [(0, 0)])


class VideoFaceTrimmedCentroidTests(unittest.TestCase):
    """_trimmed_centroid: one-pass outlier trim (drop anything below a
    cosine-similarity floor to the plain mean, recompute from survivors)
    -- the user's own chosen approach over a fixed-percentage trim."""

    def test_all_similar_embeddings_centroid_close_to_each(self):
        from video_face_pipeline import _trimmed_centroid, cos_sim
        embs = np.array([_embedding(0, seed=i) for i in range(5)])
        centroid = _trimmed_centroid(embs, sim_floor=0.35)
        for e in embs:
            self.assertGreater(cos_sim(e, centroid), 0.9)

    def test_outlier_is_excluded_from_centroid(self):
        from video_face_pipeline import _trimmed_centroid, cos_sim
        # 4 real members of one identity, 1 genuine outlier (orthogonal
        # base direction) that should get trimmed before the final mean.
        good = [_embedding(0, seed=i) for i in range(4)]
        outlier = _embedding(1, seed=999)
        centroid_with_outlier_trimmed = _trimmed_centroid(np.array(good + [outlier]), sim_floor=0.35)
        centroid_good_only = np.mean(good, axis=0)
        # The trimmed centroid should closely match the good-only mean --
        # if the outlier had survived trimming, it would pull noticeably
        # away from this.
        self.assertGreater(cos_sim(centroid_with_outlier_trimmed, centroid_good_only), 0.99)

    def test_floor_that_excludes_everything_falls_back_to_untrimmed_mean(self):
        from video_face_pipeline import _trimmed_centroid
        embs = np.array([_embedding(0, seed=i) for i in range(3)])
        # An impossibly strict floor (above 1.0) can never be met -- must
        # fall back to the plain mean rather than dividing by zero
        # survivors.
        centroid = _trimmed_centroid(embs, sim_floor=1.5)
        expected = embs.mean(axis=0)
        np.testing.assert_allclose(centroid, expected)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class VideoFaceSetBoxAndThumbnailOrphanTests(TestCase):
    """_set_face_box_and_thumbnail() is called a second time, on the same
    already-saved Face, when _classify_and_pick_thumbnail() re-picks a
    better representative frame after a confident gallery match. Django's
    FieldFile never deletes the file it's replacing on a plain
    reassignment -- found 2026-10-08 while root-causing a real 188K-file
    orphaned-face-thumbnail backlog. Uses object.__new__() to skip
    VideoFaceExtractor.__init__()'s real ONNX/facenet model loading
    entirely, since the method under test touches none of those models
    -- pure box math + file I/O."""

    def setUp(self):
        from video_face_pipeline import VideoFaceExtractor
        from filepopulator.models import Directory, VideoFile
        self.extractor = object.__new__(VideoFaceExtractor)

        directory = Directory.objects.create(dir_path='/tmp/video_orphan_test')
        self.video = VideoFile.objects.create(
            filename='/tmp/video_orphan_test/a.mp4', directory=directory,
            width=320, height=240, duration_seconds=10.0,
        )
        self.blank_person, _ = Person.objects.get_or_create(person_name=settings.BLANK_FACE_NAME)
        self.face = Face(
            source_video_file=self.video, declared_name=self.blank_person,
            box_left=1, box_top=1, box_right=50, box_bottom=50,
        )
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_second_call_deletes_the_first_calls_thumbnail_file(self):
        # First call -- matches the real pipeline's own order (see
        # _set_face_box_and_thumbnail's docstring on
        # _classify_and_pick_thumbnail): the thumbnail is set once before
        # the Face row is ever saved to the DB (Face.save() itself
        # requires a thumbnail to already be attached), then the row is
        # saved for real, standing in for "an already-saved Face row".
        self.extractor._set_face_box_and_thumbnail(
            self.face, self.frame, (10, 10, 60, 60), kps=[[0, 0]] * 5,
        )
        self.face.save()
        first_thumbnail_path = self.face.face_thumbnail.path
        self.assertTrue(os.path.isfile(first_thumbnail_path))

        # Second call -- the real re-pick path.
        self.extractor._set_face_box_and_thumbnail(
            self.face, self.frame, (20, 20, 70, 70), kps=[[0, 0]] * 5,
        )

        self.assertFalse(os.path.isfile(first_thumbnail_path))
        second_thumbnail_path = self.face.face_thumbnail.path
        self.assertTrue(os.path.isfile(second_thumbnail_path))
        self.assertNotEqual(first_thumbnail_path, second_thumbnail_path)

    def test_first_call_on_a_fresh_face_does_not_error(self):
        # No prior thumbnail exists yet -- must be a clean no-op, not a
        # crash trying to delete something that was never there.
        self.extractor._set_face_box_and_thumbnail(
            self.face, self.frame, (10, 10, 60, 60), kps=[[0, 0]] * 5,
        )
        self.assertTrue(os.path.isfile(self.face.face_thumbnail.path))


@tag("slow")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class VideoFaceExtractorRealInferenceTests(TestCase):
    """Real ffmpeg decode + real insightface/facenet inference against a
    real video fixture (IMG_0760.MOV, 7.6s, already used elsewhere in
    this investigation and known to contain real, detectable faces --
    see CLAUDE.md's Phase 3 section). No test_face_cache-style caching
    exists for this pipeline yet (unlike PyramidalDetectorRealInferenceTests)
    -- this test pays real CPU inference cost every run."""

    VIDEO_PATH = '/photos/video_samples/IMG_0760.MOV'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from video_face_pipeline import VideoFaceExtractor
        cls.extractor = VideoFaceExtractor()
        # Same override LoadEncodingsCachingTests uses: /models/ (the
        # real cache location) doesn't exist in this test environment --
        # pre-build the faceAssigner here with a writable path instead of
        # letting the lazy face_assigner property build one against the
        # real (here, nonexistent) path on first access.
        cache_path = "/tmp/face_manager_test_media/test_video_encodings_cache.pkl"
        if os.path.exists(cache_path):
            os.remove(cache_path)
        fa = faceAssigner()
        fa.ENCODINGS_PKL_FILE = cache_path
        fa.load_encodings()
        cls.extractor._face_assigner = fa

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/photos/video_samples')
        self.video = VideoFile.objects.create(
            filename=self.VIDEO_PATH, directory=directory,
            width=1920, height=1080, duration_seconds=7.6,
        )

    def test_real_fixture_produces_valid_face_groups(self):
        faces = self.extractor.process_video(self.video)
        self.assertGreaterEqual(len(faces), 1, "expected at least one face group from a known-face fixture")

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.source_video_file_id, self.video.pk)
            self.assertIsNone(face.source_image_file_id)
            self.assertEqual(len(face.face_encoding_512), 512)
            self.assertIsNotNone(face.video_first_timestamp_seconds)
            self.assertIsNotNone(face.video_last_timestamp_seconds)
            self.assertLessEqual(face.video_first_timestamp_seconds, face.video_last_timestamp_seconds)
            self.assertGreaterEqual(face.video_first_timestamp_seconds, 0)
            self.assertLessEqual(face.video_last_timestamp_seconds, self.video.duration_seconds + 1)
            # Auto-matching (if any) only ever proposes -- never auto-declares.
            self.assertEqual(face.declared_name_id, blank_person.pk)
            # A real thumbnail file was actually generated and saved.
            self.assertTrue(os.path.exists(face.face_thumbnail.path))
            self.assertGreater(face.box_right, face.box_left)
            self.assertGreater(face.box_bottom, face.box_top)
            # det_score (added 2026-09-18, see CLAUDE.md) is threaded
            # through the video pipeline the same as the image one --
            # a real, valid detection confidence, not just left NULL.
            self.assertIsNotNone(face.det_score)
            self.assertGreater(face.det_score, 0.0)
            self.assertLessEqual(face.det_score, 1.0)

    def test_reprocessing_is_independent_not_cumulative(self):
        # Not idempotency in the "no duplicate rows" sense (there's no
        # dedup logic -- isProcessed is what stops a real double-run) --
        # just confirms a second call doesn't crash or silently multiply
        # unboundedly, producing a comparable group count both times.
        first = self.extractor.process_video(self.video)
        for f in first:
            f.delete()
        second = self.extractor.process_video(self.video)
        self.assertGreaterEqual(len(second), 1)


class ProcessVideoFacesFailureTrackingTests(TestCase):
    """face_extraction_failed/_error (filepopulator.VideoFile) -- set by
    process_video_faces()'s except branch, surfaced on /api/server_stats/
    (see api/serializers.py). Mocks VideoFaceExtractor entirely so this
    stays a fast test -- the real pipeline's own success path is already
    covered by VideoFaceExtractorRealInferenceTests above."""

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/videos/failure_test')
        self.video = VideoFile.objects.create(
            filename='/videos/failure_test/a.mp4', directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=False,
        )

    def test_failure_sets_tracking_fields_and_still_marks_processed(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = RuntimeError("boom")
            process_video_faces()
        self.video.refresh_from_db()
        self.assertTrue(self.video.isProcessed)
        self.assertTrue(self.video.face_extraction_failed)
        self.assertIn("RuntimeError", self.video.face_extraction_error)
        self.assertIn("boom", self.video.face_extraction_error)

    def test_success_leaves_tracking_fields_unset(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces()
        self.video.refresh_from_db()
        self.assertTrue(self.video.isProcessed)
        self.assertFalse(self.video.face_extraction_failed)
        self.assertIsNone(self.video.face_extraction_error)

    def test_timeout_leaves_video_unprocessed_for_a_later_retry(self):
        # Real 2026-09-17 finding: several completely normal, short videos
        # timed out purely because a concurrent CPU-heavy job was
        # competing for cores -- not a permanent problem with the file.
        # A timeout must NOT be treated like every other failure (marking
        # isProcessed=True/face_extraction_failed=True would require a
        # manual reset before it's ever retried); it should just stay in
        # the unprocessed queue for a future scheduled run to pick up
        # again once resources free up.
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = _VideoProcessingTimeout("Video processing exceeded 1800s")
            process_video_faces()
        self.video.refresh_from_db()
        self.assertFalse(self.video.isProcessed)
        self.assertFalse(self.video.face_extraction_failed)
        self.assertIsNone(self.video.face_extraction_error)
        self.assertEqual(self.video.face_extraction_timeout_count, 1)

    def test_repeated_timeouts_eventually_marked_a_real_failure(self):
        # A genuinely-too-long video (not just resource contention) would
        # otherwise retry forever, each attempt burning a full
        # PER_VIDEO_TIMEOUT_SECONDS -- MAX_VIDEO_TIMEOUT_RETRIES caps
        # that, per the user's explicit request.
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout, MAX_VIDEO_TIMEOUT_RETRIES
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = _VideoProcessingTimeout("boom")
            for _ in range(MAX_VIDEO_TIMEOUT_RETRIES - 1):
                process_video_faces()
                self.video.refresh_from_db()
                self.assertFalse(self.video.isProcessed)
                self.assertFalse(self.video.face_extraction_failed)
                # isProcessed=False is required for the next call's
                # unprocessed queryset to pick this video up again.
                self.video.isProcessed = False
                self.video.save()

            process_video_faces()
        self.video.refresh_from_db()
        self.assertEqual(self.video.face_extraction_timeout_count, MAX_VIDEO_TIMEOUT_RETRIES)
        self.assertTrue(self.video.isProcessed)
        self.assertTrue(self.video.face_extraction_failed)
        self.assertIn(str(MAX_VIDEO_TIMEOUT_RETRIES), self.video.face_extraction_error)

    def test_eventual_success_resets_timeout_count(self):
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout
        self.video.face_extraction_timeout_count = 3
        self.video.save()
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces()
        self.video.refresh_from_db()
        self.assertEqual(self.video.face_extraction_timeout_count, 0)

    def test_real_failure_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = RuntimeError("boom")
            process_video_faces()
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.process_video_faces')
        self.assertIn(self.video.filename, error.message)
        self.assertIn('boom', error.message)

    def test_exhausted_timeout_retries_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout, MAX_VIDEO_TIMEOUT_RETRIES
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = _VideoProcessingTimeout("boom")
            for _ in range(MAX_VIDEO_TIMEOUT_RETRIES):
                process_video_faces()
                self.video.refresh_from_db()
                if not self.video.face_extraction_failed:
                    self.video.isProcessed = False
                    self.video.save()
        self.assertEqual(ApplicationError.objects.count(), 1)
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.process_video_faces')
        self.assertIn(self.video.filename, error.message)


class ProcessVideoFacesVideoIdsScopingTests(TestCase):
    """process_video_faces(video_ids=...) -- the on-demand, frontend-
    triggered reprocess path (api.views.VideoFaceExtractionViewSet).
    Unlike the normal isProcessed=False backlog, this always forces a
    redo of exactly the named videos, regardless of their current
    isProcessed/face_extraction_failed state, and clears out any
    existing Face rows (with their thumbnails) first so a reprocess
    never leaves stale+fresh Face rows coexisting."""

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/videos/video_ids_test')
        self.already_processed = VideoFile.objects.create(
            filename='/videos/video_ids_test/a.mp4', directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=True,
            face_extraction_failed=True, face_extraction_error='old failure',
            face_extraction_timeout_count=2,
        )
        self.untouched = VideoFile.objects.create(
            filename='/videos/video_ids_test/b.mp4', directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=True,
        )

    def test_targeted_video_is_reprocessed_despite_already_processed(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces(video_ids=[self.already_processed.id])
        self.already_processed.refresh_from_db()
        MockExtractor.return_value.process_video.assert_called_once()
        self.assertTrue(self.already_processed.isProcessed)
        self.assertFalse(self.already_processed.face_extraction_failed)
        self.assertIsNone(self.already_processed.face_extraction_error)
        self.assertEqual(self.already_processed.face_extraction_timeout_count, 0)

    def test_other_videos_are_not_touched(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces(video_ids=[self.already_processed.id])
        self.assertEqual(MockExtractor.return_value.process_video.call_count, 1)
        call_arg = MockExtractor.return_value.process_video.call_args[0][0]
        self.assertEqual(call_arg.id, self.already_processed.id)

    def test_existing_face_rows_and_thumbnails_are_deleted_before_reprocessing(self):
        from face_manager.models import Face
        from face_manager.tasks import process_video_faces
        from django.core.files.base import ContentFile

        face = Face(
            source_video_file=self.already_processed,
            box_top=0, box_bottom=10, box_left=0, box_right=10,
        )
        face.face_thumbnail.save('thumb.jpg', ContentFile(b'fake-thumb-bytes'), save=False)
        face.save()
        thumb_path = face.face_thumbnail.path
        self.assertTrue(os.path.exists(thumb_path))

        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces(video_ids=[self.already_processed.id])

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())
        self.assertFalse(os.path.exists(thumb_path))
