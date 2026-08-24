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
from io import BytesIO

import cv2
import numpy as np
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings, tag

from face_manager.face_extract_encode import FaceExtractor
from face_manager.models import Face, Person, get_default_blank_person
from face_manager.pyramidal_detector import PyramidalDetector
from face_manager.test_face_cache import cached_detect
from filepopulator.models import Directory, ImageFile
from filepopulator.scripts import create_image_file


def _tiny_jpeg_bytes(size=(50, 50)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return BytesIO(buf).read()


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

    def test_deleting_image_cascades_to_faces(self):
        face = make_face(self.image)
        face_id = face.id
        self.image.delete()
        self.assertFalse(Face.objects.filter(id=face_id).exists())

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

    def test_reject_association_removes_from_possibles(self):
        person = make_person("Rejectable")
        face = make_face(self.image)
        face.set_possible_person(person.id, 1, 0.75)
        face.reject_association(person.id)
        face.refresh_from_db()
        self.assertIsNone(face.poss_ident1)
        self.assertIn(person.id, face.rejected_fields)

    def test_clear_person(self):
        person = make_person("Clearable")
        face = make_face(self.image, declared_name=person)
        face.clear_person()
        face.refresh_from_db()
        self.assertIsNone(face.declared_name)


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
