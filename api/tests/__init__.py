import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
import zipfile
from datetime import datetime, time
from datetime import timezone as dt_timezone
from io import BytesIO
from unittest import mock

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils.functional import SimpleLazyObject
from rest_framework import status
from rest_framework.test import APIClient

from api.models import ScheduleRule, SlideshowConfig, resolve_active_slideshow_config
from api.throttles import AccessKeyMediaThrottle
from face_manager.models import Face, Person
from filepopulator.models import GeocodeCache, ImageFile, Directory, VideoFile
from filepopulator.scripts import create_image_file

# This module intentionally never imports `api.views` (or triggers URL
# resolution) at import time. api/views.py runs a module-level query for
# the '.ignore' / '.realignore' Person rows the first time it's imported,
# so those rows must exist before the *first* request in the whole test
# run touches an api/ URL. ApiTestCase.setUpClass creates them, and
# setUpClass always runs before any test method body (where the first
# `self.client.get(...)` call actually happens).


def _tiny_jpeg_bytes(size=(50, 50), color=(0, 0, 0)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:, :] = color
    ok, buf = cv2.imencode(".jpg", img)
    return BytesIO(buf).read()


def ensure_sentinel_people():
    """Create the well-known Person rows api/views.py and face_manager
    assume exist (BLANK_FACE_NAME, '.ignore', '.realignore', etc).
    Nothing in the codebase creates these via a migration or fixture --
    on the live system they were created by hand at some point. This is
    a real gap in the app, not something to paper over with app-code
    changes here."""
    for name in set(settings.IGNORED_NAMES):
        if Person.objects.filter(person_name=name).exists():
            continue
        p = Person(person_name=name)
        p.highlight_img.save(
            f"{name.strip('.') or 'sentinel'}_sentinel.jpg",
            ContentFile(_tiny_jpeg_bytes()),
            save=False,
        )
        p.save()


class FaceFixtureMixin:
    """Shared fixture helpers (make_image/make_face) - pulled out of
    ApiTestCase so a test that specifically needs TransactionTestCase
    (real commits, visible to other DB connections/threads - see
    IgnoreReviewFlaggedCountTests) can reuse them without inheriting
    TestCase's rolled-back-transaction behavior."""

    def make_image(self, relative_fixture="naming/good/1.JPG"):
        """Create a real ImageFile row (with real thumbnails on disk) from
        one of the repo's filepopulator test fixture images."""
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/{relative_fixture}"
        create_image_file(path)
        return ImageFile.objects.get(filename=path)

    def make_face(self, image_file, declared_name=None, **overrides):
        """Create a Face row pointing at a real (small) thumbnail file on
        disk, since Face.save() validates the thumbnail actually exists
        and that the box fits inside the source image."""
        if declared_name is None:
            declared_name, created = Person.objects.get_or_create(person_name="Test Person")
            if created:
                declared_name.highlight_img.save(
                    "test_person_sentinel.jpg", ContentFile(_tiny_jpeg_bytes()), save=True
                )

        w, h = image_file.width, image_file.height
        box = dict(box_left=1, box_top=1, box_right=min(40, w - 1), box_bottom=min(40, h - 1))
        box.update({k: v for k, v in overrides.items() if k in box})
        remaining = {k: v for k, v in overrides.items() if k not in box}

        face = Face(declared_name=declared_name, source_image_file=image_file, **box)
        for k, v in remaining.items():
            setattr(face, k, v)

        face.face_thumbnail.save(
            "face_thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False
        )
        face.save()
        return face


@override_settings(MEDIA_ROOT="/tmp/api_test_media")
class ApiTestCase(FaceFixtureMixin, TestCase):
    """Base class for api/ tests: authenticated client + sentinel people."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ensure_sentinel_people()

    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="pw123456")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.anon_client = APIClient()


