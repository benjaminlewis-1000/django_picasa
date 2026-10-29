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
from contextlib import contextmanager
import zlib
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import psycopg2
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.db import connection, transaction
from django.db.utils import IntegrityError
from django.test import TestCase, TransactionTestCase, override_settings, tag

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


