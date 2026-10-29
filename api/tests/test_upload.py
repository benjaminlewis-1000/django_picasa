"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_upload
"""
import hashlib
import json
import math
import os
import random
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

from api.tests import FaceFixtureMixin, ApiTestCase, _tiny_jpeg_bytes, ensure_sentinel_people


def _sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def _make_zip(entries):
    """entries: list of (name_in_zip, bytes). Returns raw zip bytes."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


@override_settings(UPLOAD_MAX_FILE_SIZE_BYTES=10 * 1024 * 1024,
                    UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES=10 * 1024 * 1024,
                    UPLOAD_MAX_ZIP_ENTRY_COUNT=5000)
class UploadFileViewTests(ApiTestCase):
    """api/upload_views.py -- authenticated file upload with real content
    verification (not just extension/Content-Type), checksum enforcement,
    and zip handling (zip-slip safety, nested-zip/non-media skipping,
    size/entry-count guards)."""

    def setUp(self):
        super().setUp()
        self._staging_dir = tempfile.mkdtemp(prefix='upload_test_staging_')
        self._settings_override = override_settings(UPLOAD_STAGING_DIR=self._staging_dir)
        self._settings_override.enable()
        self.addCleanup(self._settings_override.disable)
        self.addCleanup(shutil.rmtree, self._staging_dir, True)

    def _staged_files(self):
        """Every file path under the staging dir, relative to it."""
        found = []
        for root, _dirs, files in os.walk(self._staging_dir):
            for f in files:
                found.append(os.path.relpath(os.path.join(root, f), self._staging_dir))
        return found

    def _post_upload(self, client, filename, data, checksum=None):
        upload = SimpleUploadedFile(filename, data)
        payload = {'file': upload}
        if checksum is not None:
            payload['checksum'] = checksum
        return client.post('/api/upload/', payload, format='multipart')

    def test_requires_authentication(self):
        data = _tiny_jpeg_bytes()
        resp = self._post_upload(self.anon_client, 'photo.jpg', data, _sha256_hex(data))
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))
        self.assertEqual(self._staged_files(), [])

    def test_missing_checksum_rejected(self):
        data = _tiny_jpeg_bytes()
        resp = self._post_upload(self.client, 'photo.jpg', data)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])

    def test_wrong_checksum_rejected_and_nothing_written(self):
        data = _tiny_jpeg_bytes()
        resp = self._post_upload(self.client, 'photo.jpg', data, checksum='0' * 64)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('checksum', json.loads(resp.content)['error'].lower())
        self.assertEqual(self._staged_files(), [])

    def test_valid_image_is_accepted_and_staged(self):
        data = _tiny_jpeg_bytes()
        resp = self._post_upload(self.client, 'photo.jpg', data, _sha256_hex(data))
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        staged = self._staged_files()
        self.assertEqual(len(staged), 1)
        # Flat -- lands directly in the staging root, not a subdirectory.
        self.assertEqual(staged[0], 'photo.jpg')

    def test_unsupported_extension_rejected(self):
        data = b'hello world'
        resp = self._post_upload(self.client, 'notes.txt', data, _sha256_hex(data))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])

    def test_text_file_renamed_as_jpg_is_rejected(self):
        # The core "don't trust the extension" guard -- a real decode
        # attempt catches this even though the filename/extension look
        # like a legitimate image.
        data = b'this is definitely not a real jpeg\x00\x01\x02'
        resp = self._post_upload(self.client, 'fake.jpg', data, _sha256_hex(data))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])

    def test_valid_video_is_accepted_and_staged(self):
        # ci_fixtures/video_stub/synthetic.mp4 is committed in-repo (see
        # ci_fixtures/generate_fixtures.py) specifically so tests like
        # this one work the same in CI as locally, with no dependency on
        # any real-sample mount.
        video_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'ci_fixtures', 'video_stub', 'synthetic.mp4',
        )
        with open(video_path, 'rb') as f:
            data = f.read()
        resp = self._post_upload(self.client, 'clip.mp4', data, _sha256_hex(data))
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        staged = self._staged_files()
        self.assertEqual(len(staged), 1)
        self.assertTrue(staged[0].endswith('clip.mp4'))

    def test_file_over_size_limit_rejected(self):
        data = _tiny_jpeg_bytes()
        with override_settings(UPLOAD_MAX_FILE_SIZE_BYTES=len(data) - 1):
            resp = self._post_upload(self.client, 'photo.jpg', data, _sha256_hex(data))
        self.assertEqual(resp.status_code, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        self.assertEqual(self._staged_files(), [])

    def test_zip_with_mixed_valid_and_invalid_entries(self):
        good1 = _tiny_jpeg_bytes(color=(10, 10, 10))
        good2 = _tiny_jpeg_bytes(color=(20, 20, 20))
        bad = b'not an image'
        zip_bytes = _make_zip([
            ('photos/a.jpg', good1),
            ('photos/b.jpg', good2),
            ('notes.txt', bad),
            ('nested.zip', _make_zip([('inner.jpg', good1)])),
        ])
        resp = self._post_upload(self.client, 'batch.zip', zip_bytes, _sha256_hex(zip_bytes))
        self.assertEqual(resp.status_code, status.HTTP_207_MULTI_STATUS)
        data = json.loads(resp.content)
        by_status = {}
        for entry in data['files']:
            by_status.setdefault(entry['status'], []).append(entry['filename'])
        self.assertEqual(sorted(by_status['accepted']), ['photos/a.jpg', 'photos/b.jpg'])
        self.assertIn('notes.txt', by_status['skipped'])
        self.assertIn('nested.zip', by_status['skipped'])
        staged = self._staged_files()
        self.assertEqual(len(staged), 2)
        self.assertTrue(any(f.endswith('a.jpg') for f in staged))
        self.assertTrue(any(f.endswith('b.jpg') for f in staged))

    def test_zip_all_entries_invalid_is_rejected(self):
        zip_bytes = _make_zip([('notes.txt', b'not an image')])
        resp = self._post_upload(self.client, 'batch.zip', zip_bytes, _sha256_hex(zip_bytes))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])

    def test_zip_slip_path_traversal_entry_lands_safely_inside_staging_dir(self):
        # A traversal-style entry name is stripped down to its basename
        # before ever being joined to a real path, so this lands safely
        # inside the per-request subdirectory as a plain "evil.jpg" --
        # never at the traversal-implied location outside the staging
        # tree entirely.
        good = _tiny_jpeg_bytes()
        zip_bytes = _make_zip([('../../evil.jpg', good)])
        resp = self._post_upload(self.client, 'batch.zip', zip_bytes, _sha256_hex(zip_bytes))
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        staged = self._staged_files()
        self.assertEqual(len(staged), 1)
        self.assertTrue(staged[0].endswith('evil.jpg'))
        # And, critically, nothing was written outside the staging dir at
        # the literal traversal-implied path.
        parent = os.path.dirname(self._staging_dir.rstrip('/'))
        self.assertFalse(os.path.exists(os.path.join(parent, 'evil.jpg')))

    def test_zip_filename_collision_across_subfolders_is_disambiguated(self):
        good1 = _tiny_jpeg_bytes(color=(1, 1, 1))
        good2 = _tiny_jpeg_bytes(color=(2, 2, 2))
        zip_bytes = _make_zip([
            ('folderA/photo.jpg', good1),
            ('folderB/photo.jpg', good2),
        ])
        resp = self._post_upload(self.client, 'batch.zip', zip_bytes, _sha256_hex(zip_bytes))
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        staged = self._staged_files()
        self.assertEqual(len(staged), 2)
        # Both accepted, and both files genuinely exist on disk with
        # distinct names (no silent overwrite of one by the other).
        basenames = [os.path.basename(f) for f in staged]
        self.assertEqual(len(set(basenames)), 2)

    def test_zip_entry_count_over_limit_rejected(self):
        entries = [(f'{i}.jpg', _tiny_jpeg_bytes()) for i in range(5)]
        zip_bytes = _make_zip(entries)
        with override_settings(UPLOAD_MAX_ZIP_ENTRY_COUNT=3):
            resp = self._post_upload(self.client, 'batch.zip', zip_bytes, _sha256_hex(zip_bytes))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])

    def test_zip_uncompressed_size_over_limit_rejected(self):
        data = _tiny_jpeg_bytes(size=(200, 200))
        zip_bytes = _make_zip([('big.jpg', data)])
        with override_settings(UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES=len(data) - 1):
            resp = self._post_upload(self.client, 'batch.zip', zip_bytes, _sha256_hex(zip_bytes))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])

    def test_not_a_real_zip_is_rejected(self):
        data = b'PK\x03\x04not actually a zip body'
        resp = self._post_upload(self.client, 'batch.zip', data, _sha256_hex(data))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(self._staged_files(), [])


@override_settings(UPLOAD_MAX_FILE_SIZE_BYTES=10 * 1024 * 1024, UPLOAD_CHUNK_SIZE_BYTES=16)
class ChunkedUploadTests(ApiTestCase):
    """api/upload_views.py's chunked-upload flow (init/chunk/status/
    complete) -- for uploads large/slow enough that a single request
    risks exceeding a reasonable timeout. UPLOAD_CHUNK_SIZE_BYTES=16
    keeps test data tiny while still exercising multi-chunk splitting."""

    def setUp(self):
        super().setUp()
        self._staging_dir = tempfile.mkdtemp(prefix='chunked_upload_test_staging_')
        self._chunk_dir = tempfile.mkdtemp(prefix='chunked_upload_test_chunks_')
        self._settings_override = override_settings(
            UPLOAD_STAGING_DIR=self._staging_dir,
            UPLOAD_CHUNK_SCRATCH_DIR=self._chunk_dir,
        )
        self._settings_override.enable()
        self.addCleanup(self._settings_override.disable)
        self.addCleanup(shutil.rmtree, self._staging_dir, True)
        self.addCleanup(shutil.rmtree, self._chunk_dir, True)

    def _staged_files(self):
        found = []
        for root, _dirs, files in os.walk(self._staging_dir):
            for f in files:
                found.append(os.path.relpath(os.path.join(root, f), self._staging_dir))
        return found

    def _init(self, filename, data):
        return self.client.post('/api/upload/chunked/init/', {
            'filename': filename, 'total_size': len(data), 'checksum': _sha256_hex(data),
        }, format='json')

    def _chunks_of(self, data, chunk_size=16):
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def _put_chunk(self, upload_id, index, chunk_bytes, client=None, checksum=None):
        client = client or self.client
        upload = SimpleUploadedFile(f'{index}.part', chunk_bytes)
        checksum = checksum if checksum is not None else _sha256_hex(chunk_bytes)
        return client.put(
            f'/api/upload/chunked/{upload_id}/chunk/{index}/',
            {'chunk': upload, 'checksum': checksum}, format='multipart',
        )

    def test_init_computes_total_chunks_correctly(self):
        data = _tiny_jpeg_bytes()  # comfortably bigger than the 16-byte test chunk size
        resp = self._init('photo.jpg', data)
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        body = json.loads(resp.content)
        self.assertEqual(body['chunk_size'], 16)
        self.assertEqual(body['total_chunks'], math.ceil(len(data) / 16))

    def test_init_rejects_unsupported_extension(self):
        resp = self._init('notes.txt', b'hello')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_init_rejects_oversized_declared_total(self):
        with override_settings(UPLOAD_MAX_FILE_SIZE_BYTES=10):
            resp = self._init('photo.jpg', b'x' * 100)
        self.assertEqual(resp.status_code, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

    def test_full_roundtrip_reassembles_and_stages_correctly(self):
        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']
        chunks = self._chunks_of(data)
        self.assertEqual(len(chunks), init_body['total_chunks'])

        for i, c in enumerate(chunks):
            resp = self._put_chunk(upload_id, i, c)
            self.assertEqual(resp.status_code, status.HTTP_200_OK)

        status_resp = self.client.get(f'/api/upload/chunked/{upload_id}/status/')
        status_body = json.loads(status_resp.content)
        self.assertEqual(status_body['received_chunks'], list(range(len(chunks))))

        complete_resp = self.client.post(f'/api/upload/chunked/{upload_id}/complete/')
        self.assertEqual(complete_resp.status_code, status.HTTP_201_CREATED)

        staged = self._staged_files()
        self.assertEqual(len(staged), 1)
        with open(os.path.join(self._staging_dir, staged[0]), 'rb') as f:
            self.assertEqual(f.read(), data)

    def test_complete_before_all_chunks_received_reports_missing(self):
        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']
        chunks = self._chunks_of(data)
        # Only send the first chunk.
        self._put_chunk(upload_id, 0, chunks[0])

        resp = self.client.post(f'/api/upload/chunked/{upload_id}/complete/')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        body = json.loads(resp.content)
        self.assertEqual(body['missing_chunks'], list(range(1, len(chunks))))
        self.assertEqual(self._staged_files(), [])

    def test_chunk_with_wrong_checksum_is_rejected_and_not_recorded(self):
        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']
        chunks = self._chunks_of(data)

        resp = self._put_chunk(upload_id, 0, chunks[0], checksum='0' * 64)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

        status_resp = self.client.get(f'/api/upload/chunked/{upload_id}/status/')
        self.assertEqual(json.loads(status_resp.content)['received_chunks'], [])

    def test_resending_same_chunk_index_is_idempotent(self):
        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']
        chunks = self._chunks_of(data)

        self._put_chunk(upload_id, 0, chunks[0])
        resp = self._put_chunk(upload_id, 0, chunks[0])  # resend, same content
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(resp.content)['received_chunks'], [0])

    def test_chunk_index_out_of_range_rejected(self):
        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']
        resp = self._put_chunk(upload_id, init_body['total_chunks'] + 5, b'x' * 4)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_oversized_chunk_rejected(self):
        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']
        resp = self._put_chunk(upload_id, 0, b'x' * 17)  # chunk size cap is 16
        self.assertEqual(resp.status_code, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

    def test_wrong_final_checksum_marks_session_failed(self):
        # Every individual chunk checksum is correct, but the whole-file
        # checksum declared at init doesn't match the real reassembled
        # bytes -- the defense-in-depth check at complete-time must still
        # catch this.
        from api.models import UploadSession
        data = _tiny_jpeg_bytes(size=(60, 60))
        resp = self.client.post('/api/upload/chunked/init/', {
            'filename': 'photo.jpg', 'total_size': len(data), 'checksum': '0' * 64,
        }, format='json')
        upload_id = json.loads(resp.content)['upload_id']
        for i, c in enumerate(self._chunks_of(data)):
            self._put_chunk(upload_id, i, c)

        complete_resp = self.client.post(f'/api/upload/chunked/{upload_id}/complete/')
        self.assertEqual(complete_resp.status_code, status.HTTP_400_BAD_REQUEST)
        session = UploadSession.objects.get(upload_id=upload_id)
        self.assertEqual(session.status, UploadSession.STATUS_FAILED)
        self.assertEqual(self._staged_files(), [])

    def test_another_user_cannot_touch_someone_elses_session(self):
        other_user = User.objects.create_user(username='other', password='pw123456')
        other_client = APIClient()
        other_client.force_authenticate(user=other_user)

        data = _tiny_jpeg_bytes(size=(60, 60))
        init_body = json.loads(self._init('photo.jpg', data).content)
        upload_id = init_body['upload_id']

        resp = self._put_chunk(upload_id, 0, self._chunks_of(data)[0], client=other_client)
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)
        resp = other_client.get(f'/api/upload/chunked/{upload_id}/status/')
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)
        resp = other_client.post(f'/api/upload/chunked/{upload_id}/complete/')
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_requires_authentication(self):
        resp = self.anon_client.post('/api/upload/chunked/init/', {
            'filename': 'photo.jpg', 'total_size': 100, 'checksum': 'a' * 64,
        }, format='json')
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))


class CleanupStaleUploadsTaskTests(ApiTestCase):
    def test_stale_session_and_its_chunk_dir_are_removed(self):
        from datetime import timedelta

        from django.utils import timezone

        from api.models import UploadChunk, UploadSession
        from api.tasks import cleanup_stale_uploads

        chunk_dir = tempfile.mkdtemp(prefix='cleanup_task_test_chunks_')
        with override_settings(UPLOAD_CHUNK_SCRATCH_DIR=chunk_dir, UPLOAD_SESSION_TTL_HOURS=1):
            session = UploadSession.objects.create(
                user=self.user, filename='old.jpg', total_size=10,
                checksum='a' * 64, chunk_size=16, total_chunks=1,
            )
            UploadChunk.objects.create(session=session, chunk_index=0, size=10)
            session_chunk_dir = os.path.join(chunk_dir, str(session.upload_id))
            os.makedirs(session_chunk_dir, exist_ok=True)
            with open(os.path.join(session_chunk_dir, '0.part'), 'wb') as f:
                f.write(b'x' * 10)

            # Backdate creation past the TTL -- auto_now_add means this
            # needs a direct .update(), a normal .save() would reset it.
            UploadSession.objects.filter(pk=session.pk).update(
                created_at=timezone.now() - timedelta(hours=2),
            )

            fresh_session = UploadSession.objects.create(
                user=self.user, filename='new.jpg', total_size=10,
                checksum='b' * 64, chunk_size=16, total_chunks=1,
            )

            cleanup_stale_uploads()

            self.assertFalse(UploadSession.objects.filter(pk=session.pk).exists())
            self.assertFalse(os.path.exists(session_chunk_dir))
            self.assertTrue(UploadSession.objects.filter(pk=fresh_session.pk).exists())
        shutil.rmtree(chunk_dir, ignore_errors=True)


