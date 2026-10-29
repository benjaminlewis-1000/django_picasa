"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_media_serving
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


class KeyedImageViewTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.face = self.make_face(self.image)

    def test_unauthenticated_without_access_key_rejected(self):
        resp = self.anon_client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_unauthenticated_with_wrong_access_key_rejected(self):
        resp = self.anon_client.get(
            f"/api/keyed_image/face_array/?id={self.face.id}&access_key=wrong"
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_unauthenticated_with_correct_access_key_accepted(self):
        resp = self.anon_client.get(
            f"/api/keyed_image/face_array/?id={self.face.id}&access_key={settings.RANDOM_ACCESS_KEY}"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp["Content-Type"], "image/jpeg")

    def test_authenticated_session_does_not_need_access_key(self):
        resp = self.client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_invalid_type_rejected(self):
        resp = self.client.get(f"/api/keyed_image/not_a_type/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_missing_id_rejected(self):
        resp = self.client.get("/api/keyed_image/face_array/")
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class KeyedImageViewThrottleTests(ApiTestCase):
    """api/throttles.py -- real bug fixed 2026-10-26: these throttles
    never actually checked request.user.is_authenticated despite the
    module's own comment claiming session-authenticated requests never
    hit them. Rates are patched directly on the throttle classes'
    THROTTLE_RATES, not via override_settings(REST_FRAMEWORK=...) --
    DRF's SimpleRateThrottle binds THROTTLE_RATES to api_settings.
    DEFAULT_THROTTLE_RATES once at import time, so overriding the
    Django setting later doesn't actually propagate into it (a known
    DRF gotcha, confirmed the hard way: an override_settings version
    of this test silently used the real 30/min rate instead)."""

    def setUp(self):
        super().setUp()
        from django.core.cache import cache
        cache.clear()
        self.image = self.make_image()
        self.face = self.make_face(self.image)

    def test_authenticated_session_is_never_throttled(self):
        for _ in range(5):
            resp = self.client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
            self.assertEqual(resp.status_code, status.HTTP_200_OK)

    @mock.patch.object(AccessKeyMediaThrottle, 'THROTTLE_RATES', {'access_key_media': '2/min'})
    def test_unauthenticated_access_key_path_is_throttled(self):
        url = f"/api/keyed_image/face_array/?id={self.face.id}&access_key={settings.RANDOM_ACCESS_KEY}"
        self.assertEqual(self.anon_client.get(url).status_code, status.HTTP_200_OK)
        self.assertEqual(self.anon_client.get(url).status_code, status.HTTP_200_OK)
        resp = self.anon_client.get(url)
        self.assertEqual(resp.status_code, status.HTTP_429_TOO_MANY_REQUESTS)

    @mock.patch.object(AccessKeyMediaThrottle, 'THROTTLE_RATES', {'access_key_media': '2/min'})
    def test_unauthenticated_sig_slideshow_path_uses_the_higher_rate(self):
        # type='slideshow' is sig-gated, not access_key-gated -- it must
        # keep using slideshow_media's own (much higher) rate, never
        # AccessKeyMediaThrottle's, even though the latter is patched
        # down to 2/min here.
        from api.views import _slideshow_current_sig
        sig = _slideshow_current_sig('image', self.image.id)
        url = f"/api/keyed_image/slideshow/?id={self.image.id}&sig={sig}"
        for _ in range(5):
            resp = self.anon_client.get(url)
            self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_face_array_applies_clahe_equalization(self):
        # face_array now decodes, CLAHE-equalizes (common.clahe_equalize_bgr
        # -- brightens/evens out dark faces), and re-encodes the thumbnail
        # on every request, rather than serving the pre-generated file's
        # bytes as-is. Re-encoding a JPEG reliably changes its bytes even
        # at identical dimensions, so a byte difference here confirms the
        # equalization path actually ran rather than the old shortcut.
        self.face.face_thumbnail.open("rb")
        original_bytes = self.face.face_thumbnail.read()
        self.face.face_thumbnail.close()

        resp = self.client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp["Content-Type"], "image/jpeg")
        self.assertNotEqual(resp.content, original_bytes)

        original_img = cv2.imdecode(np.frombuffer(original_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        equalized_img = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(equalized_img)
        self.assertEqual(equalized_img.shape, original_img.shape)

    def test_keyed_image_response_is_cacheable(self):
        resp = self.client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("max-age", resp["Cache-Control"])
        self.assertIn("immutable", resp["Cache-Control"])

    @staticmethod
    def _is_reddish(pixel):
        # BICUBIC resampling during face_source's resize blurs the drawn
        # line's edges, so exact (255,0,0) only shows up a pixel or two
        # into the line - a tolerant "red clearly dominates" check instead
        # of exact equality, so this doesn't get brittle re-pinning exact
        # pixel values to whatever resample filter happens to be in use.
        r, g, b = pixel[:3]
        return r > 150 and g < 100 and b < 100

    # Matches the view's own box_line_width * 3 margin (api/views.py) -
    # the drawn line sits outside the actual detected box by this many
    # scaled pixels, not flush against it (per the user: flush against it
    # obscured the face's own edges).
    _HIGHLIGHT_BOX_MARGIN = 4 * 3

    def test_face_source_highlight_box_draws_a_box_around_the_face(self):
        # The face fixture's box is (1,1)-(40,40) in the *original* image's
        # pixel space (see make_face) - face_source resizes to a fixed
        # 700px-tall image, so the drawn box has to land at the *scaled*
        # (and then outward-margined) coordinates, not the raw ones.
        # Assert reddish pixels actually appear near the box's expanded
        # top edge (within a few rows, since resize blur softens exactly
        # which row it lands on), and that the opposite corner of the
        # image (nowhere near the box) is not - a cheap way to catch
        # "drew nothing" or "drew at the wrong scale/margin" without
        # hand-computing every pixel.
        from PIL import Image
        from io import BytesIO

        resp = self.client.get(
            f"/api/keyed_image/face_source/?id={self.face.id}&highlight_box=true"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        img = Image.open(BytesIO(resp.content)).convert("RGB")

        w, h = self.image.width, self.image.height
        scale_y = img.height / h
        scale_x = img.width / w
        top_y = max(0, round(self.face.box_top * scale_y) - self._HIGHLIGHT_BOX_MARGIN)
        left_x = max(0, round(self.face.box_left * scale_x) - self._HIGHLIGHT_BOX_MARGIN)
        right_x = round(self.face.box_right * scale_x) + self._HIGHLIGHT_BOX_MARGIN

        found_red_near_top_edge = any(
            self._is_reddish(img.getpixel((x, y)))
            for y in range(max(0, top_y - 1), min(img.height, top_y + 4))
            for x in range(max(0, left_x), min(img.width, right_x + 1))
        )
        self.assertTrue(found_red_near_top_edge, "Expected reddish pixels near the box's top edge")
        self.assertFalse(self._is_reddish(img.getpixel((img.width - 1, img.height - 1))))

    def test_face_source_without_highlight_box_param_draws_nothing(self):
        from PIL import Image
        from io import BytesIO

        resp = self.client.get(f"/api/keyed_image/face_source/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # No reddish pixels anywhere near where the box would have been.
        w, h = self.image.width, self.image.height
        scale_y = img.height / h
        scale_x = img.width / w
        top_y = max(0, round(self.face.box_top * scale_y) - self._HIGHLIGHT_BOX_MARGIN)
        left_x = max(0, round(self.face.box_left * scale_x) - self._HIGHLIGHT_BOX_MARGIN)
        right_x = round(self.face.box_right * scale_x) + self._HIGHLIGHT_BOX_MARGIN
        found_red = any(
            self._is_reddish(img.getpixel((x, y)))
            for y in range(max(0, top_y - 1), min(img.height, top_y + 4))
            for x in range(max(0, left_x), min(img.width, right_x + 1))
        )
        self.assertFalse(found_red)

    def test_face_source_date_param_returns_json_not_image_bytes(self):
        # Frontend's full-size modal fetches this alongside the image
        # itself (see gallery.jsx) to show the photo's capture date -
        # reuses face_source's own id/type resolution rather than a
        # separate endpoint, and returns before any decode/resize work.
        ImageFile.objects.filter(pk=self.image.pk).update(dateTaken="2012-12-20T08:30:00Z")
        resp = self.client.get(
            f"/api/keyed_image/face_source/?id={self.face.id}&date=true"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp["Content-Type"], "application/json")
        data = json.loads(resp.content)
        self.assertTrue(data["date_taken"].startswith("2012-12-20"))

    def test_slideshow_date_param_returns_json_not_image_bytes(self):
        # Folders tab's modal uses slideshow (keyed by ImageFile id, not
        # Face id) - same date param, different id resolution branch.
        ImageFile.objects.filter(pk=self.image.pk).update(dateTaken="2012-12-20T08:30:00Z")
        resp = self.client.get(
            f"/api/keyed_image/slideshow/?id={self.image.id}&date=true"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp["Content-Type"], "application/json")
        data = json.loads(resp.content)
        self.assertTrue(data["date_taken"].startswith("2012-12-20"))


@override_settings(MEDIA_ROOT="/tmp/api_test_media")
class ExtractVideoFaceFrameFastHdrTests(TestCase):
    """_extract_video_face_frame_fast() (api/views.py) must include the
    HDR tone-map filter for an HDR-tagged video's face, using the
    cached VideoFile.field_order/color_transfer when present (this
    "fast" path exists specifically to avoid an ffprobe call per
    request) and falling back to a live ffprobe_info() call only when
    either is missing. See CLAUDE.md's 2026-09-22 write-up -- naive HDR
    pixel-format conversion with no tone-mapping comes out visibly
    washed-out/desaturated."""

    def _make_video_face(self, field_order, color_transfer, sar_scale_width=0):
        # sar_scale_width defaults to 0 ("checked, square pixels", the
        # cached-and-clean case) rather than None ("not yet cached") --
        # tests that specifically want the ffprobe fallback triggered
        # (e.g. test_missing_cached_color_transfer_falls_back_to_ffprobe)
        # pass sar_scale_width=None explicitly.
        directory = Directory.objects.create(dir_path="/videos/hdr_fast_test")
        video = VideoFile.objects.create(
            filename="/videos/hdr_fast_test/a.mp4", directory=directory,
            width=100, height=100, duration_seconds=30, isProcessed=True,
            field_order=field_order, color_transfer=color_transfer,
            sar_scale_width=sar_scale_width,
        )
        person, created = Person.objects.get_or_create(person_name="Test Person")
        if created:
            person.highlight_img.save("p.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        face = Face(
            declared_name=person, source_video_file=video,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
            video_thumbnail_frame_seconds=1.5,
        )
        face.face_thumbnail.save(
            "face_thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False
        )
        face.save()
        return face

    def test_cached_hdr_transfer_adds_tonemap_filter_with_no_ffprobe_call(self):
        from api.views import _extract_video_face_frame_fast
        face = self._make_video_face(field_order='progressive', color_transfer='smpte2084')

        fake_result = mock.MagicMock(returncode=0, stdout=_tiny_jpeg_bytes())
        with mock.patch('api.views.subprocess.run', return_value=fake_result) as mock_run, \
             mock.patch('video_face_pipeline.ffprobe_info') as mock_ffprobe:
            _extract_video_face_frame_fast(face)

        mock_ffprobe.assert_not_called()
        cmd = mock_run.call_args[0][0]
        self.assertIn('-vf', cmd)
        vf_value = cmd[cmd.index('-vf') + 1]
        self.assertIn('tonemap', vf_value)
        self.assertNotIn('yadif', vf_value)

    def test_cached_non_hdr_transfer_has_no_tonemap_filter(self):
        from api.views import _extract_video_face_frame_fast
        face = self._make_video_face(field_order='progressive', color_transfer='bt709')

        fake_result = mock.MagicMock(returncode=0, stdout=_tiny_jpeg_bytes())
        with mock.patch('api.views.subprocess.run', return_value=fake_result) as mock_run, \
             mock.patch('video_face_pipeline.ffprobe_info') as mock_ffprobe:
            _extract_video_face_frame_fast(face)

        mock_ffprobe.assert_not_called()
        cmd = mock_run.call_args[0][0]
        self.assertNotIn('-vf', cmd)

    def test_missing_cached_color_transfer_falls_back_to_ffprobe(self):
        # A row ingested before color_transfer/sar_scale_width existed --
        # all three cached fields None here for simplicity, so the live
        # ffprobe_info() call must fill in all of them from a single
        # call, not silently skip tone-mapping.
        from api.views import _extract_video_face_frame_fast
        face = self._make_video_face(field_order=None, color_transfer=None, sar_scale_width=None)

        fake_result = mock.MagicMock(returncode=0, stdout=_tiny_jpeg_bytes())
        with mock.patch('api.views.subprocess.run', return_value=fake_result) as mock_run, \
             mock.patch(
                 'video_face_pipeline.ffprobe_info',
                 return_value=(100, 100, 30.0, 'progressive', 0, 'smpte2084', None),
             ) as mock_ffprobe:
            _extract_video_face_frame_fast(face)

        mock_ffprobe.assert_called_once()
        cmd = mock_run.call_args[0][0]
        vf_value = cmd[cmd.index('-vf') + 1]
        self.assertIn('tonemap', vf_value)


class KeyedVideoViewTests(ApiTestCase):
    """GET /api/keyed_video/<id>/ - streams bytes straight from this
    container's own filesystem mounts, never from Apache, and never
    needs the Apache credentials ParameterCredentialExposureTests above
    locks down."""

    @staticmethod
    def _sig(video_id):
        # Lazy import, not at module scope - see the module-level comment
        # at the top of this file on why api.views must never be imported
        # before ApiTestCase.setUpClass has created the sentinel Person
        # rows it needs at import time. By the time any test method body
        # runs, that's already happened (and api.views has, in practice,
        # already been imported anyway via URL resolution on the first
        # request).
        from api.views import _slideshow_current_sig
        return _slideshow_current_sig('video', video_id)

    def _make_transcoded_video(self):
        directory = Directory.objects.create(dir_path='/videos/keyed_video_test')
        video = VideoFile.objects.create(
            filename='/videos/keyed_video_test/original.mov', directory=directory,
            width=640, height=480, duration_seconds=5, codec='hevc',
            transcoded_path='ab/c/abc123.mp4',
        )
        out_path = os.path.join(settings.VIDEO_TRANSCODE_ROOT, 'ab/c/abc123.mp4')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(b'fake transcoded mp4 bytes')
        self.addCleanup(lambda: os.path.exists(out_path) and os.remove(out_path))
        return video

    def _make_native_video(self, tmp_path):
        directory = Directory.objects.create(dir_path='/videos/keyed_video_test_native')
        with open(tmp_path, 'wb') as f:
            f.write(b'fake native-codec mp4 bytes')
        return VideoFile.objects.create(
            filename=tmp_path, directory=directory,
            width=640, height=480, duration_seconds=5, codec='h264',
        )

    def test_requires_access_key_or_session(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(f'/api/keyed_video/{video.id}/')
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_wrong_access_key_rejected(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(f'/api/keyed_video/{video.id}/?sig=not-the-sig')
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_streams_transcoded_video_bytes_with_correct_access_key(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}'
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(b''.join(resp.streaming_content), b'fake transcoded mp4 bytes')
        self.assertEqual(resp['Content-Type'], 'video/mp4')
        self.assertEqual(resp['Accept-Ranges'], 'bytes')

    def test_range_request_returns_206_with_only_the_requested_bytes(self):
        # Regression test: Django's FileResponse (and even
        # django.views.static.serve()) has NO built-in Range support -
        # the first version of this view silently ignored the Range
        # header and always returned the whole file with a plain 200,
        # which defeats both scrubbing and Slideshow.jsx's next-video
        # preload.
        video = self._make_transcoded_video()  # body: b'fake transcoded mp4 bytes' (25 bytes)
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}',
            HTTP_RANGE='bytes=5-12',
        )
        self.assertEqual(resp.status_code, status.HTTP_206_PARTIAL_CONTENT)
        self.assertEqual(b''.join(resp.streaming_content), b'transcod')
        self.assertEqual(resp['Content-Range'], 'bytes 5-12/25')
        self.assertEqual(resp['Content-Length'], '8')

    def test_range_request_with_open_ended_end_goes_to_eof(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}',
            HTTP_RANGE='bytes=20-',
        )
        self.assertEqual(resp.status_code, status.HTTP_206_PARTIAL_CONTENT)
        self.assertEqual(b''.join(resp.streaming_content), b'bytes')
        self.assertEqual(resp['Content-Range'], 'bytes 20-24/25')

    def test_suffix_range_request_returns_last_n_bytes(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}',
            HTTP_RANGE='bytes=-5',
        )
        self.assertEqual(resp.status_code, status.HTTP_206_PARTIAL_CONTENT)
        self.assertEqual(b''.join(resp.streaming_content), b'bytes')
        self.assertEqual(resp['Content-Range'], 'bytes 20-24/25')

    def test_range_entirely_past_eof_returns_416(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}',
            HTTP_RANGE='bytes=1000-2000',
        )
        self.assertEqual(resp.status_code, status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
        self.assertEqual(resp['Content-Range'], 'bytes */25')

    def test_malformed_range_header_falls_back_to_full_200_response(self):
        video = self._make_transcoded_video()
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}',
            HTTP_RANGE='not-a-range-header',
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(b''.join(resp.streaming_content), b'fake transcoded mp4 bytes')

    def test_streams_native_codec_video_from_its_original_path(self):
        tmp_path = tempfile.mktemp(suffix='.mp4')
        video = self._make_native_video(tmp_path)
        self.addCleanup(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}'
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(b''.join(resp.streaming_content), b'fake native-codec mp4 bytes')

    def test_authenticated_session_bypasses_access_key_requirement(self):
        video = self._make_transcoded_video()
        resp = self.client.get(f'/api/keyed_video/{video.id}/')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_not_yet_transcoded_non_native_video_returns_404(self):
        directory = Directory.objects.create(dir_path='/videos/keyed_video_test_pending')
        video = VideoFile.objects.create(
            filename='/videos/keyed_video_test_pending/a.mov', directory=directory,
            width=640, height=480, duration_seconds=5, codec='hevc',
        )
        resp = self.anon_client.get(
            f'/api/keyed_video/{video.id}/?sig={self._sig(video.id)}'
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_unknown_video_id_returns_404(self):
        resp = self.anon_client.get(
            f'/api/keyed_video/999999/?sig={self._sig(999999)}'
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class VideoDataNeverExposesPlaybackCredentialsTests(ApiTestCase):
    """filteredImagesView's video_data must never include a direct
    Apache playback_url (see KeyedVideoViewTests/
    ParameterCredentialExposureTests for the full reasoning) - just
    enough metadata for the frontend to build its own
    /api/keyed_video/<id>/ URL."""

    def test_video_data_has_no_playback_url_field(self):
        directory = Directory.objects.create(dir_path='/videos/video_data_test')
        video = VideoFile.objects.create(
            filename='/videos/video_data_test/a.mp4', directory=directory,
            width=640, height=480, duration_seconds=12, codec='h264',
            # get_playback_url() returns THIS field for a native-codec,
            # not-yet-transcoded video (not just a bool check on codec) -
            # see its own docstring. Set here so 'ready' actually comes
            # back true, matching a real ingested native-codec video.
            playback_path='https://media.example.com/full_res/a.mp4',
        )
        resp = self.anon_client.get(
            f'/api/image_list/?full_data=true&key={settings.SLIDESHOW_API_KEY}'
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        entry = resp.json()['video_data'][str(video.id)]
        self.assertNotIn('playback_url', entry)
        self.assertTrue(entry['ready'])
        self.assertIn('sig', entry)


class VideoFaceExtractionApiTests(ApiTestCase):
    """api.views.VideoFaceExtractionViewSet -- on-demand, frontend-
    triggerable retry of videos that FAILED face extraction. List
    surfaces the current failed-video backlog (count + rows); create
    queues a retry, scoped to currently-failed videos only -- the user
    explicitly does not want a way to re-trigger an already-succeeded
    video. Always dispatches via Celery (mocked here, never actually
    run), never synchronously -- these tests only check that the right
    task call is queued with the right arguments, not real extraction
    behavior (covered separately in face_manager.tests for
    process_video_faces(video_ids=...) itself)."""

    def _make_video(self, filename='/videos/trigger_test/a.mp4', **overrides):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        return VideoFile.objects.create(
            filename=filename, directory=directory, width=640, height=480, duration_seconds=5,
            **overrides,
        )

    def _make_failed_video(self, filename='/videos/trigger_test/failed.mp4'):
        return self._make_video(
            filename, isProcessed=True, face_extraction_failed=True,
            face_extraction_error='boom',
        )

    def test_requires_auth(self):
        resp = self.anon_client.post("/api/video_face_extraction/", {}, format='json')
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_list_requires_auth(self):
        resp = self.anon_client.get("/api/video_face_extraction/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_list_returns_only_failed_videos(self):
        failed = self._make_failed_video()
        self._make_video('/videos/trigger_test/ok.mp4', isProcessed=True)  # succeeded, not listed
        self._make_video('/videos/trigger_test/new.mp4', isProcessed=False)  # never processed, not listed

        resp = self.client.get("/api/video_face_extraction/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['count'], 1)
        self.assertEqual(resp.data['results'][0]['id'], failed.id)
        self.assertEqual(resp.data['results'][0]['face_extraction_error'], 'boom')

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_trigger_with_no_body_queues_all_currently_failed_videos(self, mock_delay):
        failed1 = self._make_failed_video('/videos/trigger_test/f1.mp4')
        failed2 = self._make_failed_video('/videos/trigger_test/f2.mp4')
        self._make_video('/videos/trigger_test/ok.mp4', isProcessed=True)

        resp = self.client.post("/api/video_face_extraction/", {}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
        called_ids = mock_delay.call_args.kwargs['video_ids']
        self.assertEqual(set(called_ids), {failed1.id, failed2.id})

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_trigger_with_no_body_and_nothing_failed_is_a_no_op(self, mock_delay):
        self._make_video('/videos/trigger_test/ok.mp4', isProcessed=True)
        resp = self.client.post("/api/video_face_extraction/", {}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        mock_delay.assert_not_called()

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_trigger_with_specific_failed_video_id(self, mock_delay):
        failed = self._make_failed_video()
        resp = self.client.post(
            "/api/video_face_extraction/", {'video_ids': [failed.id]}, format='json'
        )
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
        mock_delay.assert_called_once_with(max_runtime_seconds=None, video_ids=[failed.id])

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_trigger_with_max_runtime_seconds(self, mock_delay):
        failed = self._make_failed_video()
        resp = self.client.post(
            "/api/video_face_extraction/",
            {'video_ids': [failed.id], 'max_runtime_seconds': 600}, format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
        mock_delay.assert_called_once_with(max_runtime_seconds=600, video_ids=[failed.id])

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_already_succeeded_video_id_is_rejected(self, mock_delay):
        succeeded = self._make_video('/videos/trigger_test/ok.mp4', isProcessed=True)
        resp = self.client.post(
            "/api/video_face_extraction/", {'video_ids': [succeeded.id]}, format='json'
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        mock_delay.assert_not_called()

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_unknown_video_id_is_rejected(self, mock_delay):
        resp = self.client.post(
            "/api/video_face_extraction/", {'video_ids': [999999]}, format='json'
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        mock_delay.assert_not_called()

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_empty_video_ids_list_is_rejected(self, mock_delay):
        resp = self.client.post(
            "/api/video_face_extraction/", {'video_ids': []}, format='json'
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        mock_delay.assert_not_called()

    @mock.patch('face_manager.tasks.process_video_faces.delay')
    def test_non_integer_video_ids_rejected(self, mock_delay):
        resp = self.client.post(
            "/api/video_face_extraction/", {'video_ids': ['abc']}, format='json'
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        mock_delay.assert_not_called()


