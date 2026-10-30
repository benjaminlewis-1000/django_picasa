import os
import zlib
from datetime import timedelta

import cv2
import numpy as np
import psycopg2
from django.conf import settings
from django.db import connection
from django.test import TestCase
from django.utils import timezone
from PIL import Image

from common.advisory_lock import advisory_lock
from common.open_img_oriented import open_img_oriented, apply_exif_orientation
from common.equalize import clahe_equalize_bgr


class OpenImgOrientedTests(TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            open_img_oriented("/tmp/does_not_exist_at_all.jpg", as_numpy=True)

    def test_does_not_leak_file_descriptors_across_many_calls(self):
        # Regression test for a real production incident (2026-09-03):
        # PIL keeps the underlying file open until Image.load() actually
        # runs, and neither _getexif() nor convert()/transpose() were
        # guaranteed to trigger that for the ORIGINALLY opened image
        # object -- a tight loop over many images (FaceExtractor.
        # reencode_missing_faces()) leaked one fd per call, hit the
        # process's ulimit -n (1024) partway through a 1,219-face batch,
        # and made 206 perfectly good images fail with a misleading
        # decode error. Calling this function many more times than a
        # typical ulimit allows must not accumulate open fds.
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        fd_dir = f"/proc/{os.getpid()}/fd"
        before = len(os.listdir(fd_dir))
        for _ in range(200):
            open_img_oriented(path, as_numpy=True)
        after = len(os.listdir(fd_dir))
        self.assertLessEqual(after - before, 5, f"fd count grew from {before} to {after}")

    def test_reads_normal_jpeg_as_numpy(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        arr = open_img_oriented(path, as_numpy=True)
        self.assertEqual(len(arr.shape), 3)

    def test_reads_normal_jpeg_as_pil_image(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        img = open_img_oriented(path, as_numpy=False)
        self.assertEqual(img.mode, "RGB")

    def test_corrupted_image_returns_none(self):
        # Regression test for a fixed bug: open_img_oriented()'s try/except
        # used to only wrap the initial PIL.Image.open() call, which
        # succeeds even for a truncated/broken JPEG (PIL parses the header
        # lazily and doesn't decode pixels yet). The actual decode error
        # only surfaced later, at `np.array(image)` (as_numpy=True) --
        # outside the old guard -- so despite *looking* like it handled bad
        # images gracefully, corrupted files actually raised an unguarded
        # OSError. Now that call is wrapped too, so this genuinely returns
        # None on failure as documented.
        #
        # Picks whatever's actually present in the fixture directory
        # rather than a hardcoded filename -- the real local fixture set
        # (5 real corrupted JPEGs pulled from production logs) and CI's
        # own synthetic ci_fixtures/corrupted/ don't share filenames, and
        # this test previously hardcoded a name ("truncated_a.jpg") that
        # matched neither, so it always failed with FileNotFoundError.
        corrupted_dir = "/photos/corrupted"
        candidates = sorted(os.listdir(corrupted_dir))
        self.assertTrue(candidates, f"No fixture files found in {corrupted_dir}")
        path = os.path.join(corrupted_dir, candidates[0])
        self.assertIsNone(open_img_oriented(path, as_numpy=True))


def _diagonal_marker_image(width=200, height=150, color=(120, 60, 200)):
    # Same construction as ci_fixtures/generate_fixtures.py's
    # _solid_jpeg_bytes(): a solid color block with a white diagonal
    # stripe breaking symmetry, so a transform's effect on geometry is
    # directly checkable by locating the white pixels afterward.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    for i in range(min(width, height)):
        img[i, i] = (255, 255, 255)
    return Image.fromarray(img)


class ApplyExifOrientationTests(TestCase):
    """Regression test for a fixed bug: this module used to only handle
    EXIF orientations 3, 6, 8 (via rotate()), silently doing nothing for
    2, 4, 5, 7 -- while filepopulator/models.py had its own separate,
    complete 8-value implementation (via transpose()). apply_exif_
    orientation() is now that single shared implementation, used by both.

    Expected (width, height) and first/last white-pixel (y, x) coordinates
    below were captured directly from this function's own output on a
    known 200x150 source image, not re-derived by hand -- they pin down
    today's verified-correct behavior (matching filepopulator's original,
    already-relied-upon transform) as a regression baseline."""

    def setUp(self):
        self.image = _diagonal_marker_image()

    def _white_pixel_range(self, out_image):
        arr = np.array(out_image)
        coords = np.argwhere((arr == 255).all(axis=2))
        return coords[0].tolist(), coords[-1].tolist()

    def test_orientation_0_is_a_no_op_same_as_1(self):
        # orientation 0 isn't a standard EXIF value, but is observed in
        # ~1090 images in the live library -- explicitly treated the same
        # as 1 (no rotation), not left to fall through unhandled.
        out0 = apply_exif_orientation(self.image, 0)
        out1 = apply_exif_orientation(self.image, 1)
        self.assertEqual(out0.size, self.image.size)
        self.assertEqual(np.array(out0).tolist(), np.array(out1).tolist())
        self.assertEqual(self._white_pixel_range(out0), ([0, 0], [149, 149]))

    def test_orientation_2_mirrors_left_right_without_resizing(self):
        # Previously unhandled entirely (silent no-op) -- now a real flip.
        out = apply_exif_orientation(self.image, 2)
        self.assertEqual(out.size, self.image.size)
        self.assertEqual(self._white_pixel_range(out), ([0, 199], [149, 50]))

    def test_orientation_3_rotates_180_without_resizing(self):
        out = apply_exif_orientation(self.image, 3)
        self.assertEqual(out.size, self.image.size)
        self.assertEqual(self._white_pixel_range(out), ([0, 50], [149, 199]))

    def test_orientation_4_mirrors_top_bottom_without_resizing(self):
        # Previously unhandled entirely (silent no-op) -- now a real flip.
        out = apply_exif_orientation(self.image, 4)
        self.assertEqual(out.size, self.image.size)
        self.assertEqual(self._white_pixel_range(out), ([0, 149], [149, 0]))

    def test_orientation_5_flips_and_rotates_swapping_dimensions(self):
        # Previously unhandled entirely (silent no-op).
        out = apply_exif_orientation(self.image, 5)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([0, 0], [149, 149]))

    def test_orientation_6_rotates_swapping_dimensions(self):
        out = apply_exif_orientation(self.image, 6)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([0, 149], [149, 0]))

    def test_orientation_7_flips_and_rotates_swapping_dimensions(self):
        # Previously unhandled entirely (silent no-op).
        out = apply_exif_orientation(self.image, 7)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([50, 0], [199, 149]))

    def test_orientation_8_rotates_swapping_dimensions(self):
        out = apply_exif_orientation(self.image, 8)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([50, 149], [199, 0]))

    def test_unrecognized_orientation_value_is_left_untouched(self):
        out = apply_exif_orientation(self.image, 99)
        self.assertEqual(np.array(out).tolist(), np.array(self.image).tolist())


class OpenImgOrientedRealOrientationFixturesTests(TestCase):
    """End-to-end check that open_img_oriented() actually reads the EXIF
    Orientation tag and applies apply_exif_orientation() with it, using
    the real (JPEG-compressed) per-orientation fixtures. Only asserts on
    output dimensions here (whether the transform swapped width/height),
    not exact pixel positions -- a single-pixel-wide diagonal stripe
    against a solid background doesn't reliably survive JPEG's lossy
    compression, so pixel-perfect geometry is covered instead by
    ApplyExifOrientationTests against an uncompressed array."""

    ORIENTATION_DIR = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/orientation"

    def test_dimensions_swap_only_for_the_four_90_degree_orientations(self):
        for orientation in range(1, 9):
            path = f"{self.ORIENTATION_DIR}/2_{orientation}.jpg"
            with self.subTest(orientation=orientation):
                img = open_img_oriented(path, as_numpy=False)
                if orientation in (5, 6, 7, 8):
                    self.assertGreater(img.size[1], img.size[0])
                else:
                    self.assertGreater(img.size[0], img.size[1])


class ClaheEqualizeTests(TestCase):
    """clahe_equalize_bgr() -- used to brighten/even out dark or
    unevenly-lit face thumbnails (see api/views.py's face_array branch),
    found via a real user complaint about dark faces in the frontend."""

    def test_output_has_same_shape_and_dtype_as_input(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = clahe_equalize_bgr(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)

    def test_uniformly_dark_image_is_brightened(self):
        # A flat dark gray square -- no local contrast for CLAHE to work
        # with, but the L channel's overall level should still come up
        # noticeably, since CLAHE's histogram-equalization step operates
        # on absolute intensity, not just local contrast.
        dark = np.full((100, 100, 3), 30, dtype=np.uint8)
        out = clahe_equalize_bgr(dark)
        self.assertGreater(out.mean(), dark.mean())

    def test_half_dark_half_bright_image_evens_out_the_dark_half(self):
        # A real "unevenly lit face" case: one half of the image much
        # darker than the other. CLAHE (tile-adaptive) should raise the
        # dark half's brightness substantially more than a global
        # technique would, since each tile equalizes against its own
        # local neighborhood.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :50] = 20   # dark half
        img[:, 50:] = 200  # bright half

        out = clahe_equalize_bgr(img)
        dark_half_before = img[:, :50].mean()
        dark_half_after = out[:, :50].mean()
        self.assertGreater(dark_half_after, dark_half_before)

    def test_pure_black_image_does_not_crash(self):
        black = np.zeros((50, 50, 3), dtype=np.uint8)
        out = clahe_equalize_bgr(black)
        self.assertEqual(out.shape, black.shape)

    def test_preserves_color_better_than_naive_per_channel_equalization(self):
        # Equalizing L in LAB (what clahe_equalize_bgr does) shouldn't
        # introduce the color shift that equalizing B/G/R independently
        # would -- a rough check: a gray (equal BGR) pixel region should
        # stay close to gray after equalization, not drift toward a
        # particular hue.
        gray_dark = np.full((100, 100, 3), 40, dtype=np.uint8)
        out = clahe_equalize_bgr(gray_dark)
        b, g, r = out[..., 0].mean(), out[..., 1].mean(), out[..., 2].mean()
        self.assertLess(max(b, g, r) - min(b, g, r), 5)

    def test_real_dark_face_thumbnail_gets_brighter(self):
        # Uses a real fixture image rather than only synthetic arrays,
        # so this fails if the LAB round-trip or CLAHE parameters ever
        # stop doing anything meaningful on real JPEG-compressed photos.
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        img = cv2.imread(path)
        self.assertIsNotNone(img)
        # Darken it artificially to simulate a dark face crop, since the
        # fixture itself isn't necessarily dark.
        dark_img = (img.astype(np.float32) * 0.25).astype(np.uint8)
        out = clahe_equalize_bgr(dark_img)
        self.assertGreater(out.mean(), dark_img.mean())


class AdvisoryLockTests(TestCase):
    """common.advisory_lock -- a Postgres advisory-lock mutex built to
    close a real race in find_and_encode_faces() (see CLAUDE.md): two
    concurrent invocations of that function, from any entry point, could
    each see an image as unprocessed and both insert duplicate Face rows
    for the same detected face. Unlike a lockfile, the lock is tied to
    the DB session, so a crash can't leave it stuck.

    django.db.connection's own settings_dict is used (rather than
    hardcoding a database name) to open a genuinely separate raw
    connection against whatever DB Django's test runner is actually
    using, so these tests exercise real cross-session contention."""

    def _second_connection(self):
        cfg = connection.settings_dict
        conn = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        conn.autocommit = True
        self.addCleanup(conn.close)
        return conn

    def test_acquires_and_releases_an_uncontended_lock(self):
        with advisory_lock('common.tests.uncontended') as acquired:
            self.assertTrue(acquired)

    def test_lock_is_free_again_after_the_with_block_exits(self):
        with advisory_lock('common.tests.reusable'):
            pass
        with advisory_lock('common.tests.reusable') as acquired:
            self.assertTrue(acquired)

    def test_lock_is_released_even_if_the_block_raises(self):
        with self.assertRaises(ValueError):
            with advisory_lock('common.tests.raises'):
                raise ValueError("boom")
        with advisory_lock('common.tests.raises') as acquired:
            self.assertTrue(acquired)

    def test_second_session_cannot_acquire_a_held_lock(self):
        other = self._second_connection()
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", [
                zlib.crc32(b'common.tests.contended')
            ])
            self.assertTrue(cur.fetchone()[0])

        with advisory_lock('common.tests.contended') as acquired:
            self.assertFalse(acquired)

    def test_releasing_the_other_session_frees_it_for_this_one(self):
        other = self._second_connection()
        key_name = 'common.tests.contended_then_freed'
        key = zlib.crc32(key_name.encode())
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
            self.assertTrue(cur.fetchone()[0])
            cur.execute("SELECT pg_advisory_unlock(%s)", [key])

        with advisory_lock(key_name) as acquired:
            self.assertTrue(acquired)

    def test_different_lock_names_do_not_contend_with_each_other(self):
        with advisory_lock('common.tests.name_a') as acquired_a:
            with advisory_lock('common.tests.name_b') as acquired_b:
                self.assertTrue(acquired_a)
                self.assertTrue(acquired_b)


class ApplicationErrorLogTests(TestCase):
    def test_log_error_creates_an_undismissed_row(self):
        from common.error_log import log_error
        from common.models import ApplicationError

        log_error('some.module.function', 'Something worth a look.')

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'some.module.function')
        self.assertEqual(error.message, 'Something worth a look.')
        self.assertFalse(error.dismissed)
        self.assertIsNone(error.dismissed_at)
        self.assertIsNotNone(error.created_at)

    def test_newest_first_ordering(self):
        from common.error_log import log_error
        from common.models import ApplicationError

        log_error('a', 'first')
        log_error('a', 'second')

        messages_in_order = list(ApplicationError.objects.values_list('message', flat=True))
        self.assertEqual(messages_in_order, ['second', 'first'])

    def test_message_under_the_cap_is_not_truncated(self):
        from common.error_log import log_error, MAX_MESSAGE_LENGTH
        from common.models import ApplicationError

        message = 'x' * (MAX_MESSAGE_LENGTH - 1)
        log_error('a', message)

        self.assertEqual(ApplicationError.objects.get().message, message)

    def test_message_over_the_cap_is_truncated_centrally(self):
        # The real backstop: even a call site that forgot to bound its
        # own message (e.g. a huge, un-truncated id list or traceback
        # dump) can't flood a single row -- enforced here regardless of
        # what the caller passed in.
        from common.error_log import log_error, MAX_MESSAGE_LENGTH
        from common.models import ApplicationError

        huge_message = 'x' * (MAX_MESSAGE_LENGTH * 10)
        log_error('a', huge_message)

        stored = ApplicationError.objects.get().message
        self.assertLess(len(stored), len(huge_message))
        self.assertTrue(stored.startswith('x' * 100))
        self.assertIn(str(len(huge_message)), stored)
        self.assertIn('truncated', stored)


class TaskHeartbeatTests(TestCase):
    def test_start_creates_a_row(self):
        from common.heartbeat import record_heartbeat_start
        from common.models import TaskHeartbeat

        record_heartbeat_start('some.task')

        heartbeat = TaskHeartbeat.objects.get(task_name='some.task')
        self.assertIsNotNone(heartbeat.last_started_at)
        self.assertIsNone(heartbeat.last_completed_at)

    def test_start_then_done_updates_both_fields(self):
        from common.heartbeat import record_heartbeat_start, record_heartbeat_done
        from common.models import TaskHeartbeat

        record_heartbeat_start('some.task')
        record_heartbeat_done('some.task')

        heartbeat = TaskHeartbeat.objects.get(task_name='some.task')
        self.assertIsNotNone(heartbeat.last_completed_at)
        self.assertGreaterEqual(heartbeat.last_completed_at, heartbeat.last_started_at)

    def test_second_start_overwrites_the_same_row_not_a_duplicate(self):
        from common.heartbeat import record_heartbeat_start
        from common.models import TaskHeartbeat

        record_heartbeat_start('some.task')
        record_heartbeat_start('some.task')

        self.assertEqual(TaskHeartbeat.objects.filter(task_name='some.task').count(), 1)

    def test_done_without_a_prior_start_is_a_harmless_no_op(self):
        from common.heartbeat import record_heartbeat_done
        from common.models import TaskHeartbeat

        record_heartbeat_done('never_started.task')  # must not raise

        self.assertFalse(TaskHeartbeat.objects.filter(task_name='never_started.task').exists())


class CheckTaskHeartbeatsTaskTests(TestCase):
    """common.check_task_heartbeats -- watches common.tasks.
    HEARTBEAT_REGISTRY for two independent failure modes a try/except
    can never catch on its own: a task that hasn't been invoked
    recently enough, and a task that started but never finished (stuck/
    hung). Exercises the registry's own real entry directly rather than
    a fake one, since the watchdog task hardcodes HEARTBEAT_REGISTRY
    (not dependency-injected) -- this is the one real registered task
    name, 'face_manager.assign_faces.weekly_full'."""

    TASK_NAME = 'face_manager.assign_faces.weekly_full'

    def test_never_recorded_is_flagged(self):
        from common.models import ApplicationError
        from common.tasks import check_task_heartbeats

        check_task_heartbeats()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'common.tasks.check_task_heartbeats')
        self.assertIn(self.TASK_NAME, error.message)
        self.assertIn('never recorded', error.message)
        # 2026-10-30: a structured link back to the specific task, not
        # just free text -- see ApplicationError.related_heartbeat_
        # task_name's own docstring, lets dismiss() offer a "clear it"
        # action without parsing the message.
        self.assertEqual(error.related_heartbeat_task_name, self.TASK_NAME)

    def test_recently_started_and_completed_is_healthy(self):
        from common.models import ApplicationError, TaskHeartbeat
        from common.tasks import check_task_heartbeats

        now = timezone.now()
        TaskHeartbeat.objects.create(
            task_name=self.TASK_NAME, last_started_at=now, last_completed_at=now,
        )

        check_task_heartbeats()

        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_stale_start_is_flagged(self):
        from common.models import ApplicationError, TaskHeartbeat
        from common.tasks import check_task_heartbeats

        long_ago = timezone.now() - timedelta(days=30)
        TaskHeartbeat.objects.create(
            task_name=self.TASK_NAME, last_started_at=long_ago, last_completed_at=long_ago,
        )

        check_task_heartbeats()

        error = ApplicationError.objects.get()
        self.assertIn("hasn't started a new run", error.message)

    def test_started_but_never_completed_within_budget_is_flagged_as_stuck(self):
        from common.models import ApplicationError, TaskHeartbeat
        from common.tasks import check_task_heartbeats

        # Recent enough that "hasn't started recently" wouldn't fire on
        # its own, but well past the 24h runtime budget with no
        # completion recorded at all -- the stuck/hung case.
        started = timezone.now() - timedelta(hours=25)
        TaskHeartbeat.objects.create(
            task_name=self.TASK_NAME, last_started_at=started, last_completed_at=None,
        )

        check_task_heartbeats()

        error = ApplicationError.objects.get()
        self.assertIn('possibly stuck', error.message)

    def test_in_progress_but_still_within_runtime_budget_is_not_flagged(self):
        from common.models import ApplicationError, TaskHeartbeat
        from common.tasks import check_task_heartbeats

        started = timezone.now() - timedelta(hours=2)
        TaskHeartbeat.objects.create(
            task_name=self.TASK_NAME, last_started_at=started, last_completed_at=None,
        )

        check_task_heartbeats()

        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_stale_previous_completion_does_not_mask_a_stuck_current_run(self):
        # last_completed_at from a PREVIOUS run, older than the current
        # last_started_at, must not be read as "this run finished."
        from common.models import ApplicationError, TaskHeartbeat
        from common.tasks import check_task_heartbeats

        old_completion = timezone.now() - timedelta(days=7)
        new_start = timezone.now() - timedelta(hours=25)
        TaskHeartbeat.objects.create(
            task_name=self.TASK_NAME, last_started_at=new_start, last_completed_at=old_completion,
        )

        check_task_heartbeats()

        error = ApplicationError.objects.get()
        self.assertIn('possibly stuck', error.message)
