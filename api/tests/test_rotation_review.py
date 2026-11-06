"""api.rotation_review_views -- see CLAUDE.md's 2026-11-06
orientation-detection write-up.

Run with: manage.py test api.tests.test_rotation_review
"""
from unittest import mock

from django.conf import settings
from rest_framework import status

from filepopulator.models import ImageFile

from api.tests import ApiTestCase


class RotationReviewListViewTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.flagged = self.make_image()
        self.flagged.rotation_scan_state = ImageFile.RotationScanState.FLAGGED
        self.flagged.proposed_rotation_degrees = 92.5
        self.flagged.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])
        self.make_face(self.flagged)

        self.not_flagged = self.make_image(relative_fixture="naming/good/2.jpg")

    def test_anonymous_request_rejected(self):
        resp = self.anon_client.get("/api/rotation_review/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_only_flagged_images_are_listed(self):
        resp = self.client.get("/api/rotation_review/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        ids = [row['id'] for row in resp.data['results']]
        self.assertIn(self.flagged.id, ids)
        self.assertNotIn(self.not_flagged.id, ids)

    def test_response_fields_and_face_count(self):
        resp = self.client.get("/api/rotation_review/")
        row = next(r for r in resp.data['results'] if r['id'] == self.flagged.id)
        self.assertEqual(row['filename'], self.flagged.filename)
        self.assertEqual(row['proposed_rotation_degrees'], 92.5)
        self.assertEqual(row['face_count'], 1)
        self.assertIn(f"id={self.flagged.id}", row['preview_url'])
        self.assertIn(settings.RANDOM_ACCESS_KEY, row['preview_url'])

    def test_limit_and_offset_are_respected(self):
        second = self.make_image(relative_fixture="naming/good/3.JPEG")
        second.rotation_scan_state = ImageFile.RotationScanState.FLAGGED
        second.proposed_rotation_degrees = 180.0
        second.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

        resp = self.client.get("/api/rotation_review/?limit=1&offset=0")
        self.assertEqual(len(resp.data['results']), 1)
        self.assertEqual(resp.data['count'], 2)
        first_page_id = resp.data['results'][0]['id']

        resp = self.client.get("/api/rotation_review/?limit=1&offset=1")
        self.assertEqual(len(resp.data['results']), 1)
        self.assertNotEqual(resp.data['results'][0]['id'], first_page_id)


class RotationReviewBulkActionViewTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.img_a = self.make_image()
        self.img_a.rotation_scan_state = ImageFile.RotationScanState.FLAGGED
        self.img_a.proposed_rotation_degrees = 90.0
        self.img_a.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

        self.img_b = self.make_image(relative_fixture="naming/good/2.jpg")
        self.img_b.rotation_scan_state = ImageFile.RotationScanState.FLAGGED
        self.img_b.proposed_rotation_degrees = -90.0
        self.img_b.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

        # Never let a real Celery dispatch happen in a test -- every test
        # below gets this mocked automatically; dispatch behavior itself
        # is asserted explicitly in the tests that care about it.
        patcher = mock.patch('api.rotation_review_views.correct_image_rotation_ids_task.delay')
        self.mock_dispatch = patcher.start()
        self.addCleanup(patcher.stop)

    def test_confirm_sets_confirmed_state(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/", {'confirm_ids': [self.img_a.id]}, format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data, {'confirmed': 1, 'dismissed': 0})
        self.img_a.refresh_from_db()
        self.assertEqual(self.img_a.rotation_scan_state, ImageFile.RotationScanState.CONFIRMED)

    def test_dismiss_sets_dismissed_state(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/", {'dismiss_ids': [self.img_b.id]}, format='json',
        )
        self.assertEqual(resp.data, {'confirmed': 0, 'dismissed': 1})
        self.img_b.refresh_from_db()
        self.assertEqual(
            self.img_b.rotation_scan_state, ImageFile.RotationScanState.DISMISSED_NOT_ROTATED,
        )

    def test_both_in_one_call(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm_ids': [self.img_a.id], 'dismiss_ids': [self.img_b.id]},
            format='json',
        )
        self.assertEqual(resp.data, {'confirmed': 1, 'dismissed': 1})

    def test_an_already_resolved_image_is_not_touched_again(self):
        self.img_a.rotation_scan_state = ImageFile.RotationScanState.CONFIRMED
        self.img_a.save(update_fields=['rotation_scan_state'])
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/", {'confirm_ids': [self.img_a.id]}, format='json',
        )
        # Already CONFIRMED, not FLAGGED -- the scoped .filter() doesn't
        # match it, so the update count is correctly 0, not 1.
        self.assertEqual(resp.data['confirmed'], 0)

    def test_anonymous_request_rejected(self):
        resp = self.anon_client.patch(
            "/api/rotation_review/bulk_action/", {'confirm_ids': [self.img_a.id]}, format='json',
        )
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_confirm_dispatches_an_on_demand_correction_task_for_those_ids(self):
        self.client.patch(
            "/api/rotation_review/bulk_action/", {'confirm_ids': [self.img_a.id]}, format='json',
        )
        self.mock_dispatch.assert_called_once_with([self.img_a.id])

    def test_dismiss_only_does_not_dispatch_anything(self):
        self.client.patch(
            "/api/rotation_review/bulk_action/", {'dismiss_ids': [self.img_b.id]}, format='json',
        )
        self.mock_dispatch.assert_not_called()
