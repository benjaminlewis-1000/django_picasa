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

    def test_confirm_sets_confirmed_state_and_stores_degrees_for_the_direction(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'cw'}]}, format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data, {'confirmed': 1, 'dismissed': 0, 'rejected': []})
        self.img_a.refresh_from_db()
        self.assertEqual(self.img_a.rotation_scan_state, ImageFile.RotationScanState.CONFIRMED)
        self.assertEqual(self.img_a.proposed_rotation_degrees, 90.0)

    def test_confirm_overrides_the_audit_proposed_degrees(self):
        # img_a's audit proposal was 90.0 -- a human overriding to 180
        # (rotate-left/right buttons landing somewhere else) must take
        # precedence, with rotation_signal_source recording that a human
        # made the call, not the audit.
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': '180'}]}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 1)
        self.img_a.refresh_from_db()
        self.assertEqual(self.img_a.proposed_rotation_degrees, 180.0)
        self.assertEqual(self.img_a.rotation_signal_source, ImageFile.RotationSignalSource.HUMAN)

    def test_confirm_on_a_never_flagged_image_is_allowed(self):
        # The audit never touches an image with no detected faces at
        # all -- a human must still be able to confirm a rotation on it
        # manually.
        never_flagged = self.make_image(relative_fixture="naming/good/4.jpeg")
        self.assertEqual(never_flagged.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)

        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': never_flagged.id, 'direction': 'ccw'}]}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 1)
        never_flagged.refresh_from_db()
        self.assertEqual(never_flagged.rotation_scan_state, ImageFile.RotationScanState.CONFIRMED)
        self.assertEqual(never_flagged.proposed_rotation_degrees, -90.0)

    def test_confirm_accepts_a_common_synonym_case_insensitively(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'Clockwise'}]}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 1)
        self.img_a.refresh_from_db()
        self.assertEqual(self.img_a.proposed_rotation_degrees, 90.0)

    def test_confirm_rejects_an_unrecognized_direction(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'sideways'}]}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 0)
        self.assertEqual(len(resp.data['rejected']), 1)
        self.assertEqual(resp.data['rejected'][0]['id'], self.img_a.id)
        self.img_a.refresh_from_db()
        self.assertEqual(self.img_a.rotation_scan_state, ImageFile.RotationScanState.FLAGGED)

    def test_confirm_rejects_a_numeric_degrees_value_not_a_direction_word(self):
        # The contract is a direction word, not a signed angle -- a raw
        # number must be rejected outright, not treated as degrees.
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 90.0}]}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 0)
        self.assertEqual(len(resp.data['rejected']), 1)

    def test_confirm_rejects_an_already_corrected_image(self):
        self.img_a.rotation_correction_applied = True
        self.img_a.save(update_fields=['rotation_correction_applied'])
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'cw'}]}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 0)
        self.assertEqual(len(resp.data['rejected']), 1)

    def test_confirm_rejects_malformed_entries(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id}, {'direction': 'cw'}, 'not_even_a_dict']}, format='json',
        )
        self.assertEqual(resp.data['confirmed'], 0)
        self.assertEqual(len(resp.data['rejected']), 3)

    def test_dismiss_sets_dismissed_state_and_clears_proposed_degrees(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/", {'dismiss_ids': [self.img_b.id]}, format='json',
        )
        self.assertEqual(resp.data, {'confirmed': 0, 'dismissed': 1, 'rejected': []})
        self.img_b.refresh_from_db()
        self.assertEqual(
            self.img_b.rotation_scan_state, ImageFile.RotationScanState.DISMISSED_NOT_ROTATED,
        )
        self.assertIsNone(self.img_b.proposed_rotation_degrees)

    def test_dismiss_works_from_confirmed_state_too(self):
        # A reviewer changing their mind before the async correction job
        # has run -- confirmed but not yet rotation_correction_applied.
        self.img_b.rotation_scan_state = ImageFile.RotationScanState.CONFIRMED
        self.img_b.save(update_fields=['rotation_scan_state'])
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/", {'dismiss_ids': [self.img_b.id]}, format='json',
        )
        self.assertEqual(resp.data['dismissed'], 1)

    def test_dismiss_rejects_an_already_corrected_image(self):
        self.img_b.rotation_correction_applied = True
        self.img_b.save(update_fields=['rotation_correction_applied'])
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/", {'dismiss_ids': [self.img_b.id]}, format='json',
        )
        self.assertEqual(resp.data['dismissed'], 0)

    def test_both_in_one_call(self):
        resp = self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'cw'}], 'dismiss_ids': [self.img_b.id]},
            format='json',
        )
        self.assertEqual(resp.data['confirmed'], 1)
        self.assertEqual(resp.data['dismissed'], 1)

    def test_anonymous_request_rejected(self):
        resp = self.anon_client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'cw'}]}, format='json',
        )
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_confirm_dispatches_an_on_demand_correction_task_for_those_ids(self):
        self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'cw'}]}, format='json',
        )
        self.mock_dispatch.assert_called_once_with([self.img_a.id])

    def test_rejected_entries_are_not_dispatched(self):
        self.client.patch(
            "/api/rotation_review/bulk_action/",
            {'confirm': [{'id': self.img_a.id, 'direction': 'sideways'}]}, format='json',
        )
        self.mock_dispatch.assert_not_called()

    def test_dismiss_only_does_not_dispatch_anything(self):
        self.client.patch(
            "/api/rotation_review/bulk_action/", {'dismiss_ids': [self.img_b.id]}, format='json',
        )
        self.mock_dispatch.assert_not_called()
