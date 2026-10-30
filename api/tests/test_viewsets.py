"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_viewsets
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


class ImageDirectoryViewSetTests(ApiTestCase):
    def test_list_images_includes_created_fixture(self):
        img = self.make_image()
        resp = self.client.get("/api/images/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        filenames = [r["filename"] for r in resp.data["results"]]
        self.assertIn(img.filename, filenames)

    def test_image_ordering_filter_by_filename(self):
        self.make_image("naming/good/1.JPG")
        self.make_image("naming/good/2.jpg")
        resp = self.client.get("/api/images/?filename=2.jpg")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(all("2.jpg" in r["filename"] for r in resp.data["results"]))

    def test_list_directories_reports_num_images(self):
        img = self.make_image()
        resp = self.client.get("/api/directories/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        matching = [d for d in resp.data["results"] if d["id"] == img.directory_id]
        self.assertEqual(len(matching), 1)
        self.assertGreaterEqual(matching[0]["num_images"], 1)


class PersonViewSetTests(ApiTestCase):
    def test_rename_person_success(self):
        person = Person.objects.create(person_name="Old Name")
        person.highlight_img.save("p.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        resp = self.client.put(
            f"/api/people/{person.id}/rename/", {"person_name": "New Name"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        person.refresh_from_db()
        self.assertEqual(person.person_name, "New Name")

    def test_rename_sentinel_person_rejected(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        resp = self.client.put(
            f"/api/people/{blank.id}/rename/", {"person_name": "Someone"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_rename_to_existing_name_rejected(self):
        p1 = Person.objects.create(person_name="Alice")
        p1.highlight_img.save("p1.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        p2 = Person.objects.create(person_name="Bob")
        p2.highlight_img.save("p2.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        resp = self.client.put(
            f"/api/people/{p2.id}/rename/", {"person_name": "Alice"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_toggle_further_unlikely(self):
        person = Person.objects.create(person_name="Toggled")
        person.highlight_img.save("p.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        self.assertFalse(person.further_images_unlikely)
        resp = self.client.put(f"/api/people/{person.id}/toggle_further_unlikely/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        person.refresh_from_db()
        self.assertTrue(person.further_images_unlikely)

    def test_list_and_retrieve_are_not_exposed(self):
        # PersonViewSet is deliberately a bare GenericViewSet, not a
        # ModelViewSet -- GET /api/people/ and GET /api/people/<id>/ used
        # to 500 on every request (PersonSerializer.get_num_possibilities
        # was commented out while still declared as a
        # SerializerMethodField). Fixing that crash would have exposed a
        # real, much worse landmine underneath: PersonSerializer's
        # face_declared nested-serialized every one of a person's Face
        # rows as its own hyperlinked URL -- for the blank sentinel
        # (~99k faces) or .ignore (~131k faces), that's building
        # six-figure numbers of URLs in pure Python on one request
        # (confirmed hanging 85+ seconds, 7GB+ RSS, no DB query even
        # involved). Removed rather than fixed -- nothing exercised list/
        # retrieve/create/update/destroy before this, only the two real
        # actions below.
        person = Person.objects.create(person_name="Listed Person")
        person.highlight_img.save("p.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)

        # A router only wires up a URL for an action that actually
        # exists on the viewset -- with no list/retrieve action
        # registered, these URLs don't resolve at all (plain 404), not
        # a DRF 405 on an existing-but-disallowed route.
        list_resp = self.client.get("/api/people/")
        self.assertEqual(list_resp.status_code, status.HTTP_404_NOT_FOUND)

        retrieve_resp = self.client.get(f"/api/people/{person.id}/")
        self.assertEqual(retrieve_resp.status_code, status.HTTP_404_NOT_FOUND)


class FaceViewSetTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.face = self.make_face(self.image)

    def test_assign_face_to_person(self):
        target = Person.objects.create(person_name="Target Person")
        target.highlight_img.save("t.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        resp = self.client.patch(
            f"/api/faces/{self.face.id}/assign_face_to_person/",
            {"declared_name_key": target.id},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.face.refresh_from_db()
        self.assertEqual(self.face.declared_name_id, target.id)

    def test_assign_face_to_nonexistent_person_returns_404(self):
        resp = self.client.patch(
            f"/api/faces/{self.face.id}/assign_face_to_person/",
            {"declared_name_key": 999999},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_verify_face(self):
        resp = self.client.patch(f"/api/faces/{self.face.id}/verify_face/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.face.refresh_from_db()
        self.assertTrue(self.face.validated)

    def test_ignore_face_soft_then_hard(self):
        resp = self.client.patch(
            f"/api/faces/{self.face.id}/ignore_face/", {"ignore_type": "soft"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.face.refresh_from_db()
        self.assertEqual(self.face.declared_name.person_name, ".ignore")

        resp = self.client.patch(
            f"/api/faces/{self.face.id}/ignore_face/", {"ignore_type": "hard"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.face.refresh_from_db()
        self.assertEqual(self.face.declared_name.person_name, ".realignore")

    def test_ignore_face_hard_without_prior_soft_rejected(self):
        resp = self.client.patch(
            f"/api/faces/{self.face.id}/ignore_face/", {"ignore_type": "hard"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_ignore_face_invalid_type_rejected(self):
        resp = self.client.patch(
            f"/api/faces/{self.face.id}/ignore_face/", {"ignore_type": "sideways"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_as_person_thumbnail_deletes_the_old_highlight_file(self):
        # Real incident, 2026-10-08: this endpoint reassigns
        # person.highlight_img.save(...) on an EXISTING person -- Django's
        # FieldFile never deletes the file it's replacing, so every time
        # someone picked a new representative photo for a person, the old
        # highlight file was silently orphaned. Found while root-causing a
        # real orphaned-file backlog (323 of 1238 face_highlights files,
        # ~26%, were exactly this).
        person = self.face.declared_name
        old_highlight_path = person.highlight_img.path
        self.assertTrue(os.path.isfile(old_highlight_path))

        resp = self.client.put(f"/api/faces/{self.face.id}/set_as_person_thumbnail/")

        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertFalse(os.path.isfile(old_highlight_path))
        person.refresh_from_db()
        self.assertTrue(os.path.isfile(person.highlight_img.path))
        self.assertNotEqual(person.highlight_img.path, old_highlight_path)

    def test_reject_association_model_method_still_works_directly(self):
        # Face.reject_association() is still live -- bulk_thread()'s
        # close_assigned branch calls it to decline a possible-match
        # candidate. Only the api/views.py HTTP wrapper around it
        # (reject_association_app_api) was removed as dead code; see
        # test_reject_association_app_api_endpoint_was_removed below.
        possible = Person.objects.create(person_name="Possible Match")
        possible.highlight_img.save("poss.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        self.face.poss_ident1 = possible
        self.face.weight_1 = 0.5
        self.face.save()

        self.face.reject_association(possible.id)
        self.face.refresh_from_db()
        self.assertIsNone(self.face.poss_ident1_id)
        self.assertIn(possible.id, self.face.rejected_fields)

    def test_face_to_new_person(self):
        resp = self.client.put(
            f"/api/faces/{self.face.id}/face_to_new_person/",
            {"person_name": "Brand New"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(Person.objects.filter(person_name="Brand New").exists())
        self.face.refresh_from_db()
        self.assertEqual(self.face.declared_name.person_name, "Brand New")

    def test_bulk_operation_rejects_malformed_payload(self):
        resp = self.client.patch("/api/faces/bulk_operation/", {}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_bulk_operation_rejects_unknown_person(self):
        resp = self.client.patch(
            "/api/faces/bulk_operation/",
            {
                "face_id_list": [self.face.id],
                "operation": "verify_face",
                "current_person_id": 999999,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_bulk_operation_accepts_well_formed_payload(self):
        resp = self.client.patch(
            "/api/faces/bulk_operation/",
            {
                "face_id_list": [self.face.id],
                "operation": "verify_face",
                "current_person_id": self.face.declared_name.id,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(json.loads(resp.content)["job_submitted"])

    def test_bulk_close_assigned_on_declared_face_clears_name_tag(self):
        # "Remove from person" / undo-of-confirm case: current_person_id is
        # the face's actual declared_name, not a poss_identN candidate.
        # bulk_operation only enqueues onto a background thread/queue (see
        # test_bulk_operation_accepts_well_formed_payload above, which only
        # checks the enqueue response) - call bulk_thread() directly here so
        # this test doesn't depend on that background thread's timing or
        # its own DB connection/transaction visibility.
        from api.views import bulk_thread

        current_person_id = self.face.declared_name_id
        bulk_thread({
            "face_id_list": [self.face.id],
            "operation": "close_assigned",
            "current_person_id": current_person_id,
        })
        self.face.refresh_from_db()
        self.assertEqual(self.face.declared_name.person_name, settings.BLANK_FACE_NAME)
        # Regression: "Remove from person" used to leave no record that
        # this person was tried and explicitly removed - classify_
        # unassigned() was free to immediately re-propose the exact same
        # assignment on its very next run.
        self.assertIn(current_person_id, self.face.rejected_fields)

    def test_bulk_close_assigned_on_possible_match_still_declines_it(self):
        # Declining a proposed candidate - reject_association()'s original,
        # still-correct behavior - must keep working once close_assigned
        # also handles the declared-face case above.
        from api.views import bulk_thread

        possible = Person.objects.create(person_name="Possible Match")
        possible.highlight_img.save("poss.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        self.face.poss_ident1 = possible
        self.face.weight_1 = 0.5
        self.face.save()
        original_declared_id = self.face.declared_name_id

        bulk_thread({
            "face_id_list": [self.face.id],
            "operation": "close_assigned",
            "current_person_id": possible.id,
        })
        self.face.refresh_from_db()
        self.assertIsNone(self.face.poss_ident1_id)
        self.assertIn(possible.id, self.face.rejected_fields)
        # Declared name is untouched - this declined a guess, not the
        # actual name tag.
        self.assertEqual(self.face.declared_name_id, original_declared_id)

    def test_bulk_thread_skips_unknown_face_id_without_crashing(self):
        # Regression test for a fixed bug: bulk_thread()'s
        # `except: print(...)` around the Face.objects.get() lookup had no
        # `continue`, so execution fell through to the operation branches
        # with `face` either unbound (raising UnboundLocalError on the
        # first list entry) or still holding the *previous* iteration's
        # Face (silently operating on the wrong face for later entries) --
        # either way swallowed by background_bulk_processor()'s blanket
        # except, so a stale/bad id in the list silently broke the whole
        # batch. A bad id followed by a real one should skip the bad one
        # and still process the real one correctly.
        from api.views import bulk_thread

        current_person_id = self.face.declared_name_id
        bulk_thread({
            "face_id_list": [999999, self.face.id],
            "operation": "close_assigned",
            "current_person_id": current_person_id,
        })
        self.face.refresh_from_db()
        self.assertEqual(self.face.declared_name.person_name, settings.BLANK_FACE_NAME)

    def test_reject_association_app_api_endpoint_was_removed(self):
        # reject_association_app_api() had the same unguarded-assert bug
        # as bulk_thread()'s close_assigned (asserts unassociate_id is a
        # poss_identN candidate, raising an unhandled AssertionError if
        # it's actually the declared_name instead) -- but unlike
        # close_assigned, nothing in either frontend repo this project
        # has access to (dev_facewire, facewires_frontend) actually calls
        # it or its disassociate_patch_url. Removed rather than fixed,
        # per the user's call, as dead code. This just documents that the
        # route is gone -- a future re-add attempt should route through
        # Face.reject_association()/associate_person() the same way
        # bulk_thread()'s close_assigned does, not call
        # reject_association() unconditionally.
        resp = self.client.patch(
            f"/api/faces/{self.face.id}/reject_association_app_api/",
            {"unassociate_id": self.face.declared_name_id},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class FaceDeclaredClusterGroupTests(ApiTestCase):
    # PersonParamView's face_declared branch, only_unverified=true -
    # surfaces Face.verification_cluster_group (set by the nightly
    # face_manager.cluster_unverified_faces job) so the frontend's verify-
    # screen "Group by cluster" mode can group visually similar faces for
    # bulk review instead of one at a time.
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.person = Person.objects.create(person_name="Cluster Test Person")
        self.clustered_a = self.make_face(
            self.image, declared_name=self.person, validated=False,
            verification_cluster_group=0,
        )
        self.clustered_b = self.make_face(
            self.image, declared_name=self.person, validated=False,
            verification_cluster_group=0,
        )
        self.singleton = self.make_face(
            self.image, declared_name=self.person, validated=False,
        )
        self.already_verified = self.make_face(
            self.image, declared_name=self.person, validated=True,
        )

    def test_cluster_groups_present_for_unverified_only(self):
        resp = self.client.get(f"/api/paginate_obj_ids/{self.person.id}/face_declared?only_unverified=true")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertIn(self.clustered_a.id, data["id_list"])
        self.assertIn(self.singleton.id, data["id_list"])
        self.assertNotIn(self.already_verified.id, data["id_list"])

        cluster_groups = data["cluster_groups"]
        self.assertEqual(cluster_groups[str(self.clustered_a.id)], 0)
        self.assertEqual(cluster_groups[str(self.clustered_b.id)], 0)
        # Singletons (and anything validated, since it's excluded from
        # id_list entirely) simply don't appear in the map.
        self.assertNotIn(str(self.singleton.id), cluster_groups)
        self.assertNotIn(str(self.already_verified.id), cluster_groups)

    def test_cluster_groups_empty_without_only_unverified(self):
        # Not meaningful outside the verify screen's own filtered view -
        # deliberately not computed (extra query) when the plain
        # face_declared list (validated + unvalidated together) is
        # requested instead.
        resp = self.client.get(f"/api/paginate_obj_ids/{self.person.id}/face_declared")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertEqual(data["cluster_groups"], {})


class DirectoryPaginateObjIdsTests(ApiTestCase):
    # The Folders tab (frontend) walks this id_list directly for both tile
    # order and modal prev/next paging - it was previously unordered
    # (plain DB insertion order), now newest-first by dateTaken.
    def test_directory_ids_ordered_newest_first(self):
        older = self.make_image("naming/good/1.JPG")
        newer = self.make_image("naming/good/2.jpg")
        ImageFile.objects.filter(pk=older.pk).update(dateTaken="2020-01-01T00:00:00Z")
        ImageFile.objects.filter(pk=newer.pk).update(dateTaken="2024-06-15T00:00:00Z")

        resp = self.client.get(f"/api/paginate_obj_ids/{older.directory_id}/directory")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertEqual(id_list.index(newer.id), 0)
        self.assertLess(id_list.index(newer.id), id_list.index(older.id))


class ApplicationErrorApiTests(ApiTestCase):
    """common.models.ApplicationError / common.error_log.log_error() --
    a curated, dismissible error log entries are written to server-side;
    this API only ever lists and dismisses them."""

    def test_requires_auth(self):
        resp = self.anon_client.get("/api/application_errors/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_list_defaults_to_undismissed_only(self):
        from common.error_log import log_error
        log_error('a.b', 'still open')
        log_error('a.b', 'already handled')
        from common.models import ApplicationError
        ApplicationError.objects.filter(message='already handled').update(dismissed=True)

        resp = self.client.get("/api/application_errors/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        messages = [row['message'] for row in resp.data['results']] if 'results' in resp.data else [
            row['message'] for row in resp.data]
        self.assertIn('still open', messages)
        self.assertNotIn('already handled', messages)

    def test_include_dismissed_query_param(self):
        from common.error_log import log_error
        log_error('a.b', 'already handled')
        from common.models import ApplicationError
        ApplicationError.objects.filter(message='already handled').update(dismissed=True)

        resp = self.client.get("/api/application_errors/?include_dismissed=true")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        messages = [row['message'] for row in resp.data['results']] if 'results' in resp.data else [
            row['message'] for row in resp.data]
        self.assertIn('already handled', messages)

    def test_dismiss_marks_it_dismissed(self):
        from common.error_log import log_error
        log_error('a.b', 'needs dismissing')
        from common.models import ApplicationError
        error = ApplicationError.objects.get()

        resp = self.client.post(f"/api/application_errors/{error.pk}/dismiss/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        error.refresh_from_db()
        self.assertTrue(error.dismissed)
        self.assertIsNotNone(error.dismissed_at)

    def test_dismiss_without_clear_heartbeat_leaves_heartbeat_untouched(self):
        """Plain dismiss must NEVER implicitly clear a heartbeat -- that
        would silently teach the watchdog a task finished when all that
        really happened is a human acknowledged the alert."""
        from common.error_log import log_error
        from common.heartbeat import record_heartbeat_start
        from common.models import ApplicationError, TaskHeartbeat
        record_heartbeat_start('some.task')
        log_error('common.tasks.check_task_heartbeats', 'stuck', related_heartbeat_task_name='some.task')
        error = ApplicationError.objects.get()

        resp = self.client.post(f"/api/application_errors/{error.pk}/dismiss/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        hb = TaskHeartbeat.objects.get(task_name='some.task')
        self.assertIsNone(hb.last_completed_at)

    def test_dismiss_with_clear_heartbeat_marks_the_task_done(self):
        from common.error_log import log_error
        from common.heartbeat import record_heartbeat_start
        from common.models import ApplicationError, TaskHeartbeat
        record_heartbeat_start('some.task')
        log_error('common.tasks.check_task_heartbeats', 'stuck', related_heartbeat_task_name='some.task')
        error = ApplicationError.objects.get()

        resp = self.client.post(f"/api/application_errors/{error.pk}/dismiss/", {'clear_heartbeat': True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        error.refresh_from_db()
        self.assertTrue(error.dismissed)
        hb = TaskHeartbeat.objects.get(task_name='some.task')
        self.assertIsNotNone(hb.last_completed_at)
        self.assertGreaterEqual(hb.last_completed_at, hb.last_started_at)

    def test_clear_heartbeat_is_a_no_op_on_an_unrelated_error(self):
        """An ApplicationError with no related_heartbeat_task_name (the
        vast majority of them) must silently ignore the flag, not
        error out -- the flag only ever does something for a real
        heartbeat alert."""
        from common.error_log import log_error
        from common.models import ApplicationError
        log_error('face_manager.assign_faces.execute', 'some unrelated error')
        error = ApplicationError.objects.get()
        self.assertIsNone(error.related_heartbeat_task_name)

        resp = self.client.post(f"/api/application_errors/{error.pk}/dismiss/", {'clear_heartbeat': True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        error.refresh_from_db()
        self.assertTrue(error.dismissed)


class PersonListLightweightTests(ApiTestCase):
    """GET /api/person_list/?lightweight=true - a cheap alternate response
    shape on the SAME endpoint facewires already uses (rather than a
    second person-listing endpoint), for net_slideshow's settings page
    People autocomplete: just name + face count, sorted by face count
    descending, skipping the heavy live-count/sentinel-breakdown work
    the default (unparameterized) response does for the admin sidebar."""

    def test_anonymous_cannot_use_lightweight_mode_either(self):
        resp = self.anon_client.get('/api/person_list/?lightweight=true')
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_anything_other_than_literal_true_falls_back_to_the_default_response(self):
        # Matches the rest of this file's own boolean query param
        # convention (params.get('x', '').lower() == 'true' - see
        # 'flagged'/'date'/'fast'/'highlight_box' above) rather than a
        # bare truthy check, so e.g. a stray ?lightweight=0 doesn't
        # silently switch callers expecting the full response shape.
        resp = self.client.get('/api/person_list/?lightweight=0')
        results = json.loads(resp.content)['results']
        self.assertTrue(results)
        self.assertIn('num_possibilities', results[0])

    def test_people_are_sorted_by_face_count_descending(self):
        image = self.make_image()
        popular, _ = Person.objects.get_or_create(person_name='Popular Person')
        rare, _ = Person.objects.get_or_create(person_name='Rare Person')
        self.make_face(image, declared_name=popular, box_left=1, box_top=1, box_right=10, box_bottom=10)
        self.make_face(image, declared_name=popular, box_left=11, box_top=1, box_right=20, box_bottom=10)
        self.make_face(image, declared_name=rare, box_left=21, box_top=1, box_right=30, box_bottom=10)

        resp = self.client.get('/api/person_list/?lightweight=true')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = json.loads(resp.content)['results']
        names = [r['person_name'] for r in results]
        self.assertLess(names.index('Popular Person'), names.index('Rare Person'))
        by_name = {r['person_name']: r['num_faces'] for r in results}
        self.assertEqual(by_name['Popular Person'], 2)
        self.assertEqual(by_name['Rare Person'], 1)

    def test_sentinel_names_are_excluded(self):
        resp = self.client.get('/api/person_list/?lightweight=true')
        names = [r['person_name'] for r in json.loads(resp.content)['results']]
        for sentinel in settings.IGNORED_NAMES:
            self.assertNotIn(sentinel, names)

    def test_lightweight_entries_have_no_extra_fields(self):
        self.make_image()
        resp = self.client.get('/api/person_list/?lightweight=true')
        for entry in json.loads(resp.content)['results']:
            self.assertEqual(set(entry.keys()), {'person_name', 'num_faces'})

    def test_default_response_is_unaffected_by_the_new_param(self):
        # Regression guard: adding the lightweight branch must not
        # change what facewires' existing unparameterized calls get
        # back - num_possibilities/url/id/etc. are still there.
        resp = self.client.get('/api/person_list/')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = json.loads(resp.content)['results']
        self.assertTrue(results)
        entry = results[0]
        for field in ('id', 'url', 'num_faces', 'num_possibilities', 'num_unverified_faces'):
            self.assertIn(field, entry)


