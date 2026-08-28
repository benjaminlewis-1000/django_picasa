import json
import unittest
from io import BytesIO

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from django.utils.functional import SimpleLazyObject
from rest_framework import status
from rest_framework.test import APIClient

from face_manager.models import Face, Person
from filepopulator.models import ImageFile
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


@override_settings(MEDIA_ROOT="/tmp/api_test_media")
class ApiTestCase(TestCase):
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


class LazySentinelPersonTests(ApiTestCase):
    """Regression test for a fixed bug: api/views.py used to run
    Person.objects.filter(person_name='.ignore')[0] etc as plain
    module-level queries -- executed at import time (URL resolution,
    Django's system checks), before any test's setUp/sentinel seeding has
    run. That crashed the entire app on any DB without those rows
    already present, which is every fresh install and every CI run (only
    production avoided it, because someone seeded them by hand once).
    Now they're SimpleLazyObject-wrapped so the query is deferred to
    first actual use."""

    def test_sentinel_people_are_lazy_not_eagerly_queried(self):
        import api.views as api_views
        self.assertIsInstance(api_views.soft_ignore_person, SimpleLazyObject)
        self.assertIsInstance(api_views.hard_ignore_person, SimpleLazyObject)
        self.assertIsInstance(api_views.blank_person, SimpleLazyObject)

    def test_sentinel_people_resolve_correctly_on_first_access(self):
        import api.views as api_views
        ignore = Person.objects.get(person_name=".ignore")
        realignore = Person.objects.get(person_name=".realignore")
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.assertEqual(api_views.soft_ignore_person.id, ignore.id)
        self.assertEqual(api_views.hard_ignore_person.id, realignore.id)
        self.assertEqual(api_views.blank_person.id, blank.id)


class AuthenticationTests(ApiTestCase):
    def test_anonymous_request_to_protected_viewset_is_rejected(self):
        resp = self.anon_client.get("/api/images/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_authenticated_request_to_protected_viewset_succeeds(self):
        resp = self.client.get("/api/images/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_token_obtain_with_valid_credentials(self):
        resp = self.anon_client.post(
            "/api/token/obtain/", {"username": "tester", "password": "pw123456"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("access", resp.data)
        self.assertIn("refresh", resp.data)

    def test_token_obtain_with_bad_credentials_rejected(self):
        resp = self.anon_client.post(
            "/api/token/obtain/", {"username": "tester", "password": "wrong"}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_jwt_access_token_authenticates_requests(self):
        token_resp = self.anon_client.post(
            "/api/token/obtain/", {"username": "tester", "password": "pw123456"}, format="json"
        )
        access = token_resp.data["access"]
        self.anon_client.credentials(HTTP_AUTHORIZATION=f"JWT {access}")
        resp = self.anon_client.get("/api/images/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


class SlideshowKeyPermissionTests(ApiTestCase):
    """Covers api/permissions.py HasSlideshowKeyOrAuthenticated, used by
    filteredImagesView and ParameterViewSet."""

    def test_no_key_and_unauthenticated_is_rejected(self):
        resp = self.anon_client.get("/api/image_list/")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_wrong_key_header_is_rejected(self):
        resp = self.anon_client.get("/api/image_list/", HTTP_X_SLIDESHOW_KEY="not-the-key")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_correct_key_header_is_accepted(self):
        resp = self.anon_client.get(
            "/api/image_list/", HTTP_X_SLIDESHOW_KEY=settings.SLIDESHOW_API_KEY
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_correct_key_query_param_is_accepted(self):
        # Regression test for a fixed bug: when query params were present
        # but none of them were 'people'/'year_start'/'year_end' (e.g. just
        # '?key=...'), `p_query` was left as None and
        # `ImageFile.objects.filter(p_query)` raised TypeError instead of
        # falling back to "all images" the way the no-params case does.
        # The header-based key auth path (tested above) doesn't hit this
        # because it sends no query params at all.
        resp = self.anon_client.get(f"/api/image_list/?key={settings.SLIDESHOW_API_KEY}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_authenticated_session_bypasses_key_requirement(self):
        resp = self.client.get("/api/image_list/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


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

    def test_face_array_serves_thumbnail_bytes_directly(self):
        # face_array used to decode/resize/re-encode the thumbnail via PIL
        # on every request even though no resize was ever actually applied
        # (no height/width is set for this branch) - it now serves the
        # pre-generated file's bytes as-is. A byte-exact match here would
        # catch a regression back to the PIL round-trip, since re-encoding
        # a JPEG (even at identical dimensions) reliably changes its bytes.
        resp = self.client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.face.face_thumbnail.open("rb")
        expected = self.face.face_thumbnail.read()
        self.face.face_thumbnail.close()
        self.assertEqual(resp.content, expected)

    def test_keyed_image_response_is_cacheable(self):
        resp = self.client.get(f"/api/keyed_image/face_array/?id={self.face.id}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("max-age", resp["Cache-Control"])
        self.assertIn("immutable", resp["Cache-Control"])


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


class MobileEndpointTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        real = Person.objects.create(person_name="Mobile Test Person")
        self.unlabeled_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=real, weight_1=0.9
        )

    def test_confident_unlabeled_lists_unlabeled_faces(self):
        resp = self.client.get("/api/mobile/confident_unlabeled/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertIn(self.unlabeled_face.id, data["unlabeled_ids"])

    def test_unlabeled_instance_info(self):
        resp = self.client.get(f"/api/mobile/unlabeled_instance/{self.unlabeled_face.id}/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertIn("face_img_url", data)
        self.assertIn("ignore_url", data)


class StatsAndParametersTests(ApiTestCase):
    def test_parameters_endpoint_requires_key_or_auth(self):
        resp = self.anon_client.get("/api/parameters/")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

        resp = self.client.get("/api/parameters/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_stats_endpoint_requires_auth(self):
        resp = self.anon_client.get("/api/server_stats/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_stats_endpoint_with_data(self):
        img = self.make_image()
        img.isProcessed = True
        img.save()
        self.make_face(img)
        resp = self.client.get("/api/server_stats/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(resp.data["num_imgs"], 1)
        self.assertGreaterEqual(resp.data["num_faces"], 1)


class MobileViewTests(ApiTestCase):
    """Tests for api/mobile_views.py -- split out of api/views.py, which
    had grown to mix these mobile-app-facing endpoints in with the
    standard ModelViewSets and slideshow-facing ones."""

    def test_confident_unlabeled_with_no_unlabeled_faces_returns_empty_list(self):
        # Regression test for a fixed bug: this used to do
        # `unlabeled[0].weight_1` unconditionally, raising IndexError the
        # moment there were zero unlabeled faces -- the *goal* state of
        # the tagging workflow (or a fresh/near-empty DB), not an edge
        # case, so this crashed exactly when tagging was fully caught up.
        resp = self.client.get("/api/mobile/confident_unlabeled/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(resp.content)["unlabeled_ids"], [])

    def test_confident_unlabeled_with_unlabeled_faces_returns_their_ids(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        real = Person.objects.create(person_name="Real Person")
        img = self.make_image()
        # A face is only served if at least one poss_identN is a real person.
        face = self.make_face(img, declared_name=blank, poss_ident1=real, weight_1=0.9)

        resp = self.client.get("/api/mobile/confident_unlabeled/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(resp.content)["unlabeled_ids"], [face.id])

    def test_confident_unlabeled_excludes_faces_with_only_sentinel_guesses(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        realignore = Person.objects.get(person_name=".realignore")
        img = self.make_image()

        # No guesses at all -> excluded.
        self.make_face(img, declared_name=blank)
        # Only sentinel guesses -> excluded.
        self.make_face(img, declared_name=blank, poss_ident1=ignore, weight_1=0.9)
        self.make_face(
            img, declared_name=blank, poss_ident1=realignore, poss_ident2=blank
        )

        resp = self.client.get("/api/mobile/confident_unlabeled/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(resp.content)["unlabeled_ids"], [])

    def test_confident_unlabeled_excludes_face_with_any_sentinel_guess(self):
        # A single sentinel anywhere in the top five disqualifies the face,
        # even alongside a real-person guess.
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        real = Person.objects.create(person_name="Real Person")
        img = self.make_image()
        self.make_face(
            img,
            declared_name=blank,
            poss_ident1=real,
            weight_1=0.9,
            poss_ident2=ignore,
            weight_2=0.7,
        )

        resp = self.client.get("/api/mobile/confident_unlabeled/")
        self.assertEqual(json.loads(resp.content)["unlabeled_ids"], [])

    def test_confident_unlabeled_includes_face_with_real_guesses_and_empty_slots(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        real1 = Person.objects.create(person_name="Real One")
        real2 = Person.objects.create(person_name="Real Two")
        img = self.make_image()
        # poss_ident3..5 left null -- that must not disqualify the face.
        face = self.make_face(
            img,
            declared_name=blank,
            poss_ident1=real1,
            weight_1=0.9,
            poss_ident2=real2,
            weight_2=0.6,
        )

        resp = self.client.get("/api/mobile/confident_unlabeled/")
        self.assertEqual(json.loads(resp.content)["unlabeled_ids"], [face.id])

    def test_reset_face_clears_person_and_returns_a_response(self):
        # Regression test for a fixed bug: this method had no return
        # statement, so DRF's dispatch() got None back instead of a
        # Response and raised AssertionError -- this crashed on every
        # single call, not just an edge case.
        img = self.make_image()
        target = Person.objects.create(person_name="Resettable")
        target.highlight_img.save("r.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        face = self.make_face(img, declared_name=target)

        resp = self.client.patch(f"/api/mobile/reset/{face.id}/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        face.refresh_from_db()
        # clear_person() sets declared_name to None (not the blank
        # sentinel Person) -- it's a nullable field, unlike associate_
        # person()'s reassignment-based clearing elsewhere in this file.
        self.assertIsNone(face.declared_name)

    def test_name_list_returns_real_people_and_excludes_sentinel_names(self):
        # Regression test for a fixed bug: this used to be an unfinished
        # stub returning a hardcoded ['a', 'b', 'c', 'd'] regardless of
        # what Person rows actually existed.
        Person.objects.create(person_name="Alice")
        Person.objects.create(person_name="Bob")

        resp = self.client.get("/api/mobile/name_list/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        names = json.loads(resp.content)["name_list"]
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)
        for ignored in settings.IGNORED_NAMES:
            self.assertNotIn(ignored, names)


@override_settings(
    AUTHELIA_ISSUER="https://auth.test.example",
    AUTHELIA_MOBILE_CLIENT_ID="photoverify_mobile",
)
class AutheliaOIDCAuthenticationTests(TestCase):
    """api/authentication.py:AutheliaOIDCAuthentication -- the PhotoVerify
    mobile app sends an Authelia-issued OIDC ID token as a Bearer token;
    this class verifies its RS256 signature against Authelia's JWKS and
    maps the email claim to a local user. Here the JWKS lookup is stubbed
    with a locally generated RSA keypair so no network / real Authelia is
    involved."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ensure_sentinel_people()
        from cryptography.hazmat.primitives.asymmetric import rsa

        cls._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def setUp(self):
        import time
        from unittest.mock import patch

        self._time = time
        self.user = User.objects.create_user(
            username="mobileuser", email="Person@Example.com", password="pw123456"
        )
        # get_signing_key_from_jwt(token) -> object with a .key attribute
        # (the verifying key). Return the public half of our test keypair.
        signing_key = type("K", (), {"key": self._key.public_key()})()
        patcher = patch(
            "api.authentication._jwks_client.get_signing_key_from_jwt",
            return_value=signing_key,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = APIClient()

    def _token(self, **overrides):
        import jwt

        now = int(self._time.time())
        claims = {
            "iss": "https://auth.test.example",
            "aud": "photoverify_mobile",
            "sub": "abc123",
            "iat": now,
            "exp": now + 3600,
            "email": "person@example.com",
        }
        claims.update(overrides)
        for k in [k for k, v in overrides.items() if v is None]:
            claims.pop(k, None)
        return jwt.encode(claims, self._key, algorithm="RS256")

    def _get(self, token):
        return self.client.get(
            "/api/mobile/name_list/", HTTP_AUTHORIZATION=f"Bearer {token}"
        )

    def test_valid_token_authenticates_and_maps_email_case_insensitively(self):
        resp = self._get(self._token())
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    REJECTED = (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_no_bearer_header_is_rejected_not_500(self):
        resp = self.client.get("/api/mobile/name_list/")
        self.assertIn(resp.status_code, self.REJECTED)

    def test_garbage_bearer_value_is_rejected_not_500(self):
        # A bearer value that isn't a JWT at all -- get_signing_key_from_jwt
        # raises jwt.DecodeError parsing the header; must not 500.
        for junk in ["not.a.jwt", "garbage", "a.b.c"]:
            resp = self._get(junk)
            self.assertIn(resp.status_code, self.REJECTED, junk)

    def test_expired_token_rejected(self):
        now = int(self._time.time())
        resp = self._get(self._token(iat=now - 7200, exp=now - 3600))
        self.assertIn(resp.status_code, self.REJECTED)
        self.assertIn("expired", resp.content.decode().lower())

    def test_wrong_audience_rejected(self):
        resp = self._get(self._token(aud="some_other_client"))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_wrong_issuer_rejected(self):
        resp = self._get(self._token(iss="https://evil.example"))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_token_signed_by_a_different_key_rejected(self):
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa

        attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = int(self._time.time())
        forged = jwt.encode(
            {
                "iss": "https://auth.test.example",
                "aud": "photoverify_mobile",
                "sub": "abc123",
                "iat": now,
                "exp": now + 3600,
                "email": "person@example.com",
            },
            attacker_key,
            algorithm="RS256",
        )
        self.assertIn(self._get(forged).status_code, self.REJECTED)

    def test_unknown_email_rejected(self):
        resp = self._get(self._token(email="nobody@example.com"))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_missing_email_claim_rejected(self):
        resp = self._get(self._token(email=None))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_inactive_user_rejected(self):
        self.user.is_active = False
        self.user.save()
        resp = self._get(self._token())
        self.assertIn(resp.status_code, self.REJECTED)
