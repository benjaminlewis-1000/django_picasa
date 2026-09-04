import json
import unittest
from io import BytesIO

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.test import TestCase, TransactionTestCase, override_settings
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


class IgnoreReviewFlaggedPartitionTests(ApiTestCase):
    # The frontend's ".ignore" main unlabeled screen and its "Flagged for
    # review" subordinate row (PersonParamView's face_poss `flagged`
    # param) are meant to be a complementary partition of the same
    # poss_ident1 candidates - a face flagged via the mobile app's
    # ignore-review flow (mobile_review_hidden=True) should show up in
    # exactly one of the two, never both and never neither.
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.flagged_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.9, mobile_review_hidden=True,
        )
        self.plain_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.5, mobile_review_hidden=False,
        )
        self.never_reviewed_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.3,
        )

    def test_default_face_poss_excludes_flagged(self):
        resp = self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertNotIn(self.flagged_face.id, id_list)
        self.assertIn(self.plain_face.id, id_list)
        self.assertIn(self.never_reviewed_face.id, id_list)

    def test_flagged_true_returns_only_flagged(self):
        resp = self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss?flagged=true")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertEqual(id_list, [self.flagged_face.id])

    def test_flagged_and_unflagged_partition_the_full_poss_ident1_set(self):
        # Explicit invariant, not just implied by the two tests above:
        # every face with poss_ident1=.ignore appears in exactly one of
        # the two queries, and their union is the whole set with no
        # overlap - queried straight against the DB (not the stored
        # Person.num_possibilities counter, which is separate
        # bookkeeping - see IgnoreReviewFlaggedCountTests below) so this
        # would catch the query itself regressing back to overlapping or
        # dropping rows, independent of that counter ever being right.
        main_ids = set(json.loads(
            self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss").content
        )["id_list"])
        flagged_ids = set(json.loads(
            self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss?flagged=true").content
        )["id_list"])
        all_poss_ident1_ids = set(
            Face.objects.filter(poss_ident1=self.ignore).values_list("id", flat=True)
        )

        self.assertEqual(main_ids & flagged_ids, set(), "a face should never appear in both")
        self.assertEqual(main_ids | flagged_ids, all_poss_ident1_ids, "union should cover every poss_ident1 face")

    def test_real_person_still_shows_a_mobile_review_hidden_candidate(self):
        # Regression test: mobile_review_hidden is set not just by the
        # .ignore-review flow, but also by the mobile app's plain "Skip"
        # action on *any* real person's suggestion queue
        # (HideFromMobile/LabelingGroupsView) - a real person has no
        # "Flagged for review" view the way .ignore does, so excluding
        # mobile_review_hidden=True faces from their face_poss query made
        # a skipped candidate disappear from the gallery entirely (while
        # still counting toward the sidebar's num_possibilities, a
        # separate stored field) - reported by the user 2026-09-02 as "a
        # person shows one unlabeled face but the gallery is empty."
        real_person = Person.objects.create(person_name="Garrett Egan Test")
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        skipped_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=real_person,
            weight_1=0.2, mobile_review_hidden=True,
        )
        resp = self.client.get(f"/api/paginate_obj_ids/{real_person.id}/face_poss")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertIn(skipped_face.id, id_list)


@override_settings(MEDIA_ROOT="/tmp/api_test_media")
class IgnoreReviewFlaggedCountTests(FaceFixtureMixin, TransactionTestCase):
    # PersonListView farms its per-person work out to worker threads
    # (api/views.py), each on its own DB connection - a plain TestCase's
    # rolled-back-at-teardown transaction is only visible on the
    # connection that opened it, so fixture rows created under a regular
    # ApiTestCase are invisible to those worker threads' queries. Needs
    # TransactionTestCase (real commits) instead, just for this one
    # count - IgnoreReviewFlaggedPartitionTests above covers the
    # non-threaded PersonParamView query the same fixture shape feeds.
    def setUp(self):
        ensure_sentinel_people()
        self.user = User.objects.create_user(username="tester_flagged_count", password="pw123456")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

        self.image = self.make_image()
        self.ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.9, mobile_review_hidden=True,
        )

    def test_person_list_num_possibilities_excludes_flagged_for_ignore(self):
        # .ignore's stored num_possibilities counter (Person model field)
        # counts every poss_ident1 candidate regardless of the flag - the
        # sidebar's main-screen count needs the same exclusion the query
        # above applies, or it would overcount relative to what the main
        # screen actually shows.
        Person.objects.filter(pk=self.ignore.pk).update(num_possibilities=3)
        resp = self.client.get("/api/person_list/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = json.loads(resp.content)["results"]
        ignore_dict = next(p for p in results if p["id"] == self.ignore.id)
        self.assertEqual(ignore_dict["num_possibilities"], 2)
        self.assertEqual(ignore_dict["num_review_flagged"], 1)


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
        self.assertTrue(data["is_unassigned"])
        self.assertEqual(data["declared_name"], settings.BLANK_FACE_NAME)

    def test_unlabeled_instance_info_flags_a_face_resolved_elsewhere(self):
        # Face got assigned to a real person after the mobile queue was
        # snapshotted -- the app uses is_unassigned to skip it silently.
        someone = Person.objects.create(person_name="Now Tagged")
        someone.highlight_img.save("t.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        self.unlabeled_face.associate_person(someone.id)

        resp = self.client.get(f"/api/mobile/unlabeled_instance/{self.unlabeled_face.id}/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertFalse(data["is_unassigned"])
        self.assertEqual(data["declared_name"], "Now Tagged")


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

    def test_labeling_groups_grouped_by_top_guess_ordered_by_size(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        alice = Person.objects.create(person_name="Alice")
        bob = Person.objects.create(person_name="Bob")
        img = self.make_image()

        # Alice: 1 face. Bob: 2 faces (should sort ahead of Alice).
        a1 = self.make_face(img, declared_name=blank, poss_ident1=alice, weight_1=0.5)
        b_lo = self.make_face(img, declared_name=blank, poss_ident1=bob, weight_1=0.3)
        b_hi = self.make_face(img, declared_name=blank, poss_ident1=bob, weight_1=0.9)
        # poss_ident1 == .ignore -> excluded from this endpoint entirely.
        self.make_face(img, declared_name=blank, poss_ident1=ignore, weight_1=0.9)

        resp = self.client.get("/api/mobile/labeling_groups/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        groups = json.loads(resp.content)["groups"]

        self.assertEqual([g["person_name"] for g in groups], ["Bob", "Alice"])
        self.assertEqual(groups[0]["count"], 2)
        # Within a group: highest weight first.
        self.assertEqual(groups[0]["face_ids"], [b_hi.id, b_lo.id])
        self.assertEqual(groups[1]["face_ids"], [a1.id])

    def test_mobile_hide_marks_faces_and_drops_them_from_labeling_groups(self):
        # The main screen's "Skip" hides a face from the mobile app via
        # mobile_review_hidden; LabelingGroupsView must then stop listing it.
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        alice = Person.objects.create(person_name="Alice")
        img = self.make_image()
        keep = self.make_face(img, declared_name=blank, poss_ident1=alice, weight_1=0.9)
        skip = self.make_face(img, declared_name=blank, poss_ident1=alice, weight_1=0.5)

        resp = self.client.patch(
            "/api/mobile/hide/", {"face_ids": [skip.id]}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(resp.content)["hidden"], 1)

        skip.refresh_from_db()
        self.assertTrue(skip.mobile_review_hidden)

        groups = json.loads(self.client.get("/api/mobile/labeling_groups/").content)["groups"]
        self.assertEqual(groups[0]["face_ids"], [keep.id])

    def test_reset_face_returns_it_to_the_unassigned_pool(self):
        # Regression test: reset used to call clear_person(), which nulls
        # declared_name -- invisible to both the Unassigned bucket and the
        # re-classifier (both filter declared_name__person_name ==
        # BLANK_FACE_NAME). Reset must land the face back on the blank
        # sentinel Person, not NULL, with its guesses cleared.
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = self.make_image()
        target = Person.objects.create(person_name="Resettable")
        target.highlight_img.save("r.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        other = Person.objects.create(person_name="Someone Else")
        face = self.make_face(
            img, declared_name=target, poss_ident1=other, weight_1=0.8
        )

        resp = self.client.patch(f"/api/mobile/reset/{face.id}/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        face.refresh_from_db()
        self.assertEqual(face.declared_name_id, blank.id)
        self.assertIsNone(face.poss_ident1_id)
        self.assertFalse(face.validated)
        # And it now shows up in the unassigned listing.
        self.assertEqual(
            Face.objects.filter(
                declared_name__person_name=settings.BLANK_FACE_NAME
            ).filter(pk=face.pk).count(),
            1,
        )

    def test_reset_face_already_unassigned_just_clears_guesses(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        other = Person.objects.create(person_name="Guess Person")
        img = self.make_image()
        face = self.make_face(
            img, declared_name=blank, poss_ident1=other, weight_1=0.7
        )
        blank_faces_before = blank.face_declared.count()

        resp = self.client.patch(f"/api/mobile/reset/{face.id}/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        face.refresh_from_db()
        self.assertEqual(face.declared_name_id, blank.id)
        self.assertIsNone(face.poss_ident1_id)
        # No double-count: it was already the blank sentinel's face.
        self.assertEqual(blank.face_declared.count(), blank_faces_before)

    def _ignore_candidate(self, **overrides):
        """An unlabeled face whose #1 guess is .ignore -- what the mobile
        ignore-review grid lists."""
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        kw = dict(declared_name=blank, poss_ident1=ignore, weight_1=0.9)
        kw.update(overrides)
        return self.make_face(self.make_image(), **kw)

    def test_ignore_candidates_lists_unlabeled_faces_with_ignore_as_top_guess(self):
        want = self._ignore_candidate()
        # declared .ignore (already labelled) -> excluded
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.make_face(self.make_image(), declared_name=ignore)
        # .ignore is only guess #2 -> excluded
        real = Person.objects.create(person_name="Real One")
        self._ignore_candidate(poss_ident1=real, poss_ident2=ignore, weight_2=0.5)

        # explicitly hidden -> excluded
        self._ignore_candidate(mobile_review_hidden=True)

        resp = self.client.get("/api/mobile/ignore_candidates/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertEqual([f["id"] for f in data["faces"]], [want.id])
        self.assertIn("access_key=", data["faces"][0]["face_img_url"])

    def test_ignore_candidates_respects_limit_and_samples_the_pool(self):
        all_ids = {self._ignore_candidate().id for _ in range(8)}

        page = json.loads(
            self.client.get("/api/mobile/ignore_candidates/?limit=3").content
        )["faces"]
        self.assertEqual(len(page), 3)
        self.assertTrue({f["id"] for f in page} <= all_ids)

        # Random order: over several calls we should see more than one
        # distinct "first" face out of a pool of 8.
        firsts = {
            json.loads(
                self.client.get("/api/mobile/ignore_candidates/?limit=1").content
            )["faces"][0]["id"]
            for _ in range(12)
        }
        self.assertGreater(len(firsts), 1)

    def test_bulk_confirm_ignore_confirms_and_hides(self):
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        confirm = self._ignore_candidate()
        hide = self._ignore_candidate()

        resp = self.client.patch(
            "/api/mobile/bulk_confirm_ignore/",
            {"confirm_ids": [confirm.id], "hide_ids": [hide.id]},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        body = json.loads(resp.content)
        self.assertEqual(body["confirmed"], 1)
        self.assertEqual(body["hidden"], 1)

        confirm.refresh_from_db()
        self.assertEqual(confirm.declared_name_id, ignore.id)

        hide.refresh_from_db()
        self.assertEqual(hide.declared_name_id, blank.id)  # untouched...
        self.assertEqual(hide.poss_ident1_id, ignore.id)  # ...still a proposed .ignore
        self.assertTrue(hide.mobile_review_hidden)
        # ...and no longer listed.
        listed = json.loads(self.client.get("/api/mobile/ignore_candidates/").content)
        self.assertNotIn(hide.id, [f["id"] for f in listed["faces"]])

    def test_bulk_confirm_ignore_skips_stale_ids(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        real = Person.objects.create(person_name="Not Ignore")
        wrong_top_guess = self._ignore_candidate(poss_ident1=real)  # #1 isn't .ignore
        already_declared = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        declared = self.make_face(self.make_image(), declared_name=already_declared)

        resp = self.client.patch(
            "/api/mobile/bulk_confirm_ignore/",
            {
                "confirm_ids": [wrong_top_guess.id, declared.id, 999999],
                "hide_ids": [declared.id],
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        body = json.loads(resp.content)
        self.assertEqual(body["confirmed"], 0)
        self.assertEqual(body["hidden"], 0)
        self.assertEqual(body["skipped"], 4)

    def test_bulk_confirm_ignore_undo_reverts_confirm_and_hides(self):
        """The "undo last screen" path: a face that was confirmed as
        .ignore goes back to unlabeled with .ignore restored as the top
        guess, and hidden from the grid."""
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        face = self._ignore_candidate()

        # Forward: confirm it as .ignore.
        self.client.patch(
            "/api/mobile/bulk_confirm_ignore/",
            {"confirm_ids": [face.id]},
            format="json",
        )
        face.refresh_from_db()
        self.assertEqual(face.declared_name_id, ignore.id)

        # Undo: tapped on the go-back screen.
        resp = self.client.patch(
            "/api/mobile/bulk_confirm_ignore/",
            {"undo_ids": [face.id]},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(resp.content)["undone"], 1)

        face.refresh_from_db()
        self.assertEqual(face.declared_name_id, blank.id)     # back to unlabeled
        self.assertEqual(face.poss_ident1_id, ignore.id)      # .ignore restored as #1
        self.assertEqual(face.weight_1, 1.0)
        self.assertIsNone(face.poss_ident2_id)
        self.assertFalse(face.validated)
        self.assertTrue(face.mobile_review_hidden)

        # No longer listed (hidden).
        listed = json.loads(self.client.get("/api/mobile/ignore_candidates/").content)
        self.assertNotIn(face.id, [f["id"] for f in listed["faces"]])

    def test_bulk_confirm_ignore_undo_skips_unknown_ids(self):
        resp = self.client.patch(
            "/api/mobile/bulk_confirm_ignore/",
            {"undo_ids": [999999]},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        body = json.loads(resp.content)
        self.assertEqual(body["undone"], 0)
        self.assertEqual(body["skipped"], 1)

    # --- verify-faces screen ------------------------------------------------

    def _unverified_face(self, person, n=1):
        img = self.make_image()
        return [
            self.make_face(img, declared_name=person, validated=False) for _ in range(n)
        ]

    def test_verify_candidates_picks_the_biggest_unverified_pile(self):
        alice = Person.objects.create(person_name="Alice V")
        bob = Person.objects.create(person_name="Bob V")
        self._unverified_face(alice, 2)
        bob_faces = self._unverified_face(bob, 3)
        # a verified Bob face and an ignore face should not count
        vf = self._unverified_face(bob, 1)[0]
        vf.validated = True
        vf.save()

        resp = self.client.get("/api/mobile/verify_candidates/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertEqual(data["person_name"], "Bob V")
        self.assertEqual(data["unverified_count"], 3)
        self.assertTrue({f["id"] for f in data["faces"]} <= {f.id for f in bob_faces})

    def test_verify_candidates_exclude_skips_a_person(self):
        alice = Person.objects.create(person_name="Alice V")
        bob = Person.objects.create(person_name="Bob V")
        self._unverified_face(alice, 2)
        self._unverified_face(bob, 5)

        data = json.loads(
            self.client.get(f"/api/mobile/verify_candidates/?exclude={bob.id}").content
        )
        self.assertEqual(data["person_name"], "Alice V")

    def test_verify_candidates_empty_when_nothing_unverified(self):
        data = json.loads(self.client.get("/api/mobile/verify_candidates/").content)
        self.assertIsNone(data["person_id"])
        self.assertEqual(data["faces"], [])

    def test_verify_candidates_person_id_pins_the_person_even_when_smaller(self):
        # The app pins a person for the session so unverified_count tracks
        # exactly what the reviewer is working on -- it must NOT jump to a
        # now-bigger pile just because the pinned person shrank.
        alice = Person.objects.create(person_name="Alice V")
        bob = Person.objects.create(person_name="Bob V")
        self._unverified_face(alice, 2)
        self._unverified_face(bob, 5)

        data = json.loads(
            self.client.get(f"/api/mobile/verify_candidates/?person_id={alice.id}").content
        )
        self.assertEqual(data["person_name"], "Alice V")
        self.assertEqual(data["person_id"], alice.id)
        self.assertTrue(len(data["faces"]) > 0)
        # Pinned loads skip the COUNT -- the app tracks it locally.
        self.assertIsNone(data["unverified_count"])

    def test_verify_candidates_falls_off_pin_when_person_exhausted(self):
        alice = Person.objects.create(person_name="Alice V")
        bob = Person.objects.create(person_name="Bob V")
        self._unverified_face(bob, 3)  # alice has none left

        data = json.loads(
            self.client.get(f"/api/mobile/verify_candidates/?person_id={alice.id}").content
        )
        self.assertEqual(data["person_name"], "Bob V")

    def test_verify_candidates_pin_respects_exclude(self):
        alice = Person.objects.create(person_name="Alice V")
        bob = Person.objects.create(person_name="Bob V")
        self._unverified_face(alice, 4)
        self._unverified_face(bob, 1)

        data = json.loads(
            self.client.get(
                f"/api/mobile/verify_candidates/?person_id={alice.id}&exclude={alice.id}"
            ).content
        )
        self.assertEqual(data["person_name"], "Bob V")

    def test_bulk_verify_verifies_and_resets(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        alice = Person.objects.create(person_name="Alice V")
        alice.highlight_img.save("a.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        keep, wrong = self._unverified_face(alice, 2)

        resp = self.client.patch(
            "/api/mobile/bulk_verify/",
            {"verify_ids": [keep.id], "reset_ids": [wrong.id]},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        body = json.loads(resp.content)
        self.assertEqual(body["verified"], 1)
        self.assertEqual(body["reset"], 1)

        keep.refresh_from_db()
        self.assertTrue(keep.validated)
        self.assertEqual(keep.declared_name_id, alice.id)

        wrong.refresh_from_db()
        self.assertEqual(wrong.declared_name_id, blank.id)  # sent to unassigned
        self.assertFalse(wrong.validated)

    def test_bulk_verify_skips_already_verified(self):
        alice = Person.objects.create(person_name="Alice V")
        f = self._unverified_face(alice, 1)[0]
        f.validated = True
        f.save()

        body = json.loads(
            self.client.patch(
                "/api/mobile/bulk_verify/",
                {"verify_ids": [f.id, 999999], "reset_ids": []},
                format="json",
            ).content
        )
        self.assertEqual(body["verified"], 0)
        self.assertEqual(body["skipped"], 2)

    def test_verify_ignore_candidates_lists_unverified_ignore_faces(self):
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        img = self.make_image()
        want = self.make_face(img, declared_name=ignore, validated=False)
        verified = self.make_face(img, declared_name=ignore, validated=True)
        alice = Person.objects.create(person_name="Alice V")
        self.make_face(img, declared_name=alice, validated=False)  # not .ignore

        data = json.loads(
            self.client.get("/api/mobile/verify_ignore_candidates/").content
        )
        self.assertEqual([f["id"] for f in data["faces"]], [want.id])

    def test_bulk_verify_works_on_ignore_faces(self):
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = self.make_image()
        keep = self.make_face(img, declared_name=ignore, validated=False)
        wrong = self.make_face(img, declared_name=ignore, validated=False)

        body = json.loads(
            self.client.patch(
                "/api/mobile/bulk_verify/",
                {"verify_ids": [keep.id], "reset_ids": [wrong.id]},
                format="json",
            ).content
        )
        self.assertEqual(body["verified"], 1)
        self.assertEqual(body["reset"], 1)
        keep.refresh_from_db()
        self.assertTrue(keep.validated)
        self.assertEqual(keep.declared_name_id, ignore.id)
        wrong.refresh_from_db()
        self.assertEqual(wrong.declared_name_id, blank.id)

    def test_bulk_verify_reset_skips_unknown_id(self):
        alice = Person.objects.create(person_name="Alice V")
        real = self._unverified_face(alice, 1)[0]
        body = json.loads(
            self.client.patch(
                "/api/mobile/bulk_verify/",
                {"verify_ids": [], "reset_ids": [real.id, 999999]},
                format="json",
            ).content
        )
        self.assertEqual(body["reset"], 1)
        self.assertEqual(body["skipped"], 1)

    def test_bulk_verify_empty_body_is_a_noop(self):
        body = json.loads(
            self.client.patch("/api/mobile/bulk_verify/", {}, format="json").content
        )
        self.assertEqual(body, {"verified": 0, "reset": 0, "skipped": 0})

    def test_verify_candidates_excludes_sentinel_declared_faces(self):
        # A pile of faces declared straight to .ignore must never be
        # offered by the "verify people" endpoint, even if it's the
        # biggest unverified pile -- those belong to verify_ignore_candidates.
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        alice = Person.objects.create(person_name="Alice V")
        self._unverified_face(ignore, 5)
        self._unverified_face(alice, 2)

        data = json.loads(self.client.get("/api/mobile/verify_candidates/").content)
        self.assertEqual(data["person_name"], "Alice V")

    def test_verify_ignore_candidates_respects_limit(self):
        ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        img = self.make_image()
        ids = {
            self.make_face(img, declared_name=ignore, validated=False).id
            for _ in range(6)
        }
        page = json.loads(
            self.client.get("/api/mobile/verify_ignore_candidates/?limit=2").content
        )["faces"]
        self.assertEqual(len(page), 2)
        self.assertTrue({f["id"] for f in page} <= ids)
        self.assertIn("access_key=", page[0]["face_img_url"])

    def test_labeling_groups_empty_when_nothing_to_label(self):
        data = json.loads(self.client.get("/api/mobile/labeling_groups/").content)
        self.assertEqual(data["groups"], [])

    def test_labeling_groups_skips_faces_with_no_top_guess(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        alice = Person.objects.create(person_name="Alice")
        img = self.make_image()
        self.make_face(img, declared_name=blank, poss_ident1=alice, weight_1=0.8)
        # poss_ident1 is null -> not groupable -> excluded
        self.make_face(img, declared_name=blank)

        groups = json.loads(
            self.client.get("/api/mobile/labeling_groups/").content
        )["groups"]
        self.assertEqual([g["person_name"] for g in groups], ["Alice"])
        self.assertEqual(groups[0]["count"], 1)

    def test_mobile_hide_reports_count_and_ignores_unknown_ids(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        alice = Person.objects.create(person_name="Alice")
        img = self.make_image()
        f = self.make_face(img, declared_name=blank, poss_ident1=alice, weight_1=0.5)

        body = json.loads(
            self.client.patch(
                "/api/mobile/hide/", {"face_ids": [f.id, 999999]}, format="json"
            ).content
        )
        self.assertEqual(body["hidden"], 1)
        f.refresh_from_db()
        self.assertTrue(f.mobile_review_hidden)

        # Idempotent: hiding it again still succeeds (update touches 1 row).
        body2 = json.loads(
            self.client.patch(
                "/api/mobile/hide/", {"face_ids": [f.id]}, format="json"
            ).content
        )
        self.assertEqual(body2["hidden"], 1)

    def test_unlabeled_instance_names_carry_hateoas_confirm_action(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        alice = Person.objects.create(person_name="Alice")
        alice.highlight_img.save("a.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        bob = Person.objects.create(person_name="Bob")
        bob.highlight_img.save("b.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        face = self.make_face(
            self.make_image(),
            declared_name=blank,
            poss_ident1=alice,
            weight_1=0.9,
            poss_ident2=bob,
            weight_2=0.4,
        )

        data = json.loads(
            self.client.get(f"/api/mobile/unlabeled_instance/{face.id}/").content
        )
        # The client never builds action URLs -- each candidate name embeds
        # the exact PATCH URL + payload to confirm it.
        self.assertEqual([n["name"] for n in data["names"]], ["Alice", "Bob"])
        alice_action = data["names"][0]
        self.assertTrue(
            alice_action["confirm_patch_url"].endswith(
                f"/faces/{face.id}/assign_face_to_person/"
            )
        )
        self.assertEqual(
            alice_action["confirm_patch_data"], {"declared_name_key": alice.id}
        )
        self.assertEqual(alice_action["weight"], 0.9)
        self.assertTrue(
            data["ignore_url"].endswith(f"/faces/{face.id}/ignore_face/")
        )
        self.assertEqual(data["ignore_payload"], {"ignore_type": "soft"})

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
        self._signing_key = type("K", (), {"key": self._key.public_key()})()
        self._jwks_patcher = patch(
            "api.authentication._jwks_client.get_signing_key_from_jwt",
            return_value=self._signing_key,
        )
        self._jwks_mock = self._jwks_patcher.start()
        self.addCleanup(self._jwks_patcher.stop)

        # The last-known-good key cache is a module global; isolate tests.
        import api.authentication as _authmod
        _authmod._last_good_keys.clear()
        self.addCleanup(_authmod._last_good_keys.clear)

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

    def test_jwks_fetch_failure_returns_503_not_401(self):
        # A transient JWKS outage must NOT look like a bad token -- 401
        # would make the app throw away a perfectly valid session.
        import jwt as _jwt

        self._jwks_mock.side_effect = _jwt.PyJWKClientError("boom")
        resp = self._get(self._token())
        self.assertEqual(resp.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)

    def test_jwks_fetch_failure_falls_back_to_last_known_good_key(self):
        import jwt as _jwt

        # 1st call succeeds -> caches the key.
        self.assertEqual(self._get(self._token()).status_code, status.HTTP_200_OK)
        # JWKS now unreachable, but we've seen this key before -> still 200.
        self._jwks_mock.side_effect = _jwt.PyJWKClientError("boom")
        self.assertEqual(self._get(self._token()).status_code, status.HTTP_200_OK)
