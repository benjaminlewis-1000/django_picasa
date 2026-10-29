"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_mobile
"""
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

from api.tests import FaceFixtureMixin, ApiTestCase, _tiny_jpeg_bytes, ensure_sentinel_people


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

    def test_verify_candidates_excludes_clustered_faces(self):
        # A face with a verification_cluster_group belongs to a batch the
        # web app's "Group by cluster" mode reviews in bulk -- mobile
        # should never serve it, and it shouldn't count toward the pile.
        bob = Person.objects.create(person_name="Bob V")
        img = self.make_image()
        clustered = [
            self.make_face(
                img, declared_name=bob, validated=False, verification_cluster_group=0,
            )
            for _ in range(3)
        ]
        singleton = self.make_face(img, declared_name=bob, validated=False)

        data = json.loads(self.client.get("/api/mobile/verify_candidates/").content)
        self.assertEqual(data["person_name"], "Bob V")
        self.assertEqual(data["unverified_count"], 1)
        served = {f["id"] for f in data["faces"]}
        self.assertEqual(served, {singleton.id})
        self.assertFalse(served & {f.id for f in clustered})

    def test_verify_candidates_skips_person_whose_whole_pile_is_clustered(self):
        clustered_person = Person.objects.create(person_name="All Clustered")
        img = self.make_image()
        for _ in range(4):
            self.make_face(
                img, declared_name=clustered_person, validated=False,
                verification_cluster_group=1,
            )
        alice = Person.objects.create(person_name="Alice V")
        self._unverified_face(alice, 2)

        data = json.loads(self.client.get("/api/mobile/verify_candidates/").content)
        self.assertEqual(data["person_name"], "Alice V")

        # Pinning the fully-clustered person directly also falls through
        # to nothing (their pile is entirely ineligible).
        pinned = json.loads(
            self.client.get(
                f"/api/mobile/verify_candidates/?person_id={clustered_person.id}"
            ).content
        )
        self.assertNotEqual(pinned["person_id"], clustered_person.id)

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


