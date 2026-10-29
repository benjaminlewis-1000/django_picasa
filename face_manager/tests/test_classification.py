"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_classification
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

from face_manager.tests import make_face, make_person, make_image, make_preexisting_image_row, _tiny_jpeg_bytes, _embedding


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClassifyUnassignedArraySizingTests(TestCase):
    """Regression test for a fixed bug: classify_unassigned()'s
    metrics_array used to be sized to self.num_likely_people (the full
    candidate roster) rather than len(candidate_ids) (this face's roster
    minus whatever's already been rejected for it), leaving stale
    np.zeros() padding in the tail rows whenever a rejection had shrunk the
    candidate list. A real (negative) similarity could lose to that padded
    0 in np.argmax(), which then indexed candidate_id_arr -- sized to the
    *shrunk* list -- out of bounds, raising IndexError. Since execute()'s
    per-face try/except was commented out, that IndexError aborted the
    entire scheduled assign_faces run, not just this one face."""

    def setUp(self):
        self.assigner = faceAssigner()
        self.person_a = make_person("Candidate A")
        self.person_b = make_person("Candidate B")
        self.assigner.likely_people_ids = [self.person_a.id, self.person_b.id]
        self.assigner.num_likely_people = 2

        self.query = np.zeros(512)
        self.query[0] = 1.0

        # Person A's only reference encoding points the opposite
        # direction -- a real, negative cosine similarity, needed to make
        # the stale zero padding (0.0) look like a better match than any
        # real candidate under the old sizing bug.
        ref_a = -self.query.copy()
        self.assigner.embedding_dict = {self.person_a.id: ref_a.reshape(512, 1)}
        self.assigner.norm_dict = {self.person_a.id: np.array([1.0])}
        self.assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        self.face = make_face(img, declared_name=blank_person)
        self.face.face_encoding_512 = self.query.tolist()
        self.face.save()

    def test_shrunk_candidate_list_does_not_raise_indexerror(self):
        # Reject person_b and the ignore person -- only person_a is left
        # as a real candidate, but the array used to still be sized for
        # both likely people (self.num_likely_people == 2).
        self.face.rejected_fields = [self.person_b.id, self.assigner.ignore_person_id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, self.person_a.id)

    def test_all_candidates_rejected_assigns_to_ignore_without_crashing(self):
        # Every likely person has been rejected for this face -- candidate_ids
        # is empty. Must not attempt any similarity math (which would raise
        # ValueError on a size-0 array reduction) and must not crash.
        self.face.rejected_fields = [self.person_a.id, self.person_b.id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        mock_set.assert_called_once_with(self.assigner.ignore_person_id, 1, 1.0)


class P99ThresholdLookupTests(TestCase):
    """_p99_threshold_for_gallery_size() implements the gallery-size-
    adaptive accept/reject gate (see CLAUDE.md's "Face-classification
    outlier-rejection" section for the experiment this is calibrated
    from) -- checks the boundary edges land in the bucket they should."""

    def setUp(self):
        self.assigner = faceAssigner()

    def test_boundary_edges(self):
        t = self.assigner.BUCKET_THRESHOLDS
        cases = [
            (1, t[0]), (49, t[0]),
            (50, t[1]), (199, t[1]),
            (200, t[2]), (499, t[2]),
            (500, t[3]), (10_000, t[3]),
        ]
        for gallery_size, expected in cases:
            with self.subTest(gallery_size=gallery_size):
                self.assertEqual(self.assigner._p99_threshold_for_gallery_size(gallery_size), expected)


class CohesionAndSpanNudgeTests(TestCase):
    """_cohesion_and_span_nudge(): the per-person threshold correction on
    top of the size-only bucket gate -- see THRESHOLD_NUDGE_REFERENCE_N's
    own comment in faceAssigner.__init__ for the full investigation (3
    real false-positive hot spots after the -0.05 cut, root-caused via
    real production data) this is calibrated from."""

    def setUp(self):
        self.assigner = faceAssigner()

    def test_high_cohesion_wide_span_gets_no_nudge(self):
        # A normal, healthy gallery -- cohesion above the floor, span
        # well past MIN_SPAN_DAYS -- should be untouched.
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.5, span_iqr_days=365, gallery_size=1000
        )
        self.assertEqual(nudge, 0.0)

    def test_low_cohesion_small_gallery_gets_penalized(self):
        # David Redd's real profile: cohesion well below the floor, a
        # small-ish gallery so the size-scale factor doesn't crush it.
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.28, span_iqr_days=5000, gallery_size=431
        )
        self.assertGreater(nudge, 0.03)

    def test_same_low_cohesion_huge_gallery_gets_almost_no_nudge(self):
        # The exact same cohesion gap as the case above, but backed by a
        # huge gallery -- the size-scale factor should crush the penalty
        # to near zero, since a percentile over thousands of faces is
        # trustworthy even at low cohesion (this is what actually
        # separates David Redd from Joshua/Daniel Lewis in production).
        small = self.assigner._cohesion_and_span_nudge(
            cohesion=0.28, span_iqr_days=5000, gallery_size=431
        )
        huge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.28, span_iqr_days=5000, gallery_size=14000
        )
        self.assertLess(huge, small)
        self.assertLess(huge, 0.02)

    def test_narrow_single_session_gallery_gets_penalized_even_with_high_cohesion(self):
        # Emma's real profile: cohesion is actually the HIGHEST of any
        # real case checked (near-duplicate photos from one session), but
        # the near-zero span means the p99 isn't measuring real
        # person-to-person variation -- must still be penalized.
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.56, span_iqr_days=0, gallery_size=19
        )
        self.assertGreater(nudge, 0.03)

    def test_nudge_is_capped(self):
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.0, span_iqr_days=0, gallery_size=10
        )
        self.assertEqual(nudge, self.assigner.MAX_THRESHOLD_NUDGE)

    def test_nudge_never_negative(self):
        # A gallery that's BETTER than the floor on both axes must never
        # get a negative nudge (i.e. loosen the threshold further).
        nudge = self.assigner._cohesion_and_span_nudge(
            cohesion=0.9, span_iqr_days=10_000, gallery_size=10
        )
        self.assertEqual(nudge, 0.0)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClassifyUnassignedBucketGateTests(TestCase):
    """Confirms the accept/reject gate actually varies by candidate
    gallery size, not just that the lookup table itself is correct --
    a candidate whose sim_99th clears its OWN bucket's (looser, large-
    gallery) threshold but not a stricter small-gallery threshold must
    still be accepted, and vice versa."""

    def setUp(self):
        self.assigner = faceAssigner()
        self.small_person = make_person("Small Gallery Person")
        self.large_person = make_person("Large Gallery Person")
        self.assigner.likely_people_ids = [self.small_person.id, self.large_person.id]
        self.assigner.num_likely_people = 2

        self.query = np.zeros(512)
        self.query[0] = 1.0

        # A similarity strictly between the smallest bucket's threshold
        # and the largest bucket's threshold (0.558 / 0.344 as of
        # 2026-09-17's -0.05 cut to the non-<50 buckets -- see
        # BUCKET_THRESHOLDS' own comment) -- accepted for a large
        # gallery, rejected for a small one.
        self.mid_similarity = 0.50
        ref = self.query.copy()
        ref[0] = self.mid_similarity
        ref[1] = np.sqrt(1 - self.mid_similarity ** 2)  # unit vector at exactly mid_similarity cosine to query

        small_gallery_size = 10   # falls in the [10,50) bucket -> threshold 0.558
        large_gallery_size = 600  # falls in the [500+) bucket -> threshold 0.344

        self.assigner.embedding_dict = {
            self.small_person.id: np.tile(ref.reshape(512, 1), (1, small_gallery_size)),
            self.large_person.id: np.tile(ref.reshape(512, 1), (1, large_gallery_size)),
        }
        self.assigner.norm_dict = {
            self.small_person.id: np.ones(small_gallery_size),
            self.large_person.id: np.ones(large_gallery_size),
        }
        self.assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        self.face = make_face(img, declared_name=blank_person)
        self.face.face_encoding_512 = self.query.tolist()
        self.face.save()

    def test_same_similarity_accepted_for_large_gallery_rejected_for_small(self):
        # Reject the small-gallery candidate and the ignore person up
        # front, isolating whether the large-gallery candidate alone
        # gets proposed -- this only happens if its looser threshold is
        # actually being applied.
        self.face.rejected_fields = [self.small_person.id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, self.large_person.id)

    def test_same_similarity_rejects_small_gallery_when_it_is_the_only_candidate(self):
        self.face.rejected_fields = [self.large_person.id, self.assigner.ignore_person_id]
        self.face.save()

        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.face)

        # Falls through to the "no match cleared threshold" branch, which
        # (since ignore_person_id is itself rejected here) proposes the
        # best-scoring candidate anyway -- confirms it took the reject
        # path, not a confident accept, for the identical similarity
        # value that DID pass for the large-gallery candidate above.
        mock_set.assert_called_once_with(self.small_person.id, 1, self.mid_similarity)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class WindowedClassificationTests(TestCase):
    """Person.use_windowed_classification -- restricts sim_99th to
    gallery members within classification_window_days of the QUERY's
    own date, instead of the whole gallery. Built to fix a real,
    validated false-positive pattern (2026-10-25, see the field's own
    docstring): a candidate that coincidentally resembles a temporally-
    distant subset of a person's gallery (e.g. a much older/younger-
    looking era) can clear the whole-gallery p99 even though it looks
    nothing like that person's photos from NEAR the query's own date.

    Gallery here is two clusters of the SAME person, 2000 days apart,
    at two different cosine similarities to a FIXED query vector
    ([1,0,0,...]): a "close" cluster (cosine 0.65, clears the bucket
    threshold) and a "distant" cluster (cosine 0.05, does not). The
    query's own dateTakenUTC (not its embedding) is what varies between
    the "false candidate" and "true positive" scenarios -- exactly
    matching the real mechanism (the query's actual photo date decides
    which gallery members are temporally relevant), rather than
    varying the query embedding itself."""

    def _gallery_member_at_similarity(self, sim):
        # A unit vector whose cosine similarity to the fixed query
        # [1,0,0,...] is exactly `sim`.
        v = np.zeros(512)
        v[0] = sim
        v[1] = np.sqrt(1 - sim ** 2)
        return v

    def setUp(self):
        self.assigner = faceAssigner()
        self.person = make_person("Windowed Person")
        self.control_person = make_person("Non-windowed Person")
        self.assigner.likely_people_ids = [self.person.id, self.control_person.id]
        self.assigner.num_likely_people = 2

        self.EARLY_DATE = 0.0
        self.LATE_DATE = 2000 * 86400.0  # 2000 days later
        self.close_sim = 0.65   # clears the <50-gallery bucket threshold (0.558)
        self.distant_sim = 0.05  # does not clear it

        # "close" cluster (dated EARLY) is a genuinely good match to the
        # fixed query; "distant" cluster (dated LATE) is not.
        close_embs = [self._gallery_member_at_similarity(self.close_sim) for _ in range(10)]
        distant_embs = [self._gallery_member_at_similarity(self.distant_sim) for _ in range(10)]
        embeddings = np.array(close_embs + distant_embs).T  # 512 x 20
        dates = np.array([self.EARLY_DATE] * 10 + [self.LATE_DATE] * 10)

        self.assigner.embedding_dict = {
            self.person.id: embeddings,
            self.control_person.id: embeddings.copy(),
        }
        self.assigner.norm_dict = {
            self.person.id: np.ones(20),
            self.control_person.id: np.ones(20),
        }
        self.assigner.candidate_dict = {
            self.person.id: pd.DataFrame({'dateTakenUTC': dates}),
            self.control_person.id: pd.DataFrame({'dateTakenUTC': dates}),
        }
        self.assigner._build_concatenated_gallery()

        self.person.use_windowed_classification = True
        self.person.classification_window_days = 182
        self.person.save()
        self.assigner.windowed_people = {self.person.id: 182}

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()

        # The fixed query vector, shared by every test -- what varies is
        # only the query's own dateTakenUTC.
        query_vec = np.zeros(512)
        query_vec[0] = 1.0
        self.query_face = make_face(img, declared_name=blank_person)
        self.query_face.face_encoding_512 = query_vec.tolist()
        self.query_face.save()

    def test_false_candidate_rejected_when_windowed_but_would_pass_unwindowed(self):
        # Query is dated LATE (matching the "distant"/dissimilar
        # cluster) but coincidentally resembles the temporally-UNrelated
        # "close" cluster (dated EARLY) well enough to clear the whole-
        # gallery p99 -- the real Leslie Williams failure shape (a
        # stranger who happens to resemble an old/different-era photo).
        self.query_face.dateTakenUTC = self.LATE_DATE
        self.query_face.rejected_fields = [self.person.id, self.assigner.ignore_person_id]
        self.query_face.save()

        # Sanity check first: WITHOUT windowing (the control person,
        # identical gallery/dates, just not opted in), this exact query
        # DOES get accepted -- confirms the test actually exercises the
        # failure mode windowing is meant to fix, not a strawman.
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.control_person.id, 1, self.close_sim, save=False)

        # WITH windowing: only the temporally-relevant "distant" cluster
        # (dated LATE, matching the query's own date) is considered --
        # correctly rejected instead.
        self.query_face.declared_name = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.person.id, 1, self.distant_sim)

    def test_true_positive_still_accepted_when_windowed(self):
        # Query dated EARLY, matching the genuinely-similar "close"
        # cluster -- windowing restricts to exactly that cluster and
        # still accepts.
        self.query_face.dateTakenUTC = self.EARLY_DATE
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.person.id, 1, self.close_sim, save=False)

    def test_falls_back_to_full_gallery_when_too_few_in_window(self):
        # A narrower window than any date gap present leaves <
        # MIN_WINDOW_SAMPLE (5) members in-window -- falls back to the
        # full gallery rather than computing a near-meaningless
        # percentile over a handful of points. Query dated well past
        # BOTH clusters with a tight window, so neither cluster
        # qualifies (not just a 1-day window around the exact LATE
        # cluster date, which would still trivially include all 10
        # same-dated LATE members).
        self.assigner.windowed_people = {self.person.id: 10}  # 10-day window
        self.query_face.dateTakenUTC = self.LATE_DATE + 1000 * 86400  # 1000 days past LATE
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        # Falls back to the full (unwindowed) gallery -- same as the
        # control-person case above, accepted via the close cluster.
        mock_set.assert_called_once_with(self.person.id, 1, self.close_sim, save=False)

    def test_no_usable_query_date_falls_back_to_full_gallery(self):
        self.query_face.dateTakenUTC = 0  # the field's own "no real date" default
        self.query_face.rejected_fields = [self.control_person.id, self.assigner.ignore_person_id]
        self.query_face.save()
        with patch.object(Face, "set_possible_person") as mock_set:
            self.assigner.classify_unassigned(self.query_face)
        mock_set.assert_called_once_with(self.person.id, 1, self.close_sim, save=False)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class IgnoreWeightMarginTests(TestCase):
    """The ignore-branch weight is deliberately the OPPOSITE sense of a
    real-match weight: frontend sorts by weight descending, and a face
    that's confidently far from every candidate (a safe, obvious ignore)
    should surface before a face that nearly cleared someone's bar (a
    genuine near-miss worth a human's eyes). Uses each candidate's own
    margin below ITS bucket threshold, not a raw score, since raw scores
    aren't comparable across differently-thresholded buckets."""

    def _make_face_with_single_candidate(self, similarity, gallery_size):
        assigner = faceAssigner()
        person = make_person(f"Candidate {similarity}-{gallery_size}")
        assigner.likely_people_ids = [person.id]

        query = np.zeros(512)
        query[0] = 1.0
        ref = query.copy()
        ref[0] = similarity
        ref[1] = np.sqrt(max(0.0, 1 - similarity ** 2))

        assigner.embedding_dict = {person.id: np.tile(ref.reshape(512, 1), (1, gallery_size))}
        assigner.norm_dict = {person.id: np.ones(gallery_size)}
        assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        face = make_face(img, declared_name=blank_person)
        face.face_encoding_512 = query.tolist()
        face.save()
        return assigner, face

    def test_near_miss_gets_low_ignore_weight(self):
        # Scores 0.004 below the large-gallery bucket's own threshold
        # (derived from the live attribute, not a hardcoded absolute
        # value, so this stays valid regardless of where that threshold
        # is calibrated to) -- should sort near the back.
        margin = 0.004
        near_miss_similarity = faceAssigner().BUCKET_THRESHOLDS[-1] - margin
        assigner, face = self._make_face_with_single_candidate(
            similarity=near_miss_similarity, gallery_size=600
        )

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)

        mock_set.assert_called_once()
        _, precedence, weight = mock_set.call_args[0]
        expected = margin / assigner.IGNORE_WEIGHT_MARGIN_CLAMP
        self.assertAlmostEqual(weight, expected, places=3)
        self.assertLess(weight, 0.1)

    def test_far_from_everyone_gets_high_ignore_weight(self):
        # Same bucket/threshold, but scoring far below it (margin well
        # past the clamp) -- should sort to the very front.
        assigner, face = self._make_face_with_single_candidate(similarity=0.0, gallery_size=600)

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)

        mock_set.assert_called_once()
        _, precedence, weight = mock_set.call_args[0]
        self.assertEqual(weight, 1.0)

    def test_far_beats_near_miss_in_sort_order(self):
        near_miss_similarity = faceAssigner().BUCKET_THRESHOLDS[-1] - 0.004
        near_miss_assigner, near_miss_face = self._make_face_with_single_candidate(
            similarity=near_miss_similarity, gallery_size=600
        )
        far_assigner, far_face = self._make_face_with_single_candidate(similarity=0.0, gallery_size=600)

        with patch.object(Face, "set_possible_person") as mock_set:
            near_miss_assigner.classify_unassigned(near_miss_face)
            near_miss_weight = mock_set.call_args[0][2]

            far_assigner.classify_unassigned(far_face)
            far_weight = mock_set.call_args[0][2]

        self.assertGreater(far_weight, near_miss_weight)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class DetScoreMarginGateTests(TestCase):
    """DET_SCORE_FLOOR/CANDIDATE_MARGIN_FLOOR: added 2026-09-21 after a
    real investigation into whether Face.det_score helps classification.
    Combined with a tightened per-candidate margin requirement (sim_99th
    at least CANDIDATE_MARGIN_FLOOR above threshold, not just > 0), this
    is a validated fix for barely-passing candidates on poorly-detected
    faces -- see CLAUDE.md for the full investigation, including a real
    visual false positive (a photo of toy building blocks) that this
    combined gate catches but det_score alone did not."""

    def _make_face_with_single_candidate(self, similarity, gallery_size=600, det_score=None, relative_size=None):
        assigner = faceAssigner()
        person = make_person(f"Candidate {similarity}-{gallery_size}-{det_score}-{relative_size}")
        assigner.likely_people_ids = [person.id]

        query = np.zeros(512)
        query[0] = 1.0
        ref = query.copy()
        ref[0] = similarity
        ref[1] = np.sqrt(max(0.0, 1 - similarity ** 2))

        assigner.embedding_dict = {person.id: np.tile(ref.reshape(512, 1), (1, gallery_size))}
        assigner.norm_dict = {person.id: np.ones(gallery_size)}
        assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        face = make_face(img, declared_name=blank_person)
        face.face_encoding_512 = query.tolist()
        face.det_score = det_score
        face.relative_size = relative_size
        face.save()
        return assigner, face, person

    def test_margin_just_above_zero_now_rejected(self):
        # Previously accepted (margin > 0 was the whole gate); the
        # tightened CANDIDATE_MARGIN_FLOOR (0.03) now rejects a candidate
        # that only barely clears its threshold.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.01  # margin=0.01, below CANDIDATE_MARGIN_FLOOR=0.03
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        # Falls through to the "no match" branch -- proposed person is
        # the ignore sentinel, not the barely-passing candidate.
        self.assertEqual(called_person_id, assigner.ignore_person_id)

    def test_margin_at_floor_with_good_det_score_is_accepted(self):
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + assigner.CANDIDATE_MARGIN_FLOOR + 0.01
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, person.id)

    def test_good_margin_but_low_det_score_is_rejected(self):
        # A candidate that clears the margin requirement comfortably, but
        # on a poorly-detected face (det_score below DET_SCORE_FLOOR) --
        # the det_score gate blocks it even though margin alone would
        # have accepted it. Mirrors the real "toy building blocks" false
        # positive found during the investigation.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.20  # comfortable margin
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.4,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        called_person_id = mock_set.call_args[0][0]
        self.assertEqual(called_person_id, assigner.ignore_person_id)

    def test_null_det_score_skips_det_score_check_but_not_margin(self):
        # A face not yet reached by the ongoing det_score backfill (NULL)
        # must not be blocked by the det_score half of the gate -- but
        # the margin requirement still applies unconditionally.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]

        # Comfortable margin, NULL det_score -- should still be accepted.
        similarity = threshold + 0.20
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=None,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

        # Thin margin (0.01), NULL det_score -- margin gate still rejects
        # it regardless of the missing det_score.
        similarity2 = threshold + 0.01
        assigner2, face2, person2 = self._make_face_with_single_candidate(
            similarity=similarity2, det_score=None,
        )
        with patch.object(Face, "set_possible_person") as mock_set2:
            assigner2.classify_unassigned(face2)
        mock_set2.assert_called_once()
        self.assertEqual(mock_set2.call_args[0][0], assigner2.ignore_person_id)

    def _make_video_face_with_single_candidate(self, similarity, gallery_size=600, det_score=None):
        # Same shape as _make_face_with_single_candidate, but the query
        # face is video-sourced -- needed to exercise VIDEO_DET_SCORE_FLOOR
        # and the video sub-gallery boost, neither of which a plain
        # image-sourced query ever touches.
        from filepopulator.models import Directory, VideoFile

        assigner = faceAssigner()
        person = make_person(f"VideoCandidate {similarity}-{gallery_size}-{det_score}")
        assigner.likely_people_ids = [person.id]

        query = np.zeros(512)
        query[0] = 1.0
        ref = query.copy()
        ref[0] = similarity
        ref[1] = np.sqrt(max(0.0, 1 - similarity ** 2))

        assigner.embedding_dict = {person.id: np.tile(ref.reshape(512, 1), (1, gallery_size))}
        assigner.norm_dict = {person.id: np.ones(gallery_size)}
        assigner._build_concatenated_gallery()

        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/video_classify_test')
        video = VideoFile.objects.create(
            filename='/tmp/video_classify_test/a.mp4', directory=directory,
            width=320, height=240, duration_seconds=10.0,
        )
        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        face = Face(
            declared_name=blank_person, source_video_file=video,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.face_encoding_512 = query.tolist()
        face.det_score = det_score
        face.save()
        return assigner, face, person

    def test_video_query_uses_video_det_score_floor_not_image_floor(self):
        # det_score=0.55 sits between VIDEO_DET_SCORE_FLOOR (0.5, passes)
        # and the image-calibrated DET_SCORE_FLOOR (0.6, would fail) --
        # must be accepted for a video query specifically because the
        # lower, video-calibrated floor applies.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.20
        assigner, face, person = self._make_video_face_with_single_candidate(
            similarity=similarity, det_score=0.55,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_image_query_at_same_det_score_still_uses_image_floor(self):
        # Sanity check: the exact same det_score (0.55) on an IMAGE query
        # still gets rejected -- confirms this is a video-specific floor,
        # not an accidental global loosening.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.20
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.55,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], assigner.ignore_person_id)

    def test_video_subgallery_boosts_a_candidate_above_threshold_for_video_query(self):
        # Full-gallery similarity sits below the margin floor on its own;
        # the SAME candidate's video-only sub-gallery scores it high
        # enough to clear -- a video query should be accepted via that
        # boost. CANDIDATE_MARGIN_FLOOR/DET_SCORE_FLOOR values are
        # deliberately generous here (det_score=0.9, well above either
        # floor) to isolate the sub-gallery boost itself.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        below_margin = threshold + 0.01  # fails CANDIDATE_MARGIN_FLOOR on its own
        above_margin = threshold + 0.20

        assigner, face, person = self._make_video_face_with_single_candidate(
            similarity=below_margin, det_score=0.9,
        )

        query = np.array(face.face_encoding_512)
        video_ref = query.copy()
        video_ref[0] = above_margin
        video_ref[1] = np.sqrt(max(0.0, 1 - above_margin ** 2))
        assigner.video_embedding_dict = {person.id: np.tile(video_ref.reshape(512, 1), (1, assigner.MIN_VIDEO_SUBGALLERY_SIZE))}
        assigner.video_norm_dict = {person.id: np.ones(assigner.MIN_VIDEO_SUBGALLERY_SIZE)}
        assigner._build_video_concatenated_gallery()

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_video_subgallery_does_not_affect_image_query(self):
        # Same boosted video sub-gallery as above, but the query itself
        # is image-sourced -- must be completely ignored, keeping the
        # already-tuned image classification path untouched.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        below_margin = threshold + 0.01
        above_margin = threshold + 0.20

        assigner, face, person = self._make_face_with_single_candidate(
            similarity=below_margin, det_score=0.9,
        )

        query = np.array(face.face_encoding_512)
        video_ref = query.copy()
        video_ref[0] = above_margin
        video_ref[1] = np.sqrt(max(0.0, 1 - above_margin ** 2))
        assigner.video_embedding_dict = {person.id: np.tile(video_ref.reshape(512, 1), (1, assigner.MIN_VIDEO_SUBGALLERY_SIZE))}
        assigner.video_norm_dict = {person.id: np.ones(assigner.MIN_VIDEO_SUBGALLERY_SIZE)}
        assigner._build_video_concatenated_gallery()

        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], assigner.ignore_person_id)


class SizeNudgeMarginGateTests(TestCase):
    """SIZE_NUDGE_REF_SIZE/SIZE_NUDGE_MAX: added 2026-10-20 after a real
    investigation into whether Face.relative_size helps classification --
    see CLAUDE.md for the full sweep (continuous linear decay vs. tiers,
    and why lowering CANDIDATE_MARGIN_FLOOR itself doesn't help even with
    this term in place). Uses a standalone copy of DetScoreMarginGateTests'
    _make_face_with_single_candidate helper (not a subclass -- subclassing
    a TestCase to reuse a helper also re-runs every inherited test method
    under this class name too) -- det_score is always a comfortable 0.9,
    well above DET_SCORE_FLOOR, isolating the size term specifically."""

    def _make_face_with_single_candidate(self, similarity, gallery_size=600, det_score=None, relative_size=None):
        assigner = faceAssigner()
        person = make_person(f"SizeCandidate {similarity}-{gallery_size}-{det_score}-{relative_size}")
        assigner.likely_people_ids = [person.id]

        query = np.zeros(512)
        query[0] = 1.0
        ref = query.copy()
        ref[0] = similarity
        ref[1] = np.sqrt(max(0.0, 1 - similarity ** 2))

        assigner.embedding_dict = {person.id: np.tile(ref.reshape(512, 1), (1, gallery_size))}
        assigner.norm_dict = {person.id: np.ones(gallery_size)}
        assigner._build_concatenated_gallery()

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        face = make_face(img, declared_name=blank_person)
        face.face_encoding_512 = query.tolist()
        face.det_score = det_score
        face.relative_size = relative_size
        face.save()
        return assigner, face, person

    def test_small_face_with_thin_margin_is_rejected(self):
        # A margin that would clear the plain CANDIDATE_MARGIN_FLOOR
        # (0.03) on its own, but not once the small-face size nudge
        # (relative_size well below SIZE_NUDGE_REF_SIZE=0.04) adds its
        # full SIZE_NUDGE_MAX=0.02 on top -- effective floor 0.05.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.04  # margin=0.04: clears 0.03, not 0.05
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.005,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], assigner.ignore_person_id)

    def test_small_face_with_margin_clearing_the_nudged_floor_is_accepted(self):
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.06  # margin=0.06, clears 0.03+0.02=0.05
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.005,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_face_at_or_above_ref_size_gets_no_nudge(self):
        # relative_size >= SIZE_NUDGE_REF_SIZE (0.04) -- same plain 0.03
        # floor as before this feature existed.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.04  # margin=0.04, clears the plain 0.03 floor
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.04,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_null_relative_size_skips_the_size_nudge(self):
        # Not yet reached by a relative_size backfill (or a video-sourced
        # query) -- must not be penalized, same convention as a NULL
        # det_score skipping its own half of the gate.
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.04  # margin=0.04: would fail a nudged floor, passes the plain one
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=None,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

    def test_nudge_decays_linearly_partway_to_ref_size(self):
        # relative_size halfway between 0 and SIZE_NUDGE_REF_SIZE (0.02 of
        # 0.04) -- nudge should be half of SIZE_NUDGE_MAX (0.01), so
        # margin=0.035 (clears 0.03+0.01=0.04) should be accepted...
        assigner = faceAssigner()
        threshold = assigner.BUCKET_THRESHOLDS[-1]
        similarity = threshold + 0.045
        assigner, face, person = self._make_face_with_single_candidate(
            similarity=similarity, det_score=0.9, relative_size=0.02,
        )
        with patch.object(Face, "set_possible_person") as mock_set:
            assigner.classify_unassigned(face)
        mock_set.assert_called_once()
        self.assertEqual(mock_set.call_args[0][0], person.id)

        # ...but margin=0.035 (just under 0.03+0.01=0.04) should not.
        assigner2 = faceAssigner()
        similarity2 = threshold + 0.035
        assigner2, face2, person2 = self._make_face_with_single_candidate(
            similarity=similarity2, det_score=0.9, relative_size=0.02,
        )
        with patch.object(Face, "set_possible_person") as mock_set2:
            assigner2.classify_unassigned(face2)
        mock_set2.assert_called_once()
        self.assertEqual(mock_set2.call_args[0][0], assigner2.ignore_person_id)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClassificationInvariantCheckTests(TestCase):
    """faceAssigner._verify_classification_invariant(): a safety net run
    at the end of every execute() call -- a face with a real encoding
    and no poss_ident1 proposal shouldn't exist after a normal pass
    (every real classify_unassigned() success path sets something).
    classify_unassigned() itself is mocked throughout -- this tests the
    invariant-check/retry/log_error wiring, not real classification
    math (already covered elsewhere)."""

    def _make_leftover_face(self):
        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        img = make_image()
        face = make_face(img, declared_name=blank_person)
        face.face_encoding_512 = (np.ones(512) / np.sqrt(512)).tolist()
        face.poss_ident1 = None
        face.save()
        return face

    def test_no_leftover_faces_is_a_no_op(self):
        from common.models import ApplicationError
        assigner = faceAssigner()
        with patch.object(faceAssigner, 'classify_unassigned') as mock_classify:
            assigner._verify_classification_invariant()
        mock_classify.assert_not_called()
        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_leftover_face_that_succeeds_on_retry_is_not_logged(self):
        from common.models import ApplicationError
        face = self._make_leftover_face()
        person = make_person("Retried Match")
        assigner = faceAssigner()

        def fake_classify(f, **kwargs):
            # Simulate a successful classification on retry.
            f.poss_ident1 = person
            f.save()

        with patch.object(faceAssigner, 'classify_unassigned', side_effect=fake_classify):
            assigner._verify_classification_invariant()

        face.refresh_from_db()
        self.assertEqual(face.poss_ident1_id, person.id)
        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_leftover_face_that_keeps_failing_is_logged(self):
        from common.models import ApplicationError
        face = self._make_leftover_face()
        assigner = faceAssigner()

        with patch.object(faceAssigner, 'classify_unassigned', side_effect=RuntimeError("boom")):
            assigner._verify_classification_invariant()

        self.assertEqual(ApplicationError.objects.count(), 1)
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.assign_faces.execute')
        self.assertIn(str(face.id), error.message)
        self.assertFalse(error.dismissed)

    def test_newly_created_face_during_retry_window_is_not_falsely_flagged(self):
        # Real production false-positive (2026-10-21): a face created
        # concurrently by face_extraction, in the window between
        # building leftover_ids and the still_broken re-check, must not
        # be reported as "still broken after a retry" -- it was never
        # part of the retry batch at all, just an ordinary new,
        # not-yet-classified face.
        from common.models import ApplicationError
        face = self._make_leftover_face()
        assigner = faceAssigner()

        def fake_classify(f, **kwargs):
            person = make_person("Retried Match For Race Test")
            f.poss_ident1 = person
            f.save()
            # A brand-new face lands concurrently, created AFTER
            # leftover_ids was already built -- never actually retried.
            self._make_leftover_face()

        with patch.object(faceAssigner, 'classify_unassigned', side_effect=fake_classify):
            assigner._verify_classification_invariant()

        self.assertEqual(ApplicationError.objects.count(), 0)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class VerificationClusterGroupTests(TestCase):
    """Face.verification_cluster_group: nightly per-person complete-linkage
    clustering of confirmed-but-unverified faces, so a human reviewer can
    spot-check a whole visually-coherent group at once. See
    verification_clustering.py / CLAUDE.md for the full investigation
    behind why complete linkage, per-person-only, was chosen."""

    def setUp(self):
        self.person = make_person("Cluster Person")
        self.other_person = make_person("Other Cluster Person")
        self.img = make_image()

    def _make_unverified_face(self, person, embedding, **overrides):
        overrides.setdefault('validated', False)
        return make_face(
            self.img, declared_name=person,
            face_encoding_512=embedding.tolist(), **overrides,
        )

    def test_two_tight_clusters_get_distinct_group_ids_and_singleton_stays_null(self):
        cluster_a = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(4)]
        cluster_b = [self._make_unverified_face(self.person, _embedding(1, seed=100 + i)) for i in range(3)]
        singleton = self._make_unverified_face(self.person, _embedding(2, seed=200))

        cluster_all_unverified_faces()

        groups_a = {Face.objects.get(pk=f.pk).verification_cluster_group for f in cluster_a}
        groups_b = {Face.objects.get(pk=f.pk).verification_cluster_group for f in cluster_b}
        self.assertEqual(len(groups_a), 1)
        self.assertEqual(len(groups_b), 1)
        self.assertNotIn(None, groups_a)
        self.assertNotIn(None, groups_b)
        self.assertNotEqual(groups_a, groups_b)
        self.assertIsNone(Face.objects.get(pk=singleton.pk).verification_cluster_group)

    def test_group_ids_are_independent_per_person(self):
        # Both people's tight clusters should each land on group id 0 --
        # ids are 0-indexed PER PERSON, not globally unique.
        cluster_a = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_b = [self._make_unverified_face(self.other_person, _embedding(0, seed=50 + i)) for i in range(3)]

        cluster_all_unverified_faces()

        self.assertEqual(Face.objects.get(pk=cluster_a[0].pk).verification_cluster_group, 0)
        self.assertEqual(Face.objects.get(pk=cluster_b[0].pk).verification_cluster_group, 0)

    def test_validated_faces_are_excluded(self):
        already_verified = self._make_unverified_face(self.person, _embedding(0, seed=1), validated=True)
        unverified = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(2, 4)]

        cluster_all_unverified_faces()

        self.assertIsNone(Face.objects.get(pk=already_verified.pk).verification_cluster_group)
        self.assertIsNotNone(Face.objects.get(pk=unverified[0].pk).verification_cluster_group)

    def test_ignored_sentinel_person_faces_are_excluded(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        faces = [self._make_unverified_face(ignore_person, _embedding(0, seed=i)) for i in range(3)]

        cluster_all_unverified_faces()

        for f in faces:
            self.assertIsNone(Face.objects.get(pk=f.pk).verification_cluster_group)

    def test_null_and_sentinel_encodings_are_excluded(self):
        f_null = self._make_unverified_face(self.person, _embedding(0, seed=1))
        f_null.face_encoding_512 = None
        f_null.save()

        f_sentinel = self._make_unverified_face(self.person, _embedding(0, seed=2))
        f_sentinel.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        f_sentinel.save()

        f_real_a = self._make_unverified_face(self.person, _embedding(0, seed=3))
        f_real_b = self._make_unverified_face(self.person, _embedding(0, seed=4))

        cluster_all_unverified_faces()

        self.assertIsNone(Face.objects.get(pk=f_null.pk).verification_cluster_group)
        self.assertIsNone(Face.objects.get(pk=f_sentinel.pk).verification_cluster_group)
        self.assertEqual(
            Face.objects.get(pk=f_real_a.pk).verification_cluster_group,
            Face.objects.get(pk=f_real_b.pk).verification_cluster_group,
        )
        self.assertIsNotNone(Face.objects.get(pk=f_real_a.pk).verification_cluster_group)

    def test_nightly_rebuild_clears_stale_groups_not_reproduced_this_run(self):
        # A face that was grouped by a previous run but is no longer
        # eligible (e.g. since verified) must not be left with a stale
        # group id -- the nightly rebuild clears ALL faces first.
        stale = self._make_unverified_face(self.person, _embedding(0, seed=1))
        stale.verification_cluster_group = 7
        stale.validated = True
        stale.save()

        cluster_all_unverified_faces()

        self.assertIsNone(Face.objects.get(pk=stale.pk).verification_cluster_group)

    def test_associate_person_clears_group_immediately(self):
        faces = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_all_unverified_faces()
        self.assertIsNotNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)

        other = make_person("Reassign Target")
        face = Face.objects.get(pk=faces[0].pk)
        face.associate_person(other.id)

        self.assertIsNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)

    def test_verify_person_in_image_clears_group_immediately(self):
        faces = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_all_unverified_faces()

        face = Face.objects.get(pk=faces[0].pk)
        face.verify_person_in_image()

        self.assertIsNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)

    def test_reset_to_pool_clears_group_immediately(self):
        faces = [self._make_unverified_face(self.person, _embedding(0, seed=i)) for i in range(3)]
        cluster_all_unverified_faces()

        face = Face.objects.get(pk=faces[0].pk)
        face.reset_to_pool()

        self.assertIsNone(Face.objects.get(pk=faces[0].pk).verification_cluster_group)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class LoadEncodingsCachingTests(TestCase):
    """load_encodings()'s persistent cache: reused as-is within the same
    day, reused across a day boundary if nothing changed, and fully
    rebuilt across a day boundary if something did -- see the method's
    own docstring for the full rationale."""

    def setUp(self):
        self.cache_path = "/tmp/face_manager_test_media/test_encodings_cache.pkl"
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

        self.assigner = faceAssigner()
        self.assigner.ENCODINGS_PKL_FILE = self.cache_path
        self.person = make_person("Cached Person")
        self.assigner.likely_people_ids = [self.person.id]

        img = make_image()
        self.face = make_face(img, declared_name=self.person)
        self.face.face_encoding_512 = ([0.1] * 512)
        self.face.save()

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def test_builds_and_persists_cache_on_first_call(self):
        self.assigner.load_encodings()
        self.assertTrue(os.path.exists(self.cache_path))
        self.assertIn(self.person.id, self.assigner.embedding_dict)

    def test_same_day_reuses_cache_without_requerying(self):
        self.assigner.load_encodings()

        with patch.object(faceAssigner, "_current_face_data_signature") as mock_sig:
            fresh_assigner = faceAssigner()
            fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
            fresh_assigner.likely_people_ids = [self.person.id]
            fresh_assigner.load_encodings()

        mock_sig.assert_not_called()
        self.assertIn(self.person.id, fresh_assigner.embedding_dict)

    def test_within_max_age_reuses_cache_without_requerying(self):
        # A cache built (CACHE_MAX_AGE_DAYS - 1) days ago is still fully
        # inside the trusted window -- not even the cheap signature check
        # should run, let alone a full re-fetch.
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(
            days=self.assigner.CACHE_MAX_AGE_DAYS - 1
        )
        with open(self.cache_path, "wb") as fh:
            pickle.dump(cached, fh)

        fresh_assigner = faceAssigner()
        fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
        fresh_assigner.likely_people_ids = [self.person.id]
        with patch.object(Face.objects, "filter", wraps=Face.objects.filter) as mock_filter, \
             patch.object(faceAssigner, "_current_face_data_signature") as mock_sig:
            fresh_assigner.load_encodings()
        mock_sig.assert_not_called()
        self.assertEqual(mock_filter.call_count, 0)
        self.assertIn(self.person.id, fresh_assigner.embedding_dict)

    def test_past_max_age_no_changes_keeps_cache(self):
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(
            days=self.assigner.CACHE_MAX_AGE_DAYS
        )
        with open(self.cache_path, "wb") as fh:
            pickle.dump(cached, fh)

        fresh_assigner = faceAssigner()
        fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
        fresh_assigner.likely_people_ids = [self.person.id]
        with patch.object(Face.objects, "filter", wraps=Face.objects.filter) as mock_filter:
            fresh_assigner.load_encodings()
        # The only DB access should be the cheap signature count query --
        # not a full re-fetch of this person's face rows.
        self.assertEqual(mock_filter.call_count, 1)
        self.assertIn(self.person.id, fresh_assigner.embedding_dict)

    def test_past_max_age_with_changes_rebuilds_cache(self):
        self.assigner.load_encodings()

        with open(self.cache_path, "rb") as fh:
            import pickle
            cached = pickle.load(fh)
        cached["built_date"] = cached["built_date"] - __import__("datetime").timedelta(
            days=self.assigner.CACHE_MAX_AGE_DAYS
        )
        cached["signature"] = -1  # force a mismatch against the real current count
        with open(self.cache_path, "wb") as fh:
            pickle.dump(cached, fh)

        # Add a second confirmed face for this person -- a real change
        # the stale cache doesn't know about.
        img2 = make_image()
        second_face = make_face(img2, declared_name=self.person)
        second_face.face_encoding_512 = [0.2] * 512
        second_face.save()

        fresh_assigner = faceAssigner()
        fresh_assigner.ENCODINGS_PKL_FILE = self.cache_path
        fresh_assigner.likely_people_ids = [self.person.id]
        fresh_assigner.load_encodings()

        self.assertEqual(len(fresh_assigner.candidate_dict[self.person.id]), 2)


