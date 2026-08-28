#! /usr/bin/env python

# from django.core.management.base import BaseCommand
# from scipy import stats
# from time import sleep
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# import collections
# import io
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional
# import torchvision
# torch.backends.nnpack.enabled = False
from datetime import datetime
from django.conf import settings
from django.db.models import Count, Q, F
from django.db.models.functions import Abs
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import pickle
import time


class faceAssigner():
    """
    docstring for faceAssigner
    """

    def __init__(self, debug: bool = False):
        super(faceAssigner, self).__init__()

        self.DEBUG=debug
        self.ENCODINGS_PKL_FILE='/models/face_assign_preload.pkl'
        self.USE_MIN_VALUE=True
        if self.USE_MIN_VALUE:
            self.IGN_VALUE = 999
        else:
            self.IGN_VALUE = 0

        self.MIN_NUM_FACES = 10
        self.NUM_DAYS = 180
        self.NUM_CLOSEST = 50
        self.NUM_TO_AVERAGE = 1
        self.N_COMPARISONS = 25

        # Gallery-size-adaptive accept/reject gate, replacing the old flat
        # ASSIGN_THRESH=0.6 applied to sim_max. Root cause: gating on a
        # single nearest-neighbor (sim_max) is maximally vulnerable to any
        # one noisy/mislabeled face in a person's gallery; switching to a
        # percentile (sim_99th, already computed below) helps, but a FIXED
        # percentile gets silently stricter as a gallery grows (rank scales
        # with sample size), crushing TPR specifically for the
        # most-photographed people -- exactly backwards, since those are
        # the people future photos keep landing on. Bucketing by gallery
        # size and calibrating sim_99th's threshold per bucket fixes that
        # unevenness. Boundaries/thresholds below are empirically derived
        # (not guessed) -- see CLAUDE.md's "Face-classification
        # outlier-rejection" section for the full experiment writeup
        # (441-person/273k-face gallery, 3000 negative + 15,621 positive
        # holdout queries). BUCKET_BOUNDARIES are upper-exclusive edges;
        # BUCKET_THRESHOLDS has one more entry than BUCKET_BOUNDARIES (the
        # last covers everyone at/above the final boundary).
        self.BUCKET_BOUNDARIES = [50, 200, 500]
        self.BUCKET_THRESHOLDS = [0.558, 0.551, 0.486, 0.394]

        # How far below its own bucket threshold a face's closest-call
        # candidate needs to be before it's treated as "confidently far
        # from everyone" for ignore-weight purposes (see the "no match"
        # branch of classify_unassigned() below) -- beyond this margin,
        # further distance doesn't make it any more clearly not-a-match.
        self.IGNORE_WEIGHT_MARGIN_CLAMP = 0.3

        # self.bogus_date = datetime(1990, 1, 1) # Very few images before that
        # self.bogus_date_utc = time.mktime(self.bogus_date.timetuple())
        self.ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME)[0]
        self.ignore_person_id = self.ignore_person.id

        ########################################################
        # Get a list of likely matches for the faces we have - people with
        # a minimum number of assigned faces. 
        ########################################################

        criterion_ign = ~Q(person_name__in=settings.IGNORED_NAMES)
        criterion_unlikely = Q(further_images_unlikely=False)

        # assigned_people = Person.objects.annotate(c=Count('face_declared', filter=criterion_ign & criterion_unlikely)).filter(c__gt=self.MIN_NUM_FACES)
        assigned_people = Person.objects.annotate(c=Count('face_declared', filter=criterion_ign )).filter(c__gt=self.MIN_NUM_FACES)

        self.likely_people_ids = [p.id for p in assigned_people]
        self.num_likely_people = len(self.likely_people_ids)

        ########################################################
        # Map people to the likely earliest date they showed up in images,
        # using some outlier statistics.
        ########################################################
        # self.known_persons_to_dates()

    # def reset_task(self):
    #     people = Person.objects.all()
    #     for p in people:
    #         p.num_faces = p.face_declared.count()
    #         p.num_possibilities = p.face_poss1.count() # + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
    #         p.num_unverified_faces = p.face_declared.filter(validated=False).count()
    #         p.save()

    def _p99_threshold_for_gallery_size(self, gallery_size: int) -> float:
        """Which bucket's calibrated sim_99th threshold applies to a
        candidate with this many confirmed faces. See BUCKET_BOUNDARIES/
        BUCKET_THRESHOLDS above for where these numbers come from."""
        for boundary, threshold in zip(self.BUCKET_BOUNDARIES, self.BUCKET_THRESHOLDS[:-1]):
            if gallery_size < boundary:
                return threshold
        return self.BUCKET_THRESHOLDS[-1]

    def _current_face_data_signature(self) -> int:
        """Cheap proxy for 'has anything changed since the cache was last
        built' -- total count of qualifying faces across all likely
        people. Not perfect (a same-day swap of one face for another
        wouldn't change the count), but Face has no modification
        timestamp to check against, and this is a single aggregate query
        rather than per-person work, cheap enough to run on every
        load_encodings() call."""
        has_long = ~Q(face_encoding_512=None)
        long_encoded = ~Q(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)
        return Face.objects.filter(
            Q(declared_name_id__in=self.likely_people_ids) & has_long & long_encoded
        ).count()

    def reset_possible_assignments(self):
        Person.objects.all().update(num_possibilities = 0)
        Face.objects.filter(~Q(poss_ident1=None)).update(poss_ident1 = None)
        Face.objects.filter(~Q(poss_ident2=None)).update(poss_ident2 = None)
        Face.objects.filter(~Q(poss_ident3=None)).update(poss_ident3 = None)
        Face.objects.filter(~Q(poss_ident4=None)).update(poss_ident4 = None)
        Face.objects.filter(~Q(poss_ident5=None)).update(poss_ident5 = None)
        Face.objects.filter(~Q(weight_1=0.0)).update(weight_1 = 0.0)
        Face.objects.filter(~Q(weight_2=0.0)).update(weight_2 = 0.0)
        Face.objects.filter(~Q(weight_3=0.0)).update(weight_3 = 0.0)
        Face.objects.filter(~Q(weight_4=0.0)).update(weight_4 = 0.0)
        Face.objects.filter(~Q(weight_5=0.0)).update(weight_5 = 0.0)
        Face.objects.filter(Q(declared_name__person_name=settings.BLANK_FACE_NAME)).update(rejected_fields = None)

    def load_encodings(self, reload_pkl_file: bool = False):
        """Loads (and persistently caches to self.ENCODINGS_PKL_FILE) the
        per-person embedding/norm data classify_unassigned() compares
        against, so a normal run doesn't re-fetch and re-vectorize
        hundreds of thousands of Face rows from the DB every time this
        task fires (currently hourly).

        Cache lifecycle: within the same calendar day, the cache is
        trusted as-is and only topped up with any brand-new likely-people
        not seen before (people freshly crossing MIN_NUM_FACES). Once a
        new day has begun since the cache was last built, do one cheap
        signature check (_current_face_data_signature()) -- if nothing
        has actually changed (no faces (re)assigned to any likely person
        since the last build), keep using the existing cache untouched
        rather than pay for a rebuild nobody needs; if something did
        change, rebuild the whole cache from scratch (not just top up),
        since an existing person's gallery may have been added to or
        corrected, not just brand-new people appearing. This naturally
        catches "once a day, if there was activity that day" without a
        separate scheduled invalidation task -- it just piggybacks on
        whichever run happens first after the day rolls over.
        """

        ss = time.time()

        self.candidate_dict = {}
        self.embedding_dict = {}
        self.norm_dict = {}

        cache_is_valid = False
        if os.path.exists(self.ENCODINGS_PKL_FILE) and not reload_pkl_file:
            with open(self.ENCODINGS_PKL_FILE, 'rb') as ph:
                combo_dict = pickle.load(ph)

            cached_date = combo_dict.get('built_date')
            today = datetime.now().date()

            if cached_date == today:
                cache_is_valid = True
            else:
                cached_signature = combo_dict.get('signature')
                current_signature = self._current_face_data_signature()
                if current_signature == cached_signature:
                    print("Daily cache check: no face changes since last build -- keeping existing cache.")
                    cache_is_valid = True
                else:
                    print("Daily cache check: face data changed since last build -- rebuilding full cache.")

            if cache_is_valid:
                self.candidate_dict = combo_dict['candidate_dict']
                self.embedding_dict = combo_dict['embedding_dict']
                self.norm_dict = combo_dict['norm_dict']

        changed = not cache_is_valid

        for face_id in tqdm(self.likely_people_ids):

            if face_id in self.candidate_dict.keys() and \
               face_id in self.embedding_dict.keys() and \
               face_id in self.norm_dict.keys():
                # print(f'No need to process this face {face_id}')
                continue
            # print(f"Processing face {face_id}")
            changed = True

            faces_person = Q(declared_name__id=face_id)
            has_long = ~Q(face_encoding_512=None)
            long_encoded = ~Q(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)

            person_data = Face.objects \
                .filter(faces_person & long_encoded & has_long)
            data = person_data.values_list('id', 'face_encoding_512', 'dateTakenUTC')

            df = pd.DataFrame(data, columns=['id', 'face_encoding_512', 'dateTakenUTC'])

            self.candidate_dict[face_id] = df

            cmp_embedding = np.array(self.candidate_dict[face_id]['face_encoding_512'].tolist())
            norm_list = np.linalg.norm(cmp_embedding, axis=1)
            self.embedding_dict[face_id] = cmp_embedding.T
            assert self.embedding_dict[face_id].shape[0] == 512
            assert self.embedding_dict[face_id].shape[1] == len(norm_list)
            assert len(norm_list) == len(self.candidate_dict[face_id])
            self.norm_dict[face_id] = norm_list

        if changed:
            all_dict = {'candidate_dict': self.candidate_dict,
                        'embedding_dict': self.embedding_dict,
                        'norm_dict': self.norm_dict,
                        'built_date': datetime.now().date(),
                        'signature': self._current_face_data_signature()}
            try:
                with open(self.ENCODINGS_PKL_FILE, 'wb') as ph:
                    pickle.dump(all_dict, ph)
            except:
                os.remove(self.ENCODINGS_PKL_FILE)

        self._build_concatenated_gallery()

        print(f"Dataframe preloading: {time.time() - ss:.2f} seconds")

    def _build_concatenated_gallery(self):
        """Concatenates every likely person's embedding matrix into one
        big (512 x N) array, with an offsets map recording which column
        range belongs to which person_id. classify_unassigned() then does
        ONE matmul against this whole array per query instead of looping
        over each of the ~441 candidates doing its own small np.dot() --
        the same technique (and the same real speedup) validated
        experimentally for the outlier-rejection analysis this design is
        based on; see CLAUDE.md. Cheap to rebuild (just concatenation, no
        DB access) every load_encodings() call, so it doesn't need its
        own cache-invalidation logic."""
        person_ids = list(self.embedding_dict.keys())
        self.gallery_offsets = {}
        cursor = 0
        embedding_chunks = []
        norm_chunks = []
        for person_id in person_ids:
            embeddings = self.embedding_dict[person_id]
            n = embeddings.shape[1]
            self.gallery_offsets[person_id] = (cursor, cursor + n)
            embedding_chunks.append(embeddings)
            norm_chunks.append(self.norm_dict[person_id])
            cursor += n

        if embedding_chunks:
            self.all_embeddings = np.concatenate(embedding_chunks, axis=1)
            self.all_norms = np.concatenate(norm_chunks)
        else:
            self.all_embeddings = np.zeros((512, 0))
            self.all_norms = np.zeros(0)

    def execute(self, redo_all: bool = False) -> None:
        """
        DOCSTRING
        """

        if type(redo_all) != bool:
            raise TypeError(f"Type of redo_all must be boolean, is {type(redo_all)}.")

        unassigned_crit = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        has_long = ~Q(face_encoding_512=None)
        long_encoded = ~Q(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)

        # Get all of the images that have no declared name. If redo_all is True,
        # then we do all the faces, otherwise just the ones that don't have an
        # assignment for poss_ident1, i.e. that have been rejected in the GUI. 

        # select_related('declared_name'): classify_unassigned() checks
        # unassigned_face.declared_name.person_name on every face (the
        # concurrent-processing guard just below) -- without this, that's
        # a separate DB round trip per face on top of everything else.
        if redo_all:
            unassigned = Face.objects.filter(unassigned_crit & has_long & long_encoded).select_related('declared_name').order_by('?')
        else:
            no_suggestions = Q(poss_ident1__person_name=None)
            unassigned = Face.objects.filter(unassigned_crit & has_long & long_encoded).filter(no_suggestions).select_related('declared_name').order_by('?')

        if self.DEBUG:
            unassigned = unassigned[:1001]
        num_unassigned = int(unassigned.count())
        print(f"There are {num_unassigned} faces to classify")

        # Always load encodings, regardless of batch size. This used to be
        # skipped for small batches (num_unassigned <= 100), leaving
        # embedding_dict/norm_dict/candidate_dict unset -- classify_
        # unassigned() unconditionally reads those, so any run with a
        # small batch crashed on every single face with AttributeError
        # (confirmed live in production). The cache above makes this
        # cheap for the common case anyway (same-day reruns just load the
        # pickle, no DB hit for the embedding data itself), so there's no
        # real cost to doing it unconditionally.
        self.load_encodings()
        
        u_idx = 0
        s = time.time()
        for u_img in tqdm(unassigned.iterator()):
            elps = time.time() - s
            s = time.time()
            if self.DEBUG:
                print(f"Assigning: {u_idx+1}/{num_unassigned} | {elps:.2f}")
                u_idx += 1
            try:
                self.classify_unassigned(u_img)
            except Exception as e:
                # A failure classifying one face must not abort the entire
                # scheduled run -- every other already-queued face in this
                # batch would otherwise silently never get classified
                # either. See classify_unassigned()'s array-sizing fix
                # above for the bug this specifically guards against.
                print(f"Exception classifying face {u_img.id}: {e}")
                settings.LOGGER.error(f"Exception classifying face {u_img.id}: {e}")

        # Finish up by "trueing up" the num_assigned for each person:
        print("Verifying face counts...")
        for p in tqdm(Person.objects.all()):
            p.num_faces = p.face_declared.count()
            p.num_possibilities = p.face_poss1.count() # + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
            p.num_unverified_faces = p.face_declared.filter(validated=False).count()
            p.save()




    def classify_unassigned(self, unassigned_face: Face, debug_face_id = None) -> None:
        """
        DOCSTRING
        """
        if self.DEBUG:
            print("=" * 80)
            print("classify ", unassigned_face.id)
            
        if type(unassigned_face) != Face:
            raise TypeError(f"Type of object passed to self.classify_unassigned was {type(unassigned_face)}, should be {Face}")

        # This is to handle cases where multiple concurrent processes 
        # may be working simultaneously. 
        if unassigned_face.declared_name.person_name != settings.BLANK_FACE_NAME:
            print("Already assigned")
            return

        # (Used to compute date_taken/date_string here from
        # unassigned_face.source_image_file.dateTaken for a commented-out
        # debug print -- removed. It was otherwise unused, but triggered
        # a real per-face DB query for the source_image_file FK, since
        # execute()'s queryset doesn't select_related it.)

        query_encoding = np.array(unassigned_face.face_encoding_512)
        query_encoding_norm = np.linalg.norm(query_encoding)
        # comparison_mat = np.ones((len(self.likely_people_ids), self.N_COMPARISONS)) * 999

        # Figure out if any possible people have already been rejected as possible
        # candidates. Their IDs should be excluded. 
        if unassigned_face.rejected_fields is not None:
            rejected_ids = unassigned_face.rejected_fields
        else:
            rejected_ids = []

        candidate_ids = list(set(self.likely_people_ids) - set(rejected_ids))

        if len(candidate_ids) == 0:
            # Every currently-"likely" person has already been rejected as
            # a candidate for this face -- there's nothing left to compare
            # against. Assign straight to the soft-ignore person rather
            # than running similarity math over zero candidates: with the
            # array now sized to len(candidate_ids) below (not the fixed
            # self.num_likely_people), a size-0 array here would make
            # np.max()/np.argmax() in the "no match" branch below raise
            # ValueError on an empty reduction.
            unassigned_face.set_possible_person(self.ignore_person_id, 1, 1.0)
            return

        candidate_id_arr = np.array(candidate_ids)

        # Pre-populate a metrics array. Bug fix: this used to be sized to
        # self.num_likely_people (the *full* candidate roster) rather than
        # len(candidate_ids) (this face's roster minus whatever's already
        # been rejected for it) -- whenever a rejection had shrunk the
        # candidate list, the array's tail rows were left as stale
        # np.zeros() padding representing no real person. That padding
        # could pollute np.max()/np.argmax() below (a real similarity can
        # be negative, so a padded 0 can look like the best match), and
        # np.argmax() over the full-size array could return an index
        # beyond the end of the smaller candidate_id_arr, raising
        # IndexError -- which, since execute()'s per-face try/except was
        # commented out, aborted the *entire* scheduled assign_faces run,
        # not just this one face. Sizing to len(candidate_ids) keeps every
        # row real and every index in bounds.
        #
        # Columns: [sim_99th, per_candidate_threshold, db_id]. The gate
        # and ranking both use sim_99th now, not sim_max -- see
        # BUCKET_BOUNDARIES/BUCKET_THRESHOLDS above for why a flat
        # sim_max gate was replaced with a gallery-size-calibrated
        # percentile.
        metrics_array = np.zeros((len(candidate_ids), 3))

        # One big matmul against the whole gallery (all likely people,
        # not just this face's candidate_ids -- cheap either way, and
        # simpler than rebuilding a filtered matrix per face) instead of
        # a separate small np.dot() per candidate. Each candidate's
        # slice is then just cheap array indexing via gallery_offsets.
        all_similarity = (query_encoding @ self.all_embeddings) / (self.all_norms * query_encoding_norm)

        for row_num, db_id in enumerate(candidate_ids):
            lo, hi = self.gallery_offsets[db_id]
            similarity = all_similarity[lo:hi]

            sim_99th = np.percentile(similarity, 99)
            candidate_threshold = self._p99_threshold_for_gallery_size(hi - lo)

            metrics_array[row_num, :] = [sim_99th, candidate_threshold, db_id]
            if self.DEBUG and db_id == debug_face_id and debug_face_id is not None:
                print("Row values: ", metrics_array[row_num, :])

        possible_idcs = np.where(metrics_array[:, 0] > metrics_array[:, 1])[0]
        if self.DEBUG:
            print(possible_idcs, "poss len is", len(possible_idcs))
            print(metrics_array[possible_idcs, :])
        if len(possible_idcs) == 0:
            # print("TODO: Assign to ignore person")
            metric_max = np.max(metrics_array[:, 0])

            if self.DEBUG:
                row = np.argmax(metrics_array[:, 0])
                print("Metric max: ", metric_max, metrics_array[row])

            if self.ignore_person_id in rejected_ids:
                if self.DEBUG:
                    print("Ignore person is rejected")
                # print("Need to reject the person")
                max_idx = np.argmax(metrics_array[:, 0])
                max_id = candidate_id_arr[max_idx]
                unassigned_face.set_possible_person(max_id, 1, metric_max)
            else:
                # This weight answers a different question than a real
                # match's weight does: not "how confident is this specific
                # match" but "how confidently can this face be ignored."
                # The frontend sorts by weight descending, so a face that's
                # far from EVERY candidate (a safe, obvious ignore) should
                # surface first; a face that nearly cleared someone's bar
                # (a genuinely marginal near-miss, worth a human's eyes)
                # should sort toward the back -- the opposite direction
                # from a real-match weight, deliberately.
                #
                # Uses each candidate's own margin below ITS threshold,
                # not a raw score -- comparing raw sim_99th values across
                # candidates wouldn't be apples-to-apples now that
                # thresholds vary by gallery-size bucket (a large-gallery
                # candidate scoring 0.39 against a 0.394 threshold is a
                # genuine near-miss; a small-gallery candidate scoring
                # 0.35 against a 0.558 threshold isn't close at all, even
                # though the raw scores look similar).
                margins_below_threshold = metrics_array[:, 1] - metrics_array[:, 0]  # >=0 for every row here
                closest_call_margin = np.min(margins_below_threshold)  # smallest gap = nearest to passing
                clamped_margin = np.clip(closest_call_margin, 0, self.IGNORE_WEIGHT_MARGIN_CLAMP)
                weight = clamped_margin / self.IGNORE_WEIGHT_MARGIN_CLAMP  # 0 (marginal) .. 1 (far from everyone)
                unassigned_face.set_possible_person(self.ignore_person_id, 1, weight)
        else:
            scores = metrics_array[possible_idcs, 0]
            assign_ids = metrics_array[possible_idcs, 2].astype(np.int64)

            order = np.argsort(scores)[::-1]
            order = order[:5]
            # save=False + one save() at the end: up to 5 calls here (one
            # per ranked candidate), and Face.save() does real validation
            # work (~20ms measured against production) -- 5 separate
            # saves per face was a real, avoidable cost multiplier across
            # a 140k-face reprocess.
            for precedence_idx, order_idx in enumerate(order):
                # print(assign_ids[order_idx], precedence_idx, scores[order_idx])
                unassigned_face.set_possible_person(int(assign_ids[order_idx]), precedence_idx + 1, float(scores[order_idx]), save=False)
            unassigned_face.save()

