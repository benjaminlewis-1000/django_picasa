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

        self.ASSIGN_THRESH=0.6

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

        ss = time.time()

        self.candidate_dict = {}
        self.embedding_dict = {}
        self.norm_dict = {}
        if os.path.exists(self.ENCODINGS_PKL_FILE) and not reload_pkl_file:
            with open(self.ENCODINGS_PKL_FILE, 'rb') as ph:
                combo_dict = pickle.load(ph)

                self.candidate_dict = combo_dict['candidate_dict']
                self.embedding_dict = combo_dict['embedding_dict']
                self.norm_dict = combo_dict['norm_dict']
        changed = False

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
            
        all_dict = {'candidate_dict': self.candidate_dict,
                    'embedding_dict': self.embedding_dict,
                    'norm_dict': self.norm_dict}

        if changed:
            try:
                with open(self.ENCODINGS_PKL_FILE, 'wb') as ph:
                    pickle.dump(all_dict, ph)
            except:
                os.remove(self.ENCODINGS_PKL_FILE)

        print(f"Dataframe preloading: {time.time() - ss:.2f} seconds")

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

        if redo_all:
            unassigned = Face.objects.filter(unassigned_crit & has_long & long_encoded).order_by('?')
        else:
            no_suggestions = Q(poss_ident1__person_name=None)
            unassigned = Face.objects.filter(unassigned_crit & has_long & long_encoded).filter(no_suggestions).order_by('?')

        if self.DEBUG:
            unassigned = unassigned[:1001]
        num_unassigned = int(unassigned.count())
        print(f"There are {num_unassigned} faces to classify")

        if num_unassigned > 100:
            print(f"There are {num_unassigned} faces to classify.' +\
                f' We are pre-loading the database's encodings, which may take several minutes.")
            self.load_encodings()
        else:
            self.encoding_dataframe = None
        
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

        date_taken = unassigned_face.source_image_file.dateTaken.timestamp()
        # print(unassigned_face.id)
        # print(date)
        date_string = time.strftime('%Y-%m-%d', time.localtime(date_taken))
        # if self.DEBUG:
        #     print(unassigned_face.face_thumbnail)
        #     print(unassigned_face.source_image_file.dateTaken, date_taken)

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
        metrics_array = np.zeros((len(candidate_ids), 3))

        for row_num, db_id in enumerate(candidate_ids):
            cmp_face_encodings = self.embedding_dict[db_id]
            cmp_encoding_norms = self.norm_dict[db_id]
            dot_product = np.dot(query_encoding, cmp_face_encodings)
            assert len(dot_product) == len(cmp_encoding_norms)
            similarity = dot_product / (cmp_encoding_norms * query_encoding_norm)
            assert len(similarity) == len(cmp_encoding_norms)

            sim_max = np.max(similarity)
            sim_99th = np.percentile(similarity, 99)
            # if sim_max > 0.5:
            #     print(sim_max, sim_99th, db_id, unassigned_face.id)

            metrics_array[row_num, :] = [sim_max, sim_99th, db_id]
            if self.DEBUG and db_id == debug_face_id and debug_face_id is not None:
                print("Row values: ", metrics_array[row_num, :])
            # print(np.max(similarity))
            # similarity_ordered = np.sort(similarity)[::-1]
            # print(similarity_ordered)

        possible_idcs = np.where(metrics_array[:, 0] > self.ASSIGN_THRESH)[0]
        if self.DEBUG:
            print(possible_idcs, "poss len is", len(possible_idcs))
            print(metrics_array[possible_idcs, :])
        if len(possible_idcs) == 0:
            # print("TODO: Assign to ignore person")
            metric_max = np.max(metrics_array[:, 0])
            weight = 1 - metric_max # High scores are presented on the
                # screen first, so something that has a low similarity
                # should have 1-value for a high score. 

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
                unassigned_face.set_possible_person(self.ignore_person_id, 1, weight)
        else:
            scores = metrics_array[possible_idcs, 0]
            weights = metrics_array[possible_idcs, 1]
            assign_ids = metrics_array[possible_idcs, 2].astype(np.int64)

            order = np.argsort(scores)[::-1]
            order = order[:5]
            for precedence_idx, order_idx in enumerate(order):
                # print(assign_ids[order_idx], precedence_idx, weights[order_idx])
                unassigned_face.set_possible_person(int(assign_ids[order_idx]), precedence_idx + 1, float(weights[order_idx]))

