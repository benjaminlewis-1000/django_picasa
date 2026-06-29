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

        self.ASSIGN_THRESH=0.8

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
        print(self.num_likely_people)

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

            print(f"There are {num_unassigned} faces to classify. We are pre-loading the database's encodings, which may take several minutes.")
            ss = time.time()

            self.candidate_dict = {}
            self.embedding_dict = {}
            self.norm_dict = {}
            if os.path.exists(self.ENCODINGS_PKL_FILE):
                with open(self.ENCODINGS_PKL_FILE, 'rb') as ph:
                    combo_dict = pickle.load(ph)

                    self.candidate_dict = combo_dict['candidate_dict']
                    self.embedding_dict = combo_dict['embedding_dict']
                    self.norm_dict = combo_dict['norm_dict']
                print("Dataframe loaded from file")
            # else:
            changed = False

            for face_id in tqdm(self.likely_people_ids):

                if face_id in self.candidate_dict.keys() and \
                   face_id in self.embedding_dict.keys() and \
                   face_id in self.norm_dict.keys():
                    print(f'No need to process this face {face_id}')
                    continue
                print(f"Processing face {face_id}")
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
        else:
            self.encoding_dataframe = None
        
        u_idx = 0
        s = time.time()
        for u_img in tqdm(unassigned.iterator()):
            # try:
            elps = time.time() - s
            s = time.time()
            if self.DEBUG:
                print(f"Assigning: {u_idx+1}/{num_unassigned} | {elps:.2f}")
                u_idx += 1
            self.classify_unassigned(u_img)
            # except Exception as e:
            #     print(f"Exception! {e}")

        # Finish up by "trueing up" the num_assigned for each person:
        print("Verifying face counts...")
        for p in tqdm(Person.objects.all()):
            p.num_faces = p.face_declared.count()
            p.num_possibilities = p.face_poss1.count() # + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
            p.num_unverified_faces = p.face_declared.filter(validated=False).count()
            p.save()




    def classify_unassigned(self, unassigned_face: Face) -> None:
        """
        DOCSTRING
        """

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
        candidate_id_arr = np.array(candidate_ids)

        # Pre-populate a metrics array
        metrics_array = np.zeros((self.num_likely_people, 3))

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
            # print(np.max(similarity))
            # similarity_ordered = np.sort(similarity)[::-1]
            # print(similarity_ordered)

        possible_idcs = np.where(metrics_array[:, 0] > self.ASSIGN_THRESH)[0]
        if len(possible_idcs) == 0:
            # print("TODO: Assign to ignore person")
            metric_max = np.max(metrics_array[:, 0])
            weight = 1 - metric_max # High scores are presented on the
                # screen first, so something that has a low similarity
                # should have 1-value for a high score. 

            if self.ignore_person_id in rejected_ids:
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

        # print(metrics_array)
        # exit()

        # exit()


        # for row_num, db_id in enumerate(candidate_ids):
        #     person = Person.objects.get(id=db_id)
        #     first_date_person = self.person_to_dates[db_id]['first_timestamp']
        #     if date_taken < first_date_person and date_taken > self.bogus_date_utc: 
        #         # This person is unlikely to be in this photo
        #         # dist_per_category.append(9999)
        #         # print(f'Person {person} is unlikely to be in this image on {date_string}')
        #         # continue
        #         pass
        #     else:
        #         faces_person = Q(declared_name__id=db_id)
        #         has_long = ~Q(face_encoding_512=None)
        #         long_encoded = ~Q(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)
                
        #         s1 = time.time()
        #         closest_faces = Face.objects \
        #             .filter(faces_person & long_encoded & has_long) \
        #             .annotate(result=Abs(F('dateTakenUTC') - date_taken)) \
        #             .order_by('result')
        #         # Get the N_COMPARISONS closest faces
        #         closest_faces = closest_faces[:self.N_COMPARISONS]
        #         print(f"Close face query takes {time.time() - s1:.2f} sec")

        #         if self.encoding_dataframe is None:
        #             cmp_encodings = np.array(closest_faces.values_list('face_encoding_512', flat=True))
        #         else:
        #             ss = time.time()
        #             cmp_ids = list(closest_faces.values_list('id', flat=True))
        #             cmp_ids.sort()
        #             print(f"Getting ids takes {time.time() - ss: .2f} sec, ids are {cmp_ids}")
        #             cmp_encodings = self.encoding_dataframe.loc[cmp_ids]
        #             cmp_encodings = np.array(cmp_encodings['face_encoding_512'].tolist())
        #             # print(cmp_ids)
        #             assert len(cmp_encodings) == len(cmp_ids)
        #             assert cmp_encodings.shape == (len(cmp_ids), 512)
        #         # print(cmp_encodings.shape, encoding.shape)




        # exit()


    # def known_persons_to_dates(self):
    #     # Get the dates for all pictures with a given person tagged in
    #     # them. Also calculate the first (non-bogus) timestamp of the person
    #     # appearing. 
    #     self.person_to_dates = {}
    #     for known_id in self.likely_people_ids:
    #         faces_person = Q(declared_name__id=known_id)
    #         p = Person.objects.get(id=known_id)
    #         faces = Face.objects.filter(faces_person).order_by('id')
    #         face_ids = list(faces.values_list('id', flat=True))
    #         # face_timestamps = [f.source_image_file.dateTakenUTC for f in faces]
    #         face_timestamps = list(faces.values_list('dateTakenUTC', flat=True))
    #         timestamps_sorted = np.sort(face_timestamps).reshape(-1, 1)
    #         earliest_date_idx = np.where(timestamps_sorted > self.bogus_date_utc)
    #         timestamps_sorted_nonbogus = timestamps_sorted[earliest_date_idx]
    #         # Modified Z score

    #         median = np.median(timestamps_sorted_nonbogus, axis=0)
    #         diff = (timestamps_sorted_nonbogus - median)**2
    #         diff = np.sqrt(diff)
    #         med_abs_deviation = np.median(diff)

    #         modified_z_score = 0.6745 * diff / med_abs_deviation

    #         min_idx = np.argmin(modified_z_score)
    #         modified_z_score = modified_z_score[:min_idx]

    #         # Compute z score as a heuristic to get earliest date
    #         # z_score = stats.zscore(timestamps_sorted_nonbogus).reshape(-1, 1)
    #         # Then inter-quartile range
    #         q1 = np.percentile(modified_z_score, 25)#! /
    #         q3 = np.percentile(modified_z_score, 75)
    #         iqr = q3 - q1
    #         lower_z = q1 - iqr 
    #         upper_z = q3 + iqr 

    #         # Get the threshold 
    #         # The modified z score is a parabola, so only get the
    #         # first half
    #         thresh_idx = np.where(modified_z_score[:min_idx] > upper_z)[0]
    #         if len(thresh_idx) > 0:
    #             thresh_idx = np.max(thresh_idx) + 1
    #         else:
    #             thresh_idx = 0
    #         # Throw in a couple more indices for fun
    #         idx_add = int(np.ceil(len(face_ids) // 1000))
    #         # print(known_id)
    #         # print(upper_z)
    #         # print(modified_z_score[:10])
    #         thresh_idx += idx_add

    #         # best_early = np.min(timestamps_sorted_nonbogus[np.where(z_score > -1)])
    #         earliest_date = timestamps_sorted_nonbogus[thresh_idx]

    #         person_data = {}
    #         person_data['timestamps'] = face_timestamps
    #         person_data['face_ids'] = face_ids
    #         # Unlikely to get images before the first timestamp - or we can
    #         # declare no images more than x days before the earliest timestamp. 
    #         person_data['first_timestamp'] = earliest_date

    #         date_string = time.strftime('%Y-%m-%d', time.localtime(earliest_date))
    #         # print(p, '|', date_string, earliest_date)


    #         self.person_to_dates[known_id] = person_data
