#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from scipy import stats
from scipy.spatial import distance
# from sklearn.cluster import DBSCAN
from time import sleep
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import dlib
# import face_recognition
# import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class encodingSiamese(nn.Module):

    def __init__(self, encoding_size, hidden_size):
        super(encodingSiamese, self).__init__()

        # self.fc1 = nn.Linear(encoding_size, 64)
        self.intermediate_size = hidden_size
        self.linear = nn.Sequential(nn.Linear(encoding_size, self.intermediate_size), nn.Sigmoid())
        self.out = nn.Linear(self.intermediate_size, 1)


    def forward_one(self, x):
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        distance = torch.abs(out1 - out2)
        out = self.out(distance)

        return out


class Command(BaseCommand):

    def __init__(self):
        super(Command, self).__init__()

        model_dir = '/models/optuna_wts'

        self.small_net_dict = {
            'trial_number': 290,
            'activation': 'Sigmoid', 
            'batch_size': 197, 
            'dropout_l0': 0.21937762517255865, 
            'dropout_l1': 0.4537798766730989, 
            'lr': 0.005352377712916207, 
            'n_layers': 3, 
            'n_units_l0': 512, 
            'n_units_l1': 313, 
            'n_units_l2': 501, 
            'noise_std': 0.015030484764573257, 
            'optimizer': 'Adam', 
            'schedule_interval': 7978,
        }

        self.big_net_dict = {
            'trial_number': 821,
            'activation': 'ReLU', 
            'batch_size': 258, 
            'dropout_l0': 0.22176656193462396, 
            'dropout_l1': 0.218724690245919, 
            'lr': 0.0032257719875885315, 
            'n_layers': 3, 
            'n_units_l0': 400, 
            'n_units_l1': 443, 
            'n_units_l2': 420, 
            'noise_std': 0.05738525389486293, 
            'optimizer': 'Adam', 
            'schedule_interval': 31020,
        }

        self.combine_net_dict = {
            'trial_number': 602,
            'activation': 'ReLU', 
            'batch_size': 267, 
            'dropout_l0': 0.32704602320278076, 
            'dropout_l1': 0.2050031183680314, 
            'lr': 0.0021381231100890533, 
            'n_layers': 3, 
            'n_units_l0': 490, 
            'n_units_l1': 315, 
            'n_units_l2': 502, 
            'noise_std': 0.023714925023645965, 
            'optimizer': 'Adam', 
            'schedule_interval': 24513,
        }

        self.build_and_load_network('short')
        self.build_and_load_network('long')
        self.build_and_load_network('combined')

        print(self.short_enc_net)
        print(self.long_enc_net)
        print(self.combined_enc_net)
        print(self.short_merge_net)
        print(self.long_merge_net)
        print(self.combined_merge_net)
        # exit()

    def build_and_load_network(self, short_or_long):
        if short_or_long == 'short':
            definition_dict = self.small_net_dict
            in_features = 128
        elif short_or_long == 'long':
            definition_dict = self.big_net_dict
            in_features = 512
        elif short_or_long == 'combined':
            definition_dict = self.combine_net_dict
            in_features = 512 + 128

        assert 'n_layers' in definition_dict.keys()
        assert 'activation' in definition_dict.keys()
        trial_num = definition_dict['trial_number']

        enc_weight_path = os.path.join(f'/models/optuna_wts/weights_encoder_{short_or_long}_trial_{trial_num}.ptw')
        merge_weight_path = os.path.join(f'/models/optuna_wts/weights_outlayer_{short_or_long}_trial_{trial_num}.ptw')

        n_layers = definition_dict['n_layers']
        activation = definition_dict['activation']
        layers = []

        for ii in range(n_layers):
            out_features = definition_dict[f'n_units_l{ii}']
            layers.append(nn.Linear(in_features, out_features))
            layers.append(getattr(nn, activation)())
            if ii < n_layers - 1:
                dropout_pct = definition_dict[f'dropout_l{ii}']
                layers.append(nn.Dropout(dropout_pct))

            in_features = out_features

        enc_net = nn.Sequential(*layers)
        enc_net.load_state_dict(torch.load(enc_weight_path, map_location=torch.device('cpu')))
        enc_net.eval()

        merge_layer = nn.Linear(in_features, 1)
        merge_layer.load_state_dict(torch.load(merge_weight_path, map_location=torch.device('cpu')))
        merge_layer.eval()

        if short_or_long == 'short':
            self.short_enc_net = enc_net
            self.short_merge_net = merge_layer
        elif short_or_long == 'long':
            self.long_enc_net = enc_net
            self.long_merge_net = merge_layer
        elif short_or_long == 'combined':
            self.combined_enc_net = enc_net
            self.combined_merge_net = merge_layer

        return enc_net, merge_layer
    
    def cos_matrix_multiplication(self, matrix, vector):
        """
        Calculating pairwise cosine distance using matrix vector multiplication.
        """
        dotted = matrix.dot(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        vector_norm = np.linalg.norm(vector)
        matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
        neighbors = np.divide(dotted, matrix_vector_norms)
        return 1 - neighbors

    def get_annotations(self):
        pass

    def clear_unassigned_images(self):
        # Only used if I want to reassign all the faces. 
        unassigned_crit = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        proposed = ~Q(poss_ident1=None)
        # proposed = ~Q(weight_1=0)
        num_to_reset = Face.objects.filter(unassigned_crit & proposed).count()
        print(f"Clearing {num_to_reset} unassigned images...")
        # Filter and update
        Face.objects.filter(unassigned_crit & proposed).update(poss_ident1=None, 
            poss_ident2=None, poss_ident3=None, poss_ident4=None, poss_ident5=None,
            weight_1 = 0, weight_2 = 0, weight_3 = 0, weight_4 = 0, weight_5 = 0,  )

        return 


    def get_filtered_list_of_faces(self, min_num_faces):

        criterion_ign = ~Q(person_name__in=settings.IGNORED_NAMES)
        criterion_unlikely = Q(further_images_unlikely=False)

        assigned_people = Person.objects.annotate(c=Count('face_declared', filter=criterion_ign & criterion_unlikely)).filter(c__gt=min_num_faces)

        person_ids = [p.id for p in assigned_people]

        return person_ids
    
    def extract_data_for_face(self, id_num):
        '''
        Given the id number for a person, get all the 
        encodings for that person. 
        '''

        faces = Face.objects.filter(declared_name=id_num).order_by('id')
        # Get the encodings. 
        face_encodings = list(faces.values_list('face_encoding', flat=True))
        face_encodings_long = list(faces.values_list('face_encoding_512', flat=True))
        face_ids = list(faces.values_list('id', flat=True))
        # print(face_ids)
        # Get the source image files. 
        face_sources = faces.values_list('source_image_file', flat=True)
        img_files = ImageFile.objects.in_bulk(face_sources)
        sorted_img_files = [img_files[ii] for ii in face_sources]
        face_dates = np.array([img.dateTaken.timestamp() for img in sorted_img_files])


        return face_encodings, face_encodings_long, face_dates, face_ids

    def compute_distance(self, compare_array, test_encoding):
        assert len(compare_array) > 0
        assert len(compare_array[0]) == len(test_encoding)

        if len(test_encoding) == 128:
            enc_net = self.short_enc_net
            merge_net = self.short_merge_net
        elif len(test_encoding) == 512:
            enc_net = self.long_enc_net
            merge_net = self.long_merge_net
        else:
            raise ValueError('Length of encoding is not in [128, 512]')

        sig = nn.Sigmoid()
        test_encoding = torch.Tensor(test_encoding)
        # test_encoding_rep = test_encoding.repeat(len(compare_array), 1)
        compare_array_tensor = torch.Tensor(compare_array)

        # distances = []
        with torch.no_grad():
            test_encoding = enc_net(test_encoding)
            cmp_encoding = enc_net(compare_array_tensor)
            intermediate_distance = torch.abs(cmp_encoding - test_encoding)
            out = merge_net(intermediate_distance)

            distances = sig(out)

        distances = distances.detach().cpu().numpy().reshape(-1)
        distances = 1 - distances
        return distances

    def compute_distance_old(self, compare_array, test_encoding):
        assert len(compare_array) > 0
        assert len(compare_array[0]) == len(test_encoding)

        if len(test_encoding) == 128:
            net = self.small_model
        elif len(test_encoding) == 512:
            net = self.big_model
        else:
            raise ValueError('Length of encoding is not in [128, 512]')

        sig = nn.Sigmoid()
        test_encoding = torch.Tensor(test_encoding)
        test_encoding_rep = test_encoding.repeat(len(compare_array), 1)
        compare_array_tensor = torch.Tensor(compare_array)

        # distances = []
        with torch.no_grad():

            out = net(compare_array_tensor, test_encoding_rep)
            distances = sig(out)

        distances = distances.detach().cpu().numpy().reshape(-1)
        distances = 1 - distances
        return distances

    def compute_distance_cosine(self, compare_array, test_encoding):

        s = time.time()
        compare_matrix = np.array(compare_array)
        test_encoding = np.array(test_encoding)
        distances = self.cos_matrix_multiplication(compare_matrix, test_encoding)
        # print(f"Elapsed: {time.time() - s}")
        return list(distances)
        # distance.cosine()


    def classify_unassigned(self, u_img):

        date = u_img.source_image_file.dateTaken.timestamp()

        if self.DEBUG:
            print(u_img.source_image_file.dateTaken, date)

        short_encoding = u_img.face_encoding
        long_encoding = u_img.face_encoding_512

        if short_encoding is None or long_encoding is None:
            return

        if u_img.declared_name.person_name != settings.BLANK_FACE_NAME:
            print("Already assigned")
            return

        short_encoding = torch.Tensor(short_encoding)
        long_encoding = torch.Tensor(long_encoding)

        distances_matrix = None
        dist_per_category = []

        for known_id in self.person_keys_list:
            # Find the encodings for each of the candidate people
            # that are temporally closest. 
            known_dates = self.image_dates[known_id]
            date_diffs = np.abs(known_dates - date)
            date_diff_sort = np.argsort(date_diffs)
            closest_date_idcs = date_diff_sort[:self.NUM_CLOSEST]
            # closest_date_idcs = np.where(date_diffs < self.NUM_DAYS * 24 * 60 * 60)[0]

            # Get the corresponding face encodings
            encoding_list_short = self.person_encodings_short[known_id]
            encoding_list_long = self.person_encodings_long[known_id]

            if self.DEBUG:
                encoding_list_short = [np.array(i) for i in encoding_list_short if i is not None]
                encoding_list_long = [np.array(i) for i in encoding_list_long if i is not None]
                temporally_closest_encodings_s = encoding_list_short
                temporally_closest_encodings_l = encoding_list_long
            else:
                temporally_closest_encodings_s = [np.array(encoding_list_short[i]) for i in closest_date_idcs if encoding_list_short[i] is not None]
                temporally_closest_encodings_l = [np.array(encoding_list_long[i]) for i in closest_date_idcs if encoding_list_long[i] is not None]


            if len(temporally_closest_encodings_s) == 0:
                dist_per_category.append(self.IGN_VALUE)
            else:
                # Compute the distance
                if self.DISTANCE_SIAMESE:
                    distances_long = self.compute_distance(temporally_closest_encodings_l, long_encoding)
                else:
                    distances_long = self.compute_distance_cosine(temporally_closest_encodings_l, long_encoding)
                # distances_short = self.compute_distance(temporally_closest_encodings_s, short_encoding)
                # distances_long = np.sort(distances_long)

                # long_idcs = np.argsort(distances_long)
                # print(distances_long[long_idcs[-1]])
                # print(np.array(long_encoding), np.array(temporally_closest_encodings_l[long_idcs[-1]]))
                # exit()
                # print(distances_short[:5])
                # print(distances_long[:5])                
                if self.DEBUG:
                    # print(np.max(distances_short), np.max(distances_long))
                    print(np.min(distances_long))
                    # print(np.mean(distances_short), np.mean(distances_long))
                    print(np.mean(distances_long))

                # # dist_idcs_short = np.argsort(distances_short)
                # distances_short.sort()
                # distances_short = distances_short[::-1]
                # if self.DEBUG:
                #    print(distances_short[:20])
                # mean_short = np.mean(distances_short[6:])
                # dist_idcs_long = np.argsort(distances_long)
                distances_long.sort()
                if not self.USE_MIN_VALUE:
                    distances_long = distances_long[::-1]
                mean_long = np.mean(distances_long[:10])
                min_long = np.min(distances_long)

                if self.DISTANCE_SIAMESE:
                    array_val = mean_long
                else:
                    array_val = min_long

                if self.DEBUG:
                    print(distances_long[:20])
                # min_dist_per_category.append(np.max((mean_long, mean_short)))
                if self.USE_MIN_VALUE:
                    dist_per_category.append(array_val)
                else:
                    dist_per_category.append(mean_long)


        # Set the indices for rejected fields to something huge
        if u_img.rejected_fields is not None:
            for rejected_id in u_img.rejected_fields:
                if rejected_id in self.person_ids:
                    idx = self.rejected_field_to_id_list_idx(rejected_id)
                    dist_per_category[idx] = self.IGN_VALUE
                    # distances_matrix[idx, :] = self.IGN_VALUE


        print(dist_per_category)
        # Find the minimum index
        if self.USE_MIN_VALUE:
            select_idx = np.argmin(dist_per_category)
            select_val = np.min(dist_per_category)

            # Want the weight to be high in both cases. 
            if select_val < self.WEIGHT_THRESH_MIN:
                person_id = Person.objects.get(id=self.person_keys_list[ select_idx ])
                weight_val = (self.WEIGHT_THRESH_MIN - select_val) / self.WEIGHT_THRESH_MIN
            else: 
                # Weight for out-of-library person. 
                person_id = self.ignore_person
                weight_val = select_val
            print(select_idx, select_val, weight_val)
        else:
            select_idx = np.argmax(dist_per_category)
            select_val = np.max(dist_per_category)

            if select_val > self.WEIGHT_THRESH_MAX:
                person_id = Person.objects.get(id=self.person_keys_list[ select_idx ])
                weight_val = select_val
            else: 
                person_id = self.ignore_person
                weight_val = (self.WEIGHT_THRESH_MAX - select_val) / self.WEIGHT_THRESH_MAX


        u_img.set_possibles_zero()
        u_img.set_possible_person( person_id.id, 1, weight_val)

        print("==========================")

        #     # if ignore_person.id in u_img.rejected_fields:
        #     if len(u_img.rejected_fields) > 0:
        #         # Theory: If I rejected them once, it probably means I want
        #         # them to be assigned *somewhere*. 
        #         min_idx = np.argmin(min_dist_per_category)
        #         min_val = np.min(min_dist_per_category)
        #         min_dist_per_category[min_idx] = np.min((min_val, WEIGHT_THRESH-0.01))

        # def smallestN_indices(a, N):
        #     a = a.copy()
        #     idx = a.ravel().argsort()[:N]
        #     vals = a.ravel()
        #     vals.sort()
        #     vals = vals[:N]
        #     idcs_2d = np.stack(np.unravel_index(idx, a.shape)).T
        #     row_nums = idcs_2d[:, 0]
        #     return row_nums, vals

        # # N_SMALLEST = 25
        # # min_rows, min_vals = smallestN_indices(distances_matrix, N_SMALLEST)
        # # bincount = np.bincount(min_rows)
        # # max_bin_size = np.max(bincount)

        # # if max_bin_size <= N_SMALLEST * 0.2:
        # #     person_id = ignore_person
        # #     weight_val = (N_SMALLEST - max_bin_size) / N_SMALLEST
        # # else:
        # #     person_idx = np.argmax(bincount)
        # #     person_id = Person.objects.get(id=person_keys_list[person_idx])
        # #     weight_val = max_bin_size / N_SMALLEST 

        # # if min_vals[0] > 0.4:
        # #     person_id = ignore_person
        # #     weight_val = min_vals[0]
        # # else:
        # #     person_idx = min_rows[0]
        # #     person_id = Person.objects.get(id=person_keys_list[person_idx])
        # #     weight_val = min_vals[0]

        # # sleep(1)
    def rejected_field_to_id_list_idx(self, rejected_person_id):
        idx = self.person_ids.index(rejected_person_id)
        return idx


    def handle(self, *args, **options):

        # self.clear_unassigned_images()
    
        small_encoding_weights = '/models/weights_short_stable.ptw'
        big_encoding_weights = '/models/weights_long_stable.ptw'

        self.small_model = encodingSiamese(128, 256)
        self.small_model.eval()
        self.big_model = encodingSiamese(512, 256)
        self.big_model.eval()
        
        self.small_model.load_state_dict(torch.load(small_encoding_weights, map_location='cpu'))
        self.big_model.load_state_dict(torch.load(big_encoding_weights, map_location='cpu'))

        if False:
            self.clear_unassigned_images()

        self.DEBUG=False
        self.USE_MIN_VALUE=True
        if self.USE_MIN_VALUE:
            self.IGN_VALUE = 999
        else:
            self.IGN_VALUE = 0

        self.DISTANCE_SIAMESE=True
        self.WEIGHT_THRESH_MAX = 0.6
        self.WEIGHT_THRESH_MIN = 0.4

        self.MIN_NUM = 50
        self.NUM_DAYS = 180
        self.NUM_CLOSEST = 50
        self.NUM_TO_AVERAGE = 1

        self.person_ids = self.get_filtered_list_of_faces(min_num_faces = self.MIN_NUM)

        if self.DEBUG:
            self.person_ids = self.person_ids[:3]
            # self.person_ids = [self.person_ids[1]]

        


        # print(rejected_field_to_id_list_idx(unassigned[0].rejected_fields[0]))

        self.person_encodings_short = {}
        self.person_encodings_long = {}
        self.image_dates = {}

        for idx, id_num in enumerate(self.person_ids):
            print(f'{idx+1}/{len(self.person_ids)}')
            enc, enc_long, dates, face_ids = self.extract_data_for_face(id_num)
            self.person_encodings_short[id_num] = enc
            self.person_encodings_long[id_num] = enc_long
            self.image_dates[id_num] = dates

        ###############

        # Now we want to get each unassigned image
        unassigned_crit = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        unassigned = Face.objects.filter(unassigned_crit).order_by('?')
        self.person_keys_list = list(self.person_encodings_short.keys())
        self.ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME)[0]


        u_idx = 0
        for u_img in unassigned.iterator():
            try:
                print(f"Assigning: {u_idx+1}/{unassigned.count()}")
                u_idx += 1
                self.classify_unassigned(u_img)
                sleep(0.2)
            except Exception as e:
                print(f"Exception! {e}")
                # 
