#! /usr/bin/env python

from datetime import datetime
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count, Q, F
from django.db.models.functions import Abs
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from scipy import stats
from time import sleep
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import collections
import io
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional
import torchvision

class faceAssigner():

    """docstring for faceAssigner"""
    def __init__(self):
        super(faceAssigner, self).__init__()

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

        self.bogus_date = datetime(1990, 1, 1) # Very few images before that
        self.bogus_date_utc = time.mktime(self.bogus_date.timetuple())
        self.ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME)[0]

        model_dir = '/models/optuna_wts'

        self.siam_triplet = self.load_network(os.path.join(model_dir, 'output_37.pkl'))
        
        if False:
            self.clear_unassigned_images()

        # Sets self.person_ids
        self.get_filtered_list_of_faces(min_num_faces = self.MIN_NUM)
        # Shouldn't do anything, but double checks that 
        # a field is set in the database
        self.doubleCheckFacesTimes()
        # Get earliest date for each person. 
        self.known_persons_to_dates()

    def execute(self, redo_all=False):
        
        # Now we want to get each unassigned image
        unassigned_crit = Q(declared_name__person_name=settings.BLANK_FACE_NAME)

        if redo_all:
            unassigned = Face.objects.filter(unassigned_crit).order_by('?')
        else:
            no_suggestions = Q(poss_ident1__person_name=None)
            unassigned = Face.objects.filter(unassigned_crit).filter(no_suggestions).order_by('?')

        num_unassigned = int(unassigned.count())
        u_idx = 0
        for u_img in unassigned.iterator():
            try:
                print(f"Assigning: {u_idx+1}/{num_unassigned}")
                u_idx += 1
                self.classify_unassigned(u_img)
            except Exception as e:
                print(f"Exception! {e}")

    def load_network(self, net_data_pkl):
        with open(net_data_pkl, 'rb') as fh:
            data = pickle.load(fh)

        weight_file = io.BytesIO(data['weight_file_data'])
        weights = torch.load(weight_file, map_location=torch.device('cpu'))
        params = data['params']

        n_layers = params['n_layers']
        activation = params['activation']
        layers_sizes = []
        for ii in range(n_layers - 1):
            layers_sizes.append(params[f'n_units_l{ii}'])
        layers_sizes.append(400)

        dropouts = []
        for ii in range(n_layers - 1):
            dropouts.append(params[f'dropout_l{ii}'])

        net = siameseModel(n_layers, activation, layers_sizes, dropouts, 640, 'triplet')
        net.load_state_dict(weights)
        net.eval()

        return net


    # def compute_distance(self, base, comparisons):
    #     base_enc = self.combined_siamese_net(base).unsqueeze(0)
    #     cmp_enc = self.combined_siamese_net(comparisons)

    #     l2_dist = torch.cdist(base_enc, cmp_enc).squeeze(1)
    #     return l2_dist

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

        Person.objects.all().update(num_possibilities = 0)

        return 

    def get_filtered_list_of_faces(self, min_num_faces):

        criterion_ign = ~Q(person_name__in=settings.IGNORED_NAMES)
        criterion_unlikely = Q(further_images_unlikely=False)

        assigned_people = Person.objects.annotate(c=Count('face_declared', filter=criterion_ign & criterion_unlikely)).filter(c__gt=min_num_faces)

        self.person_ids = [p.id for p in assigned_people]

    def known_persons_to_dates(self):
        # Get the dates for all pictures with a given person tagged in
        # them. Also calculate the first (non-bogus) timestamp of the person
        # appearing. 
        self.person_to_dates = {}
        for known_id in self.person_ids:
            faces_person = Q(declared_name__id=known_id)
            p = Person.objects.get(id=known_id)
            faces = Face.objects.filter(faces_person).order_by('id')
            face_ids = list(faces.values_list('id', flat=True))
            # face_timestamps = [f.source_image_file.dateTakenUTC for f in faces]
            face_timestamps = list(faces.values_list('dateTakenUTC', flat=True))
            timestamps_sorted = np.sort(face_timestamps).reshape(-1, 1)
            earliest_date_idx = np.where(timestamps_sorted > self.bogus_date_utc)
            timestamps_sorted_nonbogus = timestamps_sorted[earliest_date_idx]
            # Modified Z score

            median = np.median(timestamps_sorted_nonbogus, axis=0)
            diff = (timestamps_sorted_nonbogus - median)**2
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation

            min_idx = np.argmin(modified_z_score)
            modified_z_score = modified_z_score[:min_idx]

            # Compute z score as a heuristic to get earliest date
            # z_score = stats.zscore(timestamps_sorted_nonbogus).reshape(-1, 1)
            # Then inter-quartile range
            q1 = np.percentile(modified_z_score, 25)#! /
            q3 = np.percentile(modified_z_score, 75)
            iqr = q3 - q1
            lower_z = q1 - iqr 
            upper_z = q3 + iqr 

            # Get the threshold 
            # The modified z score is a parabola, so only get the
            # first half
            thresh_idx = np.where(modified_z_score[:min_idx] > upper_z)[0]
            if len(thresh_idx) > 0:
                thresh_idx = np.max(thresh_idx) + 1
            else:
                thresh_idx = 0
            # Throw in a couple more indices for fun
            idx_add = int(np.ceil(len(face_ids) // 1000))
            # print(known_id)
            # print(upper_z)
            # print(modified_z_score[:10])
            thresh_idx += idx_add

            # best_early = np.min(timestamps_sorted_nonbogus[np.where(z_score > -1)])
            earliest_date = timestamps_sorted_nonbogus[thresh_idx]

            person_data = {}
            person_data['timestamps'] = face_timestamps
            person_data['face_ids'] = face_ids
            # Unlikely to get images before the first timestamp - or we can
            # declare no images more than x days before the earliest timestamp. 
            person_data['first_timestamp'] = earliest_date

            date_string = time.strftime('%Y-%m-%d', time.localtime(earliest_date))
            # print(p, '|', date_string, earliest_date)


            self.person_to_dates[known_id] = person_data

    def doubleCheckFacesTimes(self):
        # This one shouldn't do anything, but it's a backup 
        # for now.
        faces = Face.objects.filter(dateTakenUTC=0)

        count = faces.count()
        # print(f"Faces found: {count}")

        for idx, face in enumerate(faces.iterator()):
            if idx % 100 == 0:
                print(f'{idx / count * 100:.2f}%')
            date_utc = face.source_image_file.dateTakenUTC
            face.dateTakenUTC = date_utc
            super(Face, face).save()

    def classify_unassigned(self, u_img):

        date = u_img.source_image_file.dateTaken.timestamp()
        # print(date)
        date_string = time.strftime('%Y-%m-%d', time.localtime(date))
        if self.DEBUG:
            print(u_img.face_thumbnail)
            print(u_img.source_image_file.dateTaken, date)

        short_encoding = torch.Tensor(u_img.face_encoding)
        long_encoding = torch.Tensor(u_img.face_encoding_512)
        base_combined = torch.Tensor(np.concatenate((long_encoding, short_encoding)))
        base_enc = self.siam_triplet.feature_fwd(base_combined).unsqueeze(0)

        if short_encoding is None or long_encoding is None:
            print("Short and/or long encoding not set")
            return

        if u_img.declared_name.person_name != settings.BLANK_FACE_NAME:
            print("Already assigned")
            return

        distances_matrix = None
        # dist_per_category = []

        N_COMPARISONS=25

        comparison_mat = np.ones((len(self.person_ids), N_COMPARISONS)) * 999

        s = time.time()
        if u_img.rejected_fields is not None:
            rejected_ids = u_img.rejected_fields
        else:
            rejected_ids = []
        for row_num, known_id in enumerate(self.person_ids):
            
            person = Person.objects.get(id=known_id)
            first_date_person = self.person_to_dates[known_id]['first_timestamp']
            if date < first_date_person and date > self.bogus_date_utc: 
                # This person is unlikely to be in this photo
                # dist_per_category.append(9999)
                # print(f'Person {person} is unlikely to be in this image on {date_string}')
                # continue
                pass

            elif known_id in rejected_ids:
                pass
            else:
                # Get a number of comparison images from this person, weighted closely
                # temporally. 
                faces_person = Q(declared_name__id=known_id)
                has_short = ~Q(face_encoding=None)
                has_long = ~Q(face_encoding_512=None)
                # Annotate - absolute temporal distance of other faces
                # from this face's date (using annotate). 
                # Filter by faces with this known ID. 
                closest_faces = Face.objects.annotate(result=Abs(F('dateTakenUTC') - date)) \
                    .filter(faces_person & has_short & has_long).order_by('result')
                # Get the N_COMPARISONS closest faces
                closest_faces = closest_faces[:N_COMPARISONS]

                short_encs = np.array(closest_faces.values_list('face_encoding', flat=True))
                long_encs = np.array(closest_faces.values_list('face_encoding_512', flat=True))
                cmp_combined = torch.Tensor(np.concatenate((long_encs, short_encs), 1))

                # Compute distance with siamese net
                cmp_enc = self.siam_triplet.feature_fwd(cmp_combined)
                l2_dist = torch.cdist(base_enc, cmp_enc).squeeze(1).detach().numpy().reshape(-1)
                l2_dist.sort()

                # print(l2_dist)
                comparison_mat[row_num, :len(l2_dist)] = l2_dist
                # print('here')
                if len(l2_dist) < N_COMPARISONS:
                    comparison_mat[row_num, len(l2_dist):] = l2_dist[-1]
                # print('here')

                # dist_per_category.append(float(torch.min(l2_dist).detach()))

        # print(comparison_mat)
        top_votes = np.argmin(comparison_mat, 0)
        vote_counts = np.bincount(top_votes)
        # print(top_votes)
        # print(vote_counts)
        most_votes_ranked = np.argsort(vote_counts)[::-1]
        average_dists = np.mean(comparison_mat, 1)

        best_id = self.person_ids[most_votes_ranked[0]]
        first_wt = average_dists[most_votes_ranked[0]]
        person = Person.objects.get(id=best_id)
        if self.DEBUG:
            print(time.time() - s, person, first_wt, len(average_dists), len(self.person_ids), average_dists)

        # Set a threshold for a not-person
        thresh = 55
        if first_wt > thresh and self.ignore_person.id not in rejected_ids:
            # Set as an unknown person
            person_id = self.ignore_person
            weight_val = first_wt
        else:
            person_id = person
            weight_val = np.max((0, thresh - first_wt))

        u_img.set_possibles_zero()
        u_img.set_possible_person( person_id.id, 1, weight_val)


class siameseModel(nn.Module):
    def __init__(self, n_layers, activation, layers_sizes, dropouts, in_size, loss_type):

        super(siameseModel, self).__init__()

        layers = []
        for ii in range(n_layers):
            out_size = layers_sizes[ii]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(getattr(nn, activation)())
            if ii < n_layers - 1:
                drop_pct = dropouts[ii]
                layers.append(nn.Dropout(drop_pct))

            in_size = out_size

        layers.append(getattr(nn, activation)())
        self.feature_fwd = nn.Sequential(*layers)

        self.loss_type = loss_type
        if self.loss_type == 'cross_entropy':
            self.joint_layer = nn.Linear(in_size, 1)

    def forward(self, *arg):
        if self.loss_type == 'cross_entropy':
            assert len(arg) == 2

            base = arg[0]
            second = arg[1]

            enc_1 = self.feature_fwd(base)
            enc_2 = self.feature_fwd(second)
            distance = torch.abs(enc_1 - enc_2)

            out = self.joint_layer(distance)

            return out # No sigmoid

        elif self.loss_type == 'triplet':
            assert len(arg) == 3
            base = self.feature_fwd(arg[0])
            pos = self.feature_fwd(arg[1])
            neg = self.feature_fwd(arg[2])

            return base, pos, neg