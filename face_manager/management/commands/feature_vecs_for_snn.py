#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager import face_classifier
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from scipy import stats
from time import sleep
import time
import dlib
import os
import face_recognition
import numpy as np
import pickle

class Command(BaseCommand):
    
    # def add_arguments(self, parser):
    #     parser.add_argument('epoch', type=int)

    def __init__(self):
        super(Command, self).__init__()
        self.TEST_YEAR=2020
        self.MIN_NUM_IMAGES=100


        self.criterion_short_feature = ~Q(face_encoding=None)
        self.criterion_long_feature = ~Q(face_encoding_512=None)
        self.criterion_ign = ~Q(declared_name__person_name__in=settings.IGNORED_NAMES)
        self.unassigned_crit = ~Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        self.criterion_ign_person = ~Q(person_name__in=settings.IGNORED_NAMES)

    def get_ignore_encs(self):

        def dump_in_sections(data, pkl_name):
            data_len = len(data)
            chunk_size = 500
            num_iters = int(np.ceil(data_len / chunk_size))
            print(num_iters, data_len)
            with open(pkl_name, 'wb') as ph:
                for ii in range(num_iters):
                    print(f"Dumping iter {ii} to file {os.path.basename(pkl_name)}")
                    # print(ii * chunk_size,(ii + 1) * chunk_size)
                    data_chunk = data[ii * chunk_size:(ii + 1) * chunk_size]
                    print(len(data_chunk))
                    pickle.dump(data_chunk, ph)

        criterion_rejected = Q( declared_name__person_name__in=['.ignore', '.realignore'])
        criterion_short_feature = ~Q(face_encoding=None)
        criterion_long_feature = ~Q(face_encoding_512=None)
        ign_faces = Face.objects.filter(criterion_short_feature & criterion_long_feature & criterion_rejected).order_by('id')

        # id_nums_ign = list(ign_faces.values_list('declared_name__id', flat=True))
        short_encodings_ign = list(ign_faces.values_list('face_encoding', flat=True))
        dump_in_sections(short_encodings_ign, '/code/MISC_DATA/short_enc_IGNORE.pkl')

        long_encodings_ign = list(ign_faces.values_list('face_encoding_512', flat=True))
        dump_in_sections(long_encodings_ign, '/code/MISC_DATA/long_enc_IGNORE.pkl')

        combined_encodings_ign = [short_encodings_ign[i] + long_encodings_ign[i] for i in range(len(short_encodings_ign))]
        del long_encodings_ign
        del short_encodings_ign
        dump_in_sections(combined_encodings_ign, '/code/MISC_DATA/combined_enc_IGNORE.pkl')
        del combined_encodings_ign

        id_nums_ign = [99999] * ign_faces.count()
        dump_in_sections(id_nums_ign, '/code/MISC_DATA/id_nums_IGNORE.pkl')
        del id_nums_ign

    def get_test_images(self):
        criterion_date = Q(source_image_file__dateTaken__year=self.TEST_YEAR)

        test_has_features = Face.objects.filter(self.criterion_short_feature & \
                self.criterion_long_feature & criterion_date & self.unassigned_crit & \
                self.criterion_ign).order_by('id')

        print(test_has_features.count())

        person_id_nums = np.array(test_has_features.values_list('declared_name__id', flat=True))
        image_id_nums = np.array(test_has_features.values_list('id', flat=True))

        sufficent_instances_people = Person.objects.annotate(c=Count('face_declared', filter=self.criterion_ign_person )).filter(c__gt=self.MIN_NUM_IMAGES)

        # def chunks(lst, n_chunks):
        #     """Yield successive n-sized chunks from lst."""
        #     n = len(lst) // n_chunks
        #     for i in range(0, len(lst), n):
        #         yield lst[i:i + n]


        for idx, person in enumerate(sufficent_instances_people):
            person_idcs = np.where(person_id_nums == person.id)[0]
            if len(person_idcs) < self.MIN_NUM_IMAGES:
                # print("NOT Processing person ", person.person_name, len(person_idcs))
                continue
            print("Processing person ", person.person_name, len(person_idcs))

            short_encodings = {}
            long_encodings = {}
            cat_encodings = {}
            labels_dict = {}

            ii = 0

            # sub_idcs = next(idx_chunker)
            person_img_ids = [image_id_nums[xx] for xx in person_idcs]
            sub_faces = Face.objects.filter(id__in=person_img_ids).order_by('source_image_file__dateTaken')
            short_encodings_sub = list(sub_faces.values_list('face_encoding', flat=True))
            long_encodings_sub = list(sub_faces.values_list('face_encoding_512', flat=True))
            combined_encodings = [short_encodings_sub[i] + long_encodings_sub[i] for i in range(len(short_encodings_sub))]
            assert len(short_encodings_sub) == len(long_encodings_sub)
            labels = [idx] * len(short_encodings_sub)
            short_encodings[ii] = short_encodings_sub
            long_encodings[ii] = long_encodings_sub
            cat_encodings[ii] = combined_encodings
            labels_dict[ii] = labels


            with open(f'/code/MISC_DATA/long_enc_{idx}_testyear_{self.TEST_YEAR}.pkl', 'wb') as ph:
                pickle.dump(long_encodings, ph)
            with open(f'/code/MISC_DATA/short_enc_{idx}_testyear_{self.TEST_YEAR}.pkl', 'wb') as ph:
                pickle.dump(short_encodings, ph)
            with open(f'/code/MISC_DATA/id_nums_{idx}_testyear_{self.TEST_YEAR}.pkl', 'wb') as ph:
                pickle.dump(labels_dict, ph)
            with open(f'/code/MISC_DATA/combined_enc_{idx}_testyear_{self.TEST_YEAR}.pkl', 'wb') as ph:
                pickle.dump(cat_encodings, ph)

    
    def handle(self, *args, **options):

        # Get the ignores 
        if False:
            print("Retrieving features for ignores...")
            self.get_ignore_encs()
            print("Got features for ignores")

        if True:
            self.get_test_images()

        exit()

        criterion_date = ~Q(source_image_file__dateTaken__year=self.TEST_YEAR)

        has_features = Face.objects.filter(self.criterion_short_feature & \
                self.criterion_long_feature & criterion_date & self.unassigned_crit & \
                self.criterion_ign).order_by('source_image_file__dateTaken')
        print(has_features.count())

        person_id_nums = np.array(has_features.values_list('declared_name__id', flat=True))
        image_id_nums = np.array(has_features.values_list('id', flat=True))
        print(has_features[0].source_image_file.dateTaken)
        # short_encodings = list(has_features.values_list('face_encoding', flat=True))
        # print(len(short_encodings))
        # long_encodings = list(has_features.values_list('face_encoding_512', flat=True))
        # print(len(long_encodings))

        sufficent_instances_people = Person.objects.annotate(c=Count('face_declared', filter=self.criterion_ign_person)).filter(c__gt=self.MIN_NUM_IMAGES)

        def chunks(lst, n_chunks):
            """Yield successive n-sized chunks from lst."""
            n = len(lst) // n_chunks
            for i in range(0, len(lst), n):
                yield lst[i:i + n]


        for idx, person in enumerate(sufficent_instances_people):
            person_idcs = np.where(person_id_nums == person.id)[0]
            print("Processing person ", person.person_name)
            # Split into three datasets by date. 
            idx_chunker = chunks(person_idcs, 3)

            short_encodings = {}
            long_encodings = {}
            cat_encodings = {}
            labels_dict = {}


            for ii in range(3):
                sub_idcs = next(idx_chunker)
                sub_identities = [image_id_nums[xx] for xx in sub_idcs]
                sub_faces = Face.objects.filter(id__in=sub_identities).order_by('source_image_file__dateTaken')

                short_encodings_sub = list(sub_faces.values_list('face_encoding', flat=True))
                long_encodings_sub = list(sub_faces.values_list('face_encoding_512', flat=True))

                combined_encodings = [short_encodings_sub[i] + long_encodings_sub[i] for i in range(len(short_encodings_sub))]

                assert len(short_encodings_sub) == len(long_encodings_sub)
                labels = [idx] * len(short_encodings_sub)

                short_encodings[ii] = short_encodings_sub
                long_encodings[ii] = long_encodings_sub
                cat_encodings[ii] = combined_encodings
                labels_dict[ii] = labels



            with open(f'/code/MISC_DATA/long_enc_{idx}.pkl', 'wb') as ph:
                pickle.dump(long_encodings, ph)
            with open(f'/code/MISC_DATA/short_enc_{idx}.pkl', 'wb') as ph:
                pickle.dump(short_encodings, ph)
            with open(f'/code/MISC_DATA/id_nums_{idx}.pkl', 'wb') as ph:
                pickle.dump(labels_dict, ph)
            with open(f'/code/MISC_DATA/combined_enc_{idx}.pkl', 'wb') as ph:
                pickle.dump(cat_encodings, ph)
