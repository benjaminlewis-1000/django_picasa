#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from scipy import stats
from sklearn.cluster import DBSCAN
from time import sleep
import time
import dlib
import face_recognition
import numpy as np
import pickle

class Command(BaseCommand):
    
    # def add_arguments(self, parser):
    #     parser.add_argument('epoch', type=int)

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
        face_ids = list(faces.values_list('id', flat=True))
        # print(face_ids)
        # Get the source image files. 
        face_sources = faces.values_list('source_image_file', flat=True)
        img_files = ImageFile.objects.in_bulk(face_sources)
        sorted_img_files = [img_files[ii] for ii in face_sources]
        face_dates = np.array([img.dateTaken.timestamp() for img in sorted_img_files])


        return face_encodings, face_dates, face_ids


    def handle(self, *args, **options):
        
        if True:
            self.clear_unassigned_images()

        DEBUG=False

        MIN_NUM = 50
        NUM_DAYS = 900
        NUM_CLOSEST = 10
        WEIGHT_THRESH = 0.35
        IGN_VALUE = 999
        NUM_TO_AVERAGE = 1

        person_ids = self.get_filtered_list_of_faces(min_num_faces = MIN_NUM)

        if DEBUG:
            person_ids = person_ids[:1]

        def rejected_field_to_id_list_idx(rejected_person_id):
            idx = person_ids.index(rejected_person_id)
            return idx


        # print(rejected_field_to_id_list_idx(unassigned[0].rejected_fields[0]))

        person_encodings = {}
        image_dates = {}

        for idx, id_num in enumerate(person_ids):
            print(f'{idx+1}/{len(person_ids)}')
            enc, dates, face_ids = self.extract_data_for_face(id_num)
            person_encodings[id_num] = enc
            image_dates[id_num] = dates

        # Now we want to get each unassigned image
        unassigned_crit = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        unassigned = Face.objects.filter(unassigned_crit)
        person_keys_list = list(person_encodings.keys())
        ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME)[0]


        u_idx = 0
        for u_img in unassigned.iterator():
            # try:
                print(f"Assigning: {u_idx+1}/{unassigned.count()}")
                u_idx += 1
                date = u_img.source_image_file.dateTaken.timestamp()
                if DEBUG:
                    print(u_img.source_image_file.dateTaken, date)
                encoding = u_img.face_encoding
                if encoding is None:
                    continue
                encoding = np.array(encoding)
                # Find the images that are closest temporally in each category.

                distances_matrix = None

                min_dist_per_category = []
                for known_id in person_keys_list:
                    known_dates = image_dates[known_id]
                    date_diffs = np.abs(known_dates - date)
                    date_diff_sort = np.argsort(date_diffs)
                    closest_date_idcs = date_diff_sort[:NUM_CLOSEST]
                    # closest_date_idcs = np.where(date_diffs < NUM_DAYS * 24 * 60 * 60)[0]
                    # if DEBUG:
                    #     print("Known dates: ")
                    #     print(known_dates[date_diff_sort[:5]])
                    #     print(date)

                    # Get the corresponding face encodings
                    encoding_list = person_encodings[known_id]
                    if DEBUG:
                        encoding_list = [np.array(i) for i in encoding_list if i is not None]
                        temporally_closest_encodings = encoding_list
                    else:
                        temporally_closest_encodings = [np.array(encoding_list[i]) for i in closest_date_idcs if encoding_list[i] is not None]


                    # print(temporally_closest_encodings[0].shape)
                    if len(temporally_closest_encodings) == 0:
                        min_dist_per_category.append(IGN_VALUE)
                        # dist_np = np.ones((1, NUM_CLOSEST)) * IGN_VALUE
                        # if distances_matrix is None:
                        #     distances_matrix = dist_np
                        # else:
                        #     distances_matrix = np.concatenate((distances_matrix, dist_np), 0)
                    else:
                    # print(encoding.shape)
                    # Compute the distance
                        if DEBUG:
                            distances = face_recognition.face_distance(encoding_list, encoding)
                        else:
                            distances = face_recognition.face_distance(temporally_closest_encodings, encoding)
                        dist_idcs = np.argsort(distances)
                        distances.sort()
                        if DEBUG:
                            print([face_ids[i] for i in dist_idcs[:10]])
                            print(distances[:10])
                        # print(dist_idcs[:10])

                        # Make an array of length NUM_CLOSEST 
                        # dist_np = np.ones((1, NUM_CLOSEST)) * IGN_VALUE
                        # dist_np[0, :len(distances)] = np.array(distances)
                        # if distances_matrix is None:
                        #     distances_matrix = dist_np
                        # else:
                        #     distances_matrix = np.concatenate((distances_matrix, dist_np), 0)

                        avg_distance = np.mean(distances[:NUM_TO_AVERAGE])
                        min_dist_per_category.append(avg_distance)

                # Set the indices for rejected fields to something huge
                if u_img.rejected_fields is not None:
                    for rejected_id in u_img.rejected_fields:
                        if rejected_id in person_ids:
                            idx = rejected_field_to_id_list_idx(rejected_id)
                            min_dist_per_category[idx] = IGN_VALUE
                            # distances_matrix[idx, :] = IGN_VALUE

                    # if ignore_person.id in u_img.rejected_fields:
                    if len(u_img.rejected_fields) > 0:
                        # Theory: If I rejected them once, it probably means I want
                        # them to be assigned *somewhere*. 
                        min_idx = np.argmin(min_dist_per_category)
                        min_val = np.min(min_dist_per_category)
                        min_dist_per_category[min_idx] = np.min((min_val, WEIGHT_THRESH-0.01))

                def smallestN_indices(a, N):
                    a = a.copy()
                    idx = a.ravel().argsort()[:N]
                    vals = a.ravel()
                    vals.sort()
                    vals = vals[:N]
                    idcs_2d = np.stack(np.unravel_index(idx, a.shape)).T
                    row_nums = idcs_2d[:, 0]
                    return row_nums, vals

                # N_SMALLEST = 25
                # min_rows, min_vals = smallestN_indices(distances_matrix, N_SMALLEST)
                # bincount = np.bincount(min_rows)
                # max_bin_size = np.max(bincount)

                # if max_bin_size <= N_SMALLEST * 0.2:
                #     person_id = ignore_person
                #     weight_val = (N_SMALLEST - max_bin_size) / N_SMALLEST
                # else:
                #     person_idx = np.argmax(bincount)
                #     person_id = Person.objects.get(id=person_keys_list[person_idx])
                #     weight_val = max_bin_size / N_SMALLEST 

                # if min_vals[0] > 0.4:
                #     person_id = ignore_person
                #     weight_val = min_vals[0]
                # else:
                #     person_idx = min_rows[0]
                #     person_id = Person.objects.get(id=person_keys_list[person_idx])
                #     weight_val = min_vals[0]

                # sleep(1)

                # Find the minimum index
                min_idx = np.argmin(min_dist_per_category)
                min_val = np.min(min_dist_per_category)
                print(min_idx, min_val)# , min_rows[0], min_vals[0])
                # if NUM_TO_AVERAGE == 1:
                #     assert min_idx == min_rows[0]
                if min_val < WEIGHT_THRESH:
                    person_id = Person.objects.get(id=person_keys_list[ min_idx ])
                    weight_val = (WEIGHT_THRESH - min_val) / WEIGHT_THRESH
                else: 
                    person_id = ignore_person
                    weight_val = min_val

                u_img.poss_ident1 = u_img.poss_ident2 = u_img.poss_ident3 = u_img.poss_ident4 = u_img.poss_ident5 = None
                u_img.weight_1 = u_img.weight_2 =u_img.weight_3 = u_img.weight_4 = u_img.weight_5 = 0
                u_img.poss_ident1 = person_id
                u_img.weight_1 = weight_val
                u_img.save()
                # exit()
            # except:
            #     print('fail')

            # face_to_image = Q(source_image_file__)

        # if DEBUG:
        #     blank_face_id = Person.objects.filter(person_name=settings.BLANK_FACE_NAME)[0]
        #     print("Blank face: ", blank_face_id)

        #     unassigned_enc, _ = self.extract_data_for_face(blank_face_id)
        #     DLIB=True
        #     DBSCAN=False

        #     if DLIB:
        #         unassigned_enc_dlib = [dlib.vector(u) for u in unassigned_enc if u is not None]
        #         s = time.time()
        #         labels = dlib.chinese_whispers_clustering(unassigned_enc_dlib[:4000], 0.3)
        #         print(list(np.bincount(labels)))
        #         print(np.max(np.bincount(labels)))
        #         print(set(labels))
        #         print(f'Elapsed is {time.time() - s} sec.')
        #     if DBSCAN:
        #         print("Starting DBSCAN...")
        #         unassigned_enc = [np.array(u) for u in unassigned_enc if u is not None]
        #         unassigned_enc = np.array(unassigned_enc)

        #         s = time.time()
        #         clt = DBSCAN(metric='euclidean', n_jobs=-1, algorithm='ball_tree', eps=0.28)
        #         clt.fit(unassigned_enc[:8000, :])
        #         print(f'Elapsed is {time.time() - s} sec.')

        #         label_ids = np.unique(clt.labels_)
        #         print(clt.labels_)
        #         print(label_ids)
        #     exit()

        # Get all the unassigned Faces
        # epoch =options['epoch']
        # print(epoch)
        # if False:
        #     criterion1 = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        #     criterion2 = ~Q(poss_ident1=None)
        #     unassigned = Face.objects.filter(criterion1 & criterion2)
        #     print(unassigned.count())
        #     # unassigned = unassigned[:15]

        #     for f in unassigned:
        #         f.poss_ident1 = f.poss_ident2 = f.poss_ident3 = f.poss_ident4 = f.poss_ident5 = None
        #         f.weight_1 = f.weight_2 =f.weight_3 = f.weight_4 = f.weight_5 = 0
        #         f.save()

        #     with open('/models/id_to_folder_map.pkl', 'rb') as ph:
        #         n = pickle.load(ph)
        #         net_class_to_id = {}
        #         for k in n.keys():
        #             network_label = int(n[k].replace('/code/MISC_DATA/face_chips/', ''))
        #             net_class_to_id[network_label] = k

        #     num_classes = len(net_class_to_id)
        #     print(net_class_to_id)


        # # criterion_ign = Q(declared_name__person_name=settings.SOFT_IGNORE_NAME)
        # with open("/code/epochs.txt", "r") as fh:
        #     data = fh.read()
        #     data = data.split('\n')
        #     try:
        #         if data[-1] == '':
        #             epoch = int(data[-2])
        #         else:
        #             epoch = int(data[-1])
        #     except:
        #         epoch = 0                    

        # # face_classifier.classify_unassigned_faces(batch_processing_size=128, do_all=True)

        # criterion_ign = Q(declared_name__person_name__in=settings.IGNORED_NAMES)
        # ignored = Face.objects.filter(criterion_ign)
        # # print(ignored.count())
        # total_todo = 127300
        # num_batches = 1500
        # batch_size = int(total_todo // num_batches) + 1
        # start_batch = epoch
        # from_end = (num_batches - start_batch) * (batch_size)
        # start = max(ignored.count() - from_end, 0)
        # print(epoch, start)
        # # ignored = ignored[start:]
        # # num_ignored = len(ignored)
        # print(ignored.count())

        # for i in range(start_batch, num_batches):
        #     adj_idx = i - start_batch
        #     print(adj_idx)
        #     print(f"CLASSIFY BATCH {i}")
           
        #     with open("/code/epochs.txt", "a") as fh:
        #         fh.write(f'{i}\n')
        #     unassigned_subset = ignored[start + adj_idx * batch_size: start + (adj_idx+1) * batch_size]
        #     print(len(unassigned_subset), start + adj_idx * batch_size, start + (adj_idx+1) * batch_size)
        #     face_classifier.classify_unassigned_faces(do_ignore=True, unassigned_external = unassigned_subset)
