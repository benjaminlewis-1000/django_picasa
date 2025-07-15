#! /usr/bin/env python

from django.conf import settings
# from django.core.management.base import BaseCommand
from face_manager import models as face_models
from filepopulator import models as file_models
import sys
import time
import torch
import os
import torch.optim as optim
import torch.utils.data as data
import random
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import io
from PIL import Image
import pickle
import face_classifier
from sklearn.neighbors import KNeighborsClassifier as knn_class
from collections import Counter, OrderedDict
torch.backends.nnpack.enabled = False

def classify_unlabeled_faces():

    print("Starting KNN classification")
    hidden_size = 200
    batch_size = 128
    num_neighbor = 40

    devel = False

    train_set, val_set, ign_set = face_classifier.create_dataset(settings, devel)
    # print(len(train_set))

    train_dp = [x.numpy() for x in train_set.data_points]
    val_dp = [x.numpy() for x in val_set.data_points]
    ign_dp = [x.numpy() for x in ign_set.data_points]

    # print(train_dp[:5])

    knn = knn_class(n_neighbors = num_neighbor)
    knn.fit(train_dp, train_set.labels)

    print("KNN Fit!")
    tops_dist, tops_class = knn.kneighbors(val_dp[:10], n_neighbors = num_neighbor)
    # print(np.min(tops_dist, 1))
    # print(np.mean(tops_dist, 1))

    def sort_by_frequency(array):
        result = [item for items, c in Counter(array).most_common() 
                                      for item in [items] * c] 
        result = list(OrderedDict.fromkeys(result))
        return result

    for i, t in enumerate(tops_class):
        # print("T: ")
        # print(t, type(t))
        labels=[]
        for each in t:
            lab = train_set.labels[each]
            labels.append(lab)
            # print("Ech", each, lab)
        # print(np.unique(labels), val_set.labels[i], val_set.face_id[i], val_set.label_to_DBid[val_set.labels[i]])
        srt = sort_by_frequency(labels)
        freq = []
        for s in srt:
            f = len(np.where(np.array(labels) == s)[0])
            freq.append(f)
        print(labels, srt, freq)



    # print("Ignore Fit!")
    # # exit()
    # tops_dist, tops_class = knn.kneighbors(ign_dp[:10], n_neighbors = num_neighbor)
    # # print(tops_class)
    # # print(tops_dist)
    # print(np.min(tops_dist, 1))
    # print(np.mean(tops_dist, 1))
    # for i, t in enumerate(tops_class):
    #     # print("T: ")
    #     # print(t, type(t))
    #     labels=[]
    #     for each in t:
    #         # print(each, type(each))
    #         lab = train_set.labels[each]
    #         print(lab)
    #         labels.append(lab)
    #     print(np.unique(labels), ign_set.face_id[i])
    # print(ten_tops)


    # top_1 = 0
    # top_5 = 0 
    # total = 0 
    # print("Hr")
    # for i, val in enumerate(val_dp):
    #     tops_dist, tops_class = knn.kneighbors(val.reshape(1, -1), n_neighbors = num_neighbor)
    #     labels=[]
    #     truth = val_set.labels[i]
    #     for each in tops_class[0]:
    #         # print(each, type(each))
    #         lab = train_set.labels[each]
    #         # print(lab)
    #         labels.append(lab)

    #     total += 1
    #     labels = sort_by_frequency(labels)
    #     if labels[0] == truth:
    #         top_1 += 1
    #     if truth in labels[:5]:
    #         top_5 += 1

    #     print(f"Acc: {top_1 / total * 100 :.2f}%, {top_5  / total * 100 :.2f}%, {i}/{len(val_dp)}       \r", end='')
    # print(f"Acc: {top_1 / total * 100 :.2f}%, {top_5  / total * 100 :.2f}%, {i}/{len(val_dp)}       ")
        
    # with open('/code/train.set', 'wb') as fh:
    #     this_set = train_set
    #     data = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
    #     pickle.dump(data, fh)
    # with open('/code/val.set', 'wb') as fh:
    #     this_set = val_set
    #     data = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
    #     pickle.dump(data, fh)

    # num_classes = len(train_set.label_to_DBid)

    
    unassigned_filter = face_models.Person.objects.filter(person_name=settings.BLANK_FACE_NAME)
    ignore_person = face_models.Person.objects.filter(person_name='.ignore')[0]
    # print(ignore_person)
    unassigned_faces = face_models.Face.objects.filter(declared_name=unassigned_filter[0].id)
    unassigned_face_set = face_classifier.FaceLabelSet()
    unassigned_face_set.add_person('ignore', unassigned_filter[0].id)
    for i, un in enumerate(unassigned_faces):
        unassigned_face_set.add_datapoint(unassigned_filter[0].id, un.face_encoding, un.id)

        if i > 100 and devel:
            break
        # if len(unassigned_face_set) > 4000:
        #     break


    unassign_dp = [x.numpy() for x in unassigned_face_set.data_points]
    
    label_to_DBid = train_set.label_to_DBid
    for lbl in label_to_DBid.keys():
        dbid = label_to_DBid[lbl]
        person_obj = face_models.Person.objects.filter(id=dbid)[0]
        label_to_DBid[lbl] = person_obj

    for i, val in enumerate(unassign_dp):
        print(f"{i}/{len(unassigned_face_set)}")
        tops_dist, tops_class = knn.kneighbors(val.reshape(1, -1), n_neighbors = num_neighbor)
        labels=[]
        # truth = val_set.labels[i]
        for each in tops_class[0]:
            # print(each, type(each))
            lab = train_set.labels[each]
            # print(lab)
            labels.append(lab)

        # total += 1
        srt = sort_by_frequency(labels)
        freq = []
        for s in srt:
            f = len(np.where(np.array(labels) == s)[0])
            freq.append(f)
        db_id = unassigned_face_set.face_id[i]

        srt_to_class = [label_to_DBid[x] for x in srt]
        # print(srt[:5], freq[:5], db_id)

        face_object = face_models.Face.objects.get(id=db_id)

        face_rejects = face_object.rejected_fields
        if face_rejects is None:
            face_rejects = []
    
        for rej in face_rejects:
            if rej in srt_to_class:
                idx = srt_to_class.index(rej)
                srt_to_class.pop(idx)
                freq.pop(idx)

        freq[1:5] = freq[1:5] / np.sum(freq[1:5])


        if face_object.declared_name.person_name == settings.BLANK_FACE_NAME:
            # Make it safe to work on this while the frontend is 
            # running. In other words, if it's trained and I assigned
            # it a face in the middle of its training, it shouldn't 
            # assign any possibilities to it.

            # print(face_object.declared_name.person_name)
            # exit()

            # for i in range(5):
            face_object.poss_ident1 = srt_to_class[0]
            face_object.weight_1 = 1
            if len(srt_to_class) > 1:
                face_object.poss_ident2 = srt_to_class[1]
                face_object.weight_2 = freq[1]
            else:
                face_object.poss_ident2 = None
                face_object.weight_2 = 0
            if len(srt_to_class) > 2:
                face_object.poss_ident3 = srt_to_class[2]
                face_object.weight_3 = freq[2]
            else:
                face_object.poss_ident3 = None
                face_object.weight_3 = 0
            if len(srt_to_class) > 3:
                face_object.poss_ident4 = srt_to_class[3]
                face_object.weight_4 = freq[3]
            else:
                face_object.poss_ident4 = None
                face_object.weight_4 = 0
            if len(srt_to_class) > 4:
                face_object.poss_ident5 = srt_to_class[4]
                face_object.weight_5 = freq[4]
            else:
                face_object.poss_ident5 = None
                face_object.weight_5 = 0
            face_object.save()

    exit()
    # After the network is all trained, we can go through and work on the data 
    # from the unassigned faces. 
    # unassigned_loader =  data.DataLoader(unassigned_face_set, batch_size=batch_size, shuffle=True)


    # out_of_lib_thresh = min( out_of_lib_mean + 2 * out_of_lib_std , np.mean([out_of_lib_mean, in_lib_mean]) )

    print("Classifying unidentified faces.")
    for j, batch_u in enumerate(unassigned_loader):
        input_u, label_u, face_ids = batch_u

        _, logits, preds = net(Variable(input_u))

        max_softmax_u, pred_u = torch.max(preds.data, 1)
        top5_vals, pred_top5 = preds.topk(5)
        # print(pred_top5.shape)

        face_ids = face_ids.tolist()
        for ii in range(len(face_ids)):
            db_id = face_ids[ii]
            print(f"Classifying {db_id}")
            this_face = face_models.Face.objects.get(id=db_id)
            face_rejects = this_face.rejected_fields
            if face_rejects is None:
                face_rejects = []
            topk_vals, topk_preds = preds[ii].topk(5 + len(face_rejects))
            topk_vals = topk_vals.detach().tolist()
            topk_preds = topk_preds.detach().tolist()
    
            for rej in face_rejects:
                if rej in topk_preds:
                    idx = topk_preds.index(rej)
                    topk_preds.pop(idx)
                    topk_vals.pop(idx)

#            top5_class = pred_top5[ii, :].detach().tolist()
#            top5_logits = top5_vals[ii, :].detach().tolist()
            top5_logits = topk_vals
            top5_class = [label_to_DBid[x] for x in topk_preds]

            if np.max(top5_logits) < out_of_lib_thresh:

                top5_class = [ignore_person] + top5_class
                top5_logits = [-1] + top5_logits


            face_object = face_models.Face.objects.get(id=db_id)

            if face_object.declared_name.person_name == settings.BLANK_FACE_NAME:
                # Make it safe to work on this while the frontend is 
                # running. In other words, if it's trained and I assigned
                # it a face in the middle of its training, it shouldn't 
                # assign any possibilities to it.

                # print(face_object.declared_name.person_name)
                # exit()

                # for i in range(5):
                face_object.poss_ident1 = top5_class[0]
                face_object.weight_1 = top5_logits[0]
                face_object.poss_ident2 = top5_class[1]
                face_object.weight_2 = top5_logits[1]
                face_object.poss_ident3 = top5_class[2]
                face_object.weight_3 = top5_logits[2]
                face_object.poss_ident4 = top5_class[3]
                face_object.weight_4 = top5_logits[3]
                face_object.poss_ident5 = top5_class[4]
                face_object.weight_5 = top5_logits[4]
                face_object.save()

