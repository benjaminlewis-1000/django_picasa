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
from celery import task
from sklearn.neighbors import KNeighborsClassifier as knn_class
from collections import Counter, OrderedDict

def classify_unlabeled_faces():

    print("Starting KNN classification")
    hidden_size = 200
    batch_size = 128

    devel = False

    train_set, val_set, ign_set = face_classifier.create_dataset(settings, devel)
    # print(len(train_set))

    train_dp = [x.numpy() for x in train_set.data_points]
    val_dp = [x.numpy() for x in val_set.data_points]
    ign_dp = [x.numpy() for x in ign_set.data_points]

    # print(train_dp[:5])

    knn = knn_class(n_neighbors = 20)
    knn.fit(train_dp, train_set.labels)

    print("KNN Fit!")
    tops_dist, tops_class = knn.kneighbors(val_dp[:10], n_neighbors = 20)
    print(np.min(tops_dist, 1))
    print(np.mean(tops_dist, 1))
    for i, t in enumerate(tops_class):
        # print("T: ")
        # print(t, type(t))
        labels=[]
        for each in t:
            # print(each, type(each))
            lab = train_set.labels[each]
            labels.append(lab)
        print(np.unique(labels), val_set.labels[i], val_set.face_id[i], val_set.label_to_DBid[val_set.labels[i]])

    print("Ignore Fit!")
    tops_dist, tops_class = knn.kneighbors(ign_dp[:10], n_neighbors = 20)
    # print(tops_class)
    # print(tops_dist)
    print(np.min(tops_dist, 1))
    print(np.mean(tops_dist, 1))
    for i, t in enumerate(tops_class):
        # print("T: ")
        # print(t, type(t))
        labels=[]
        for each in t:
            # print(each, type(each))
            lab = train_set.labels[each]
            print(lab)
            labels.append(lab)
        print(np.unique(labels), ign_set.face_id[i])
    # print(ten_tops)

    def sort_by_frequency(array):
        result = [item for items, c in Counter(array).most_common() 
                                      for item in [items] * c] 
        result = list(OrderedDict.fromkeys(result))
        return result

    top_1 = 0
    top_5 = 0 
    total = 0 
    for i, val in enumerate(val_dp):
        tops_dist, tops_class = knn.kneighbors(val.reshape(1, -1), n_neighbors = 20)
        labels=[]
        truth = val_set.labels[i]
        for each in tops_class[0]:
            # print(each, type(each))
            lab = train_set.labels[each]
            # print(lab)
            labels.append(lab)

        total += 1
        labels = sort_by_frequency(labels)
        if labels[0] == truth:
            top_1 += 1
        if truth in labels[:5]:
            top_5 += 1

        print(f"Acc: {top_1 / total * 100 :.2f}%, {top_5  / total * 100 :.2f}%, {i}/{len(val_dp)}       \r", end='')
        
    # with open('/code/train.set', 'wb') as fh:
    #     this_set = train_set
    #     data = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
    #     pickle.dump(data, fh)
    # with open('/code/val.set', 'wb') as fh:
    #     this_set = val_set
    #     data = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
    #     pickle.dump(data, fh)

    # num_classes = len(train_set.label_to_DBid)