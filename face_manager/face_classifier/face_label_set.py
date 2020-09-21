#! /usr/ben/env python

import torch.utils.data as data
import numpy as np
import torch

class FaceLabelSet(data.Dataset):
    def __init__(self):
        super(FaceLabelSet, self).__init__()

        self.label_to_DBid = {}
        self.DBid_to_label = {}
        self.DBid_to_name = {}
        self.label_idx = 0

        self.labels = []
        self.data_points = []

        self.counts_per_label = {}
        self.weight = []
        self.face_id = []

    def add_person(self, person_name, person_id):
        person_id = int(person_id)
        person_name = str(person_name)
        self.label_to_DBid[self.label_idx] = person_id
        self.DBid_to_label[person_id] = self.label_idx
        self.DBid_to_name[person_id] = person_name
        self.label_idx += 1

    def add_datapoint(self, person_id, data, face_id = -1):
        person_id = int(person_id)
        face_id = int(face_id)
        label = self.DBid_to_label[person_id]
        if type(data) == type(None):
            return
        assert len(data) == 128
        self.labels.append(label)
        self.face_id.append(face_id)
        self.data_points.append(torch.Tensor(data))

        assert len(self.labels) == len(self.data_points)

    def compute_num_img_per_label(self):

        num_datapoints = len(self.data_points)

        for l in list(set(self.labels)):
            subset = [x for x in self.labels if x == l]
            weight_to_assign = num_datapoints / len(subset)
            # Weight it back in favor of heavier classes. 
            weight_to_assign = weight_to_assign * np.log10(len(subset)) / np.log10(5)
            self.counts_per_label[l] = weight_to_assign

        self.weight = [0] * num_datapoints
        for i in range(num_datapoints):
            label = self.labels[i]
            w = self.counts_per_label[label]
            self.weight[i] = w

        return self.weight

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        data = self.data_points[index]
        face_db_id = self.face_id[index]
        return data, self.labels[index], face_db_id
