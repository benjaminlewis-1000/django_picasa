#! /usr/bin/env python


from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from face_manager import models as face_models
from filepopulator import models as file_models
import sys
import time
import torch
import torch.optim as optim
import torch.utils.data as data
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

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

    def add_person(self, person_name, person_id):
        self.label_to_DBid[self.label_idx] = person_id
        self.DBid_to_label[person_id] = self.label_idx
        self.DBid_to_name[person_id] = person_name
        self.label_idx += 1

    def add_datapoint(self, person_id, data):
        label = self.DBid_to_label[person_id]
        if type(data) == type(None):
            return
        assert len(data) == 128
        self.labels.append(label)
        self.data_points.append(torch.Tensor(data))

        assert len(self.labels) == len(self.data_points)

    def compute_num_img_per_label(self):

        num_datapoints = len(self.data_points)

        for l in list(set(self.labels)):
            subset = [x for x in self.labels if x == l]
            self.counts_per_label[l] = num_datapoints / len(subset)

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
        return data, self.labels[index]

def create_dataset():
    # Get all the faces that have an assigned name and that
    # have enough faces for us to be interested in training.
    # How to filter on foreign key: 
    # https://stackoverflow.com/a/6205303/3158519
    people_filter = face_models.Person.objects.annotate(num_face=Count('face_declared'))\
        .filter(num_face__gt=settings.FACE_NUM_THRESH)\
        .exclude(person_name__in=settings.IGNORED_NAMES)

    train_set = FaceLabelSet()
    val_set = FaceLabelSet()

    print(len(people_filter))
    
    # Now to put things in a dataset! 
    for p in people_filter:
        train_set.add_person(p.person_name, p.id)
        val_set.add_person(p.person_name, p.id)

        faces_of_person = face_models.Face.objects.filter(declared_name=p.id)
        print(p.person_name, p.id, len(faces_of_person))
        # print(type(faces_of_person))

        # nn = 0
        num_train = int(len(faces_of_person) * 0.8)
        indices = list(range(len(faces_of_person)))
        random.shuffle(indices)
        for ii in range(0, num_train):
            idx = indices[ii]
            train_set.add_datapoint(p.id, faces_of_person[idx].face_encoding)
        for jj in range(num_train, len(faces_of_person)):
            idx = indices[jj]
            val_set.add_datapoint(p.id, faces_of_person[idx].face_encoding)

            # nn += 1
            # if nn > 20:
                # break

    return train_set, val_set

class FaceNetwork(nn.Module):
    def __init__(self, n_classes):
        super(FaceNetwork, self).__init__()
        
        # Input is 128-dimensional vector
        self.fc1 = nn.Linear(128, 512)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(512, 256)
        self.tanh2 = nn.Tanh()
        # self.fc3 = nn.Linear(256, 512)
        # self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(256, n_classes)
        self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        # x = self.fc3(x)
        # x = self.tanh3(x)
        x = self.fc4(x)
        # x = self.sm(x)

        return x

class Command(BaseCommand):
    
    def handle(self, *args, **options):

        train_set, val_set = create_dataset()
        print(len(train_set))

        num_people = len(train_set.label_to_DBid)

        net = FaceNetwork(num_people)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight * 10)
                m.bias.data.fill_(0.01)
            
        net.apply(init_weights)

        weights = train_set.compute_num_img_per_label()
        sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False, sampler=sampler)

        weights_test = val_set.compute_num_img_per_label()
        sampler_test = data.sampler.WeightedRandomSampler(weights_test, len(weights))
        test_loader = data.DataLoader(val_set, batch_size = 1, shuffle = True ) # sampler = sampler_test)
        debug_loader = data.DataLoader(train_set, batch_size = 128, shuffle = False ) # sampler = sampler_test)

        epochs = 500
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr = 1e-3, betas=(.9, .999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = epochs // 40, gamma = 0.95)
        print(train_set[0])

        for epoch in range(epochs):

            print(f"{optimizer.param_groups[0]['lr']:.2e}")
            total = 0
            correct = 0
            s = time.time()
            net.train()
            for i, batch_data in enumerate(train_loader):
                inputs, labels = batch_data
                input_noise = torch.rand(inputs.shape).normal_() * torch.mean(torch.abs(inputs)) * 0.2
                input_batch = Variable(inputs + input_noise)
                label_batch = Variable(labels)

                # l = [int(x) for x in label_batch]
                # l = list(set(l))
                # l.sort()
                # print(l)

                optimizer.zero_grad()

                outputs = net(input_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)

                batchCorrect = (predicted == label_batch).sum()
                # print(predicted)
                # print(label_batch)
                # print(predicted == label_batch)
                # print((predicted == label_batch).sum())
                total += label_batch.size(0)
                correct += int(batchCorrect)

            total_t = 0
            correct_t = 0
            top_correct = 0
            net.eval()
            evaluation_loader = test_loader
            if evaluation_loader == debug_loader:
                print("Warning! Debug loader")
            for j, batch_t in enumerate(evaluation_loader):
                input_t, label_t = batch_t

                # with torch.no_grad():
                outputs = net(Variable(input_t))
                _, predicted = torch.max(outputs.data, 1)

                n_top = 5
                _, pred_topN = outputs.topk(n_top)
                pred_topN = pred_topN.t()
                batchCorrectTopN = pred_topN.eq(label_t.view(1, -1).expand_as(pred_topN))
                num_topN = batchCorrectTopN[:n_top].view(-1).float().sum(0)
                batchCorrect_t = (predicted == label_t).sum()
                top_correct += num_topN
                total_t += label_t.size(0)
                correct_t += int(batchCorrect_t)

            print(f"Epoch {epoch}, Train acc: {correct/total*100:.2f}%, Val acc: {correct_t / total_t*100:.2f}%, ", \
                f"Top {n_top}: {top_correct / total_t * 100:.2f}%, {time.time() - s} sec")

            scheduler.step()

        # print(net)
        # out = net(f)
        # print(out)