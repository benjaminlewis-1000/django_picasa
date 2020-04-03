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
import numpy as np
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
        self.face_id = []

    def add_person(self, person_name, person_id):
        self.label_to_DBid[self.label_idx] = person_id
        self.DBid_to_label[person_id] = self.label_idx
        self.DBid_to_name[person_id] = person_name
        self.label_idx += 1

    def add_datapoint(self, person_id, data, face_id = -1):
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
        face_db_id = self.face_id[index]
        return data, self.labels[index], face_db_id

def create_dataset():

    ignore_filter = face_models.Person.objects.filter(person_name='.realignore')
    ignored_faces = face_models.Face.objects.filter(declared_name=ignore_filter[0].id)

    ignored_face_set = FaceLabelSet()
    ignored_face_set.add_person('ignore', 1)
    for ignf in ignored_faces:
        ignored_face_set.add_datapoint(1, ignf.face_encoding)

    print("Ignored face length: ", len(ignored_face_set))

    train_set = FaceLabelSet()
    val_set = FaceLabelSet()
    # Get all the faces that have an assigned name and that
    # have enough faces for us to be interested in training.
    # How to filter on foreign key: 
    # https://stackoverflow.com/a/6205303/3158519
    people_filter = face_models.Person.objects.annotate(num_face=Count('face_declared'))\
        .filter(num_face__gt=settings.FACE_NUM_THRESH)\
        .exclude(person_name__in=settings.IGNORED_NAMES)

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

    return train_set, val_set, ignored_face_set

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

        train_set, val_set, ignored_face_set = create_dataset()
        print(len(train_set))

        in_lib_mean = 0.9
        in_lib_std = 0.2
        out_of_lib_mean = 0.3
        out_of_lib_std = 0.2

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

        ignored_loader = data.DataLoader(ignored_face_set, batch_size=128, shuffle=False)

        epochs = 500
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr = 1e-3, betas=(.9, .999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = max(1, epochs // 40), gamma = 0.95)

        def compact_loss(output, target):
            # print(output.shape)
            batch_mean = torch.mean(output, 1, keepdim=True)
            # print(batch_mean.shape)
            diff = output - batch_mean
            # print(diff.shape)

            loss = torch.mean(diff ** 2)

            return loss

        for epoch in range(epochs):

            print(f"{optimizer.param_groups[0]['lr']:.2e}")
            total_tr = 0
            correct_tr = 0
            s = time.time()
            net.train()
            softmax_vals_tr = []
            for i, batch_data_tr in enumerate(train_loader):
                inputs_tr, labels_tr, _ = batch_data_tr
                input_noise = torch.rand(inputs_tr.shape).normal_() * torch.mean(torch.abs(inputs_tr)) * 0.1
                input_batch = Variable(inputs_tr + input_noise)
                label_batch = Variable(labels_tr)

                optimizer.zero_grad()

                outputs_tr = net(input_batch)
                loss = criterion(outputs_tr, label_batch) + 0.15 * compact_loss(outputs_tr, label_batch)

                sm = nn.Softmax(dim=1)
                sm_out_tr = sm(outputs_tr)

                # print(compact_loss(outputs, label_batch))
                loss.backward()
                optimizer.step()

                # _, predicted = torch.max(outputs.data, 1)
                max_softmax_tr, predicted_tr = torch.max(sm_out_tr.data, 1)
                # print(max_softmax, predicted.tolist())
                # print(outputs.shape, predicted.shape, max_softmax.shape)
                softmax_vals_tr += max_softmax_tr.tolist()

                batchCorrect_tr = (predicted_tr == label_batch).sum()
                # print(predicted)
                # print(label_batch)
                # print(predicted == label_batch)
                # print((predicted == label_batch).sum())
                total_tr += label_batch.size(0)
                correct_tr += int(batchCorrect_tr)

            total_t = 0
            correct_t = 0
            top_correct = 0
            net.eval()
            softmax_vals_eval = []
            evaluation_loader = test_loader
            if evaluation_loader == debug_loader:
                print("Warning! Debug loader")
            for j, batch_t in enumerate(evaluation_loader):
                input_t, label_t, _ = batch_t

                outputs = net(Variable(input_t))
                _, predicted = torch.max(outputs.data, 1)

                sm = nn.Softmax(dim=1)
                sm_out = sm(outputs)
                max_softmax, _ = torch.max(sm_out.data, 1)
                softmax_vals_eval += max_softmax.tolist()

                n_top = 5
                _, pred_topN = outputs.topk(n_top)
                pred_topN = pred_topN.t()
                batchCorrectTopN = pred_topN.eq(label_t.view(1, -1).expand_as(pred_topN))
                num_topN = batchCorrectTopN[:n_top].view(-1).float().sum(0)
                batchCorrect_t = (predicted == label_t).sum()
                top_correct += num_topN
                total_t += label_t.size(0)
                correct_t += int(batchCorrect_t)


            ignore_sm = []
            for j, batch_i in enumerate(ignored_loader):
                input_i, label_i, _ = batch_i

                outputs_i = net(Variable(input_i))

                sm = nn.Softmax(dim=1)
                sm_out_i = sm(outputs_i)
                max_softmax_i, _ = torch.max(sm_out_i.data, 1)
                ignore_sm += max_softmax_i.tolist()

            in_lib_mean = np.mean(np.array(softmax_vals_tr))
            in_lib_std = np.std(np.array(softmax_vals_tr))
            mean_max_sm_val = np.mean(np.array(softmax_vals_eval))
            st2 = np.std(np.array(softmax_vals_eval))
            out_of_lib_mean = np.mean(np.array(ignore_sm))
            out_of_lib_std = np.std(np.array(ignore_sm))

            print(f"Epoch {epoch}, Train acc: {correct_tr/total_tr*100:.2f}%, Val acc: {correct_t / total_t*100:.2f}%, ", \
                f"Top {n_top}: {top_correct / total_t * 100:.2f}%, {time.time() - s:.2f} sec, ", \
                f" Mean Softmax {in_lib_mean:.2f}|{mean_max_sm_val:.2f}|{out_of_lib_mean:.2f}  || ",\
                f"{in_lib_std:.2f}|{st2:.2f}|{out_of_lib_std:.2f}")

            scheduler.step()

        label_to_DBid = train_set.label_to_DBid
        for lbl in label_to_DBid.keys():
            dbid = label_to_DBid[lbl]
            person_obj = face_models.Person.objects.filter(id=dbid)[0]
            label_to_DBid[lbl] = person_obj
        # DBid_to_label = train_loader.DBid_to_label
        # DBid_to_name = train_loader.DBid_to_name
        del(train_loader)
        del(test_loader)
        del(val_set)
        del(debug_loader)
        del(train_set)
        del(ignored_loader)
        del(ignored_face_set)
        print(label_to_DBid)

        unassigned_filter = face_models.Person.objects.filter(person_name=settings.BLANK_FACE_NAME)
        ignore_person = face_models.Person.objects.filter(person_name='.ignore')[0]
        print(ignore_person)
        unassigned_faces = face_models.Face.objects.filter(declared_name=unassigned_filter[0].id)
        unassigned_face_set = FaceLabelSet()
        unassigned_face_set.add_person('ignore', unassigned_filter[0].id)
        for un in unassigned_faces:
            unassigned_face_set.add_datapoint(unassigned_filter[0].id, un.face_encoding, un.id)
            # if len(unassigned_face_set) > 4000:
            #     break
            
        # After the network is all trained, we can go through and work on the data 
        # from the unassigned faces. 
        unassigned_loader =  data.DataLoader(unassigned_face_set, batch_size=128, shuffle=True)


        out_of_lib_thresh = min( out_of_lib_mean + 2 * out_of_lib_std , np.mean([out_of_lib_mean, in_lib_mean]) )

        for j, batch_u in enumerate(unassigned_loader):
            input_u, label_u, face_ids = batch_u

            outputs_u = net(Variable(input_u))

            sm = nn.Softmax(dim=1)
            sm_out_u = sm(outputs_u)
            max_softmax_u, pred_u = torch.max(sm_out_u.data, 1)
            top5_vals, pred_top5 = sm_out_u.topk(5)
            print(pred_top5.shape)

            face_ids = face_ids.tolist()
            for ii in range(len(face_ids)):
                db_id = face_ids[ii]
                top5_class = pred_top5[ii, :].detach().tolist()
                top5_logits = top5_vals[ii, :].detach().tolist()
                this_face = face_models.Face.objects.get(id=db_id)
                top5_class = [label_to_DBid[x] for x in top5_class]

                if np.max(top5_logits) < out_of_lib_thresh:

                    top5_class = [ignore_person] + top5_class
                    top5_logits = [-1] + top5_logits

                print(db_id)
                print(top5_class)
                print(top5_logits)
                print()
