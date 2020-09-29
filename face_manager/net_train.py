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
from celery import shared_task

def classify_unlabeled_faces():

    print("Starting classification")
    hidden_size = 200
    batch_size = 128

    devel = False

    train_set, val_set, ignored_face_set = face_classifier.create_dataset(settings, devel)

    with open('/code/train.set', 'wb') as fh:
        this_set = train_set
        data_list = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
        pickle.dump(data_list, fh)
    with open('/code/val.set', 'wb') as fh:
        this_set = val_set
        data_list = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
        pickle.dump(data_list, fh)

    num_classes = len(train_set.label_to_DBid)

    in_lib_mean = 0.9
    in_lib_std = 0.2
    out_of_lib_mean = 0.3
    out_of_lib_std = 0.2

    num_people = len(train_set.label_to_DBid)

    net = face_classifier.FaceNetwork(num_people, hidden_size)

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight * 10)
            m.bias.data.fill_(0.01)
        
    net.apply(init_weights)

    weights = train_set.compute_num_img_per_label()
    sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)

    weights_test = val_set.compute_num_img_per_label()
    sampler_test = data.sampler.WeightedRandomSampler(weights_test, len(weights))
    test_loader = data.DataLoader(val_set, batch_size = 1, shuffle = True ) # sampler = sampler_test)
    debug_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = False ) # sampler = sampler_test)

    ignored_loader = data.DataLoader(ignored_face_set, batch_size=batch_size, shuffle=False)

    if devel:
        epochs = 100
    else:
        epochs = 40

    def compact_loss(output):
        # print(output.shape)
        batch_mean = torch.mean(output, 1, keepdim=True)
        # print(batch_mean.shape)
        diff = output - batch_mean
        # print(diff.shape)

        loss = torch.mean(diff ** 2)

        return loss


    ccl = face_classifier.ContrastiveCenterLoss(dim_hidden = hidden_size, num_classes = num_classes)

    optimizer = optim.Adam(net.parameters(), lr = 5e-3, betas=(.9, .999))
    optimizer_c = optim.SGD(ccl.parameters(), lr= 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = max(8, epochs // 5), gamma = 0.95)

    # criterion = 
    criterion = [nn.CrossEntropyLoss(), ccl]
    
    for epoch in range(epochs):

        # print(f"{optimizer.param_groups[0]['lr']:.2e}")
        print(f"Training epoch {epoch} with learning rate {optimizer.param_groups[0]['lr']:.2e}")
        total_tr = 0
        correct_tr = 0
        s = time.time()
        net.train()
        softmax_vals_tr = []
        for i, batch_data_tr in enumerate(train_loader):
            # if i % 50 == 0: 
            #     print(f"Batch {i}/{len(train_loader)}")
            inputs_tr, labels_tr, _ = batch_data_tr
            input_noise = torch.rand(inputs_tr.shape).normal_() * torch.mean(torch.abs(inputs_tr)) * 0.1
            # input_noise = torch.rand(inputs_tr.shape).normal_() * 0.05
            input_batch = Variable(inputs_tr + input_noise)
            label_batch = Variable(labels_tr)

            optimizer.zero_grad()
            optimizer_c.zero_grad()

            hidden, logits, pred = net(input_batch)
            # loss = criterion(outputs_tr, label_batch) + ccl(outputs_tr, label_batch) #  0.15 * compact_loss(outputs_tr, label_batch)
            loss = 1.2 * criterion[0](logits, label_batch)  + 0.8 * criterion[1](label_batch, hidden)#  + 0.15 * compact_loss(logits)

            # sm = nn.Softmax(dim=1)
            # sm_out_tr = sm(out_hidden)

            # print(compact_loss(outputs, label_batch))
            loss.backward()
            optimizer.step()
            optimizer_c.step()

            # _, predicted = torch.max(outputs.data, 1)
            max_softmax_tr, predicted_tr = torch.max(pred.data, 1)
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

            # outputs, _ = net(Variable(input_t))

            _, logits, preds = net(Variable(input_t))
            pred_weight, pred_class = torch.max(preds.data, 1)

            softmax_vals_eval += pred_weight.tolist()

            n_top = 5
            _, pred_topN = preds.topk(n_top)
            pred_topN = pred_topN.t()
            batchCorrectTopN = pred_topN.eq(label_t.view(1, -1).expand_as(pred_topN))
            num_topN = batchCorrectTopN[:n_top].view(-1).float().sum(0)
            batchCorrect_t = (pred_class == label_t).sum()
            top_correct += num_topN
            total_t += label_t.size(0)
            correct_t += int(batchCorrect_t)
            

        ignore_sm = []
        for j, batch_i in enumerate(ignored_loader):
            if not devel and j > 5:
                continue
            input_i, label_i, _ = batch_i

            _, logits, preds = net(Variable(input_i))

            pred_weight, _ = torch.max(preds.data, 1)
            ignore_sm += pred_weight.tolist()

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


    net.eval()

    ignore_sm = []
    for j, batch_i in enumerate(ignored_loader):
        input_i, label_i, _ = batch_i

        _, logits, preds = net(Variable(input_i))

        max_softmax_i, _ = torch.max(preds.data, 1)
        ignore_sm += max_softmax_i.tolist()
        

    out_of_lib_mean = np.mean(np.array(ignore_sm))
    out_of_lib_std = np.std(np.array(ignore_sm))

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
        
    # After the network is all trained, we can go through and work on the data 
    # from the unassigned faces. 
    unassigned_loader =  data.DataLoader(unassigned_face_set, batch_size=batch_size, shuffle=True)


    out_of_lib_thresh = min( out_of_lib_mean + 2 * out_of_lib_std , np.mean([out_of_lib_mean, in_lib_mean]) )

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
            top5_class = pred_top5[ii, :].detach().tolist()
            top5_logits = top5_vals[ii, :].detach().tolist()
            this_face = face_models.Face.objects.get(id=db_id)
            top5_class = [label_to_DBid[x] for x in top5_class]

            if np.max(top5_logits) < out_of_lib_thresh:

                top5_class = [ignore_person] + top5_class
                top5_logits = [-1] + top5_logits

            # print(db_id)
            # print(top5_class)
            # print(top5_logits)
            # print()

            face_object = face_models.Face.objects.get(id=db_id)

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

