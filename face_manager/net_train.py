#! /usr/bin/env python

from celery import shared_task
from django.conf import settings
from face_manager import models as face_models
from filepopulator import models as file_models
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import face_classifier
import io
import numpy as np
import os
import pickle
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
torch.backends.nnpack.enabled = False

def classify_unlabeled_faces():

    print("Starting classification")
    hidden_size = 300
    batch_size = 128
    which_features = 'short'
    # which_features = 'long'

    devel = False

    train_set, val_set, ignored_face_set = face_classifier.create_dataset(settings, devel, which_features)

    train_feat = np.array([np.array(x) for x in train_set.data_points])

    scaler = StandardScaler()
    scaler.fit(train_feat)

    def transform_and_reassign(features):
            
        scaled_feat = scaler.transform(features)
        scaled_feat = [torch.Tensor(x) for x in scaled_feat]
        return scaled_feat

    train_transformed = transform_and_reassign(train_feat)
    train_set.data_points = train_transformed
    val_feat = np.array([np.array(x) for x in val_set.data_points])
    val_transformed = transform_and_reassign(val_feat)
    val_set.data_points = val_transformed
    ign_feat = np.array([np.array(x) for x in ignored_face_set.data_points])
    ign_transformed = transform_and_reassign(ign_feat)
    ignored_face_set.data_points = ign_transformed

    print("Built dataset")

    """
    b = [x.numpy() for x in train_set.data_points]
    train_data = np.vstack(b)
    b = [x.numpy() for x in ignored_face_set.data_points]
    ignored_data = np.vstack(b)
    b = [x.numpy() for x in val_set.data_points]
    val_data = np.vstack(b)


    train_labels_file = f'/code/train_labels_{which_features}.npy'
#    np.save(train_labels_file, train_set.labels)
    train_data_file = f'/code/train_data_{which_features}.npy'
#    np.save(train_data_file, train_data)

    ignored_labels_file = f'/code/ignored_labels_{which_features}.npy'
#    np.save(ignored_labels_file, ignored_face_set.labels)
    ignored_data_file = f'/code/ignored_data_{which_features}.npy'
#    np.save(ignored_data_file, ignored_data)
    
    val_labels_file = f'/code/val_labels_{which_features}.npy'
#    np.save(val_labels_file, val_set.labels)
    val_data_file = f'/code/val_data_{which_features}.npy'
#    np.save(val_data_file, val_data)

#    exit()
   """    

#    with open('/code/train.set', 'wb') as fh:
#        this_set = train_set
#        data_list = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
#        pickle.dump(data_list, fh)
#    with open('/code/val.set', 'wb') as fh:
#        this_set = val_set
#        data_list = [this_set.labels, this_set.data_points, this_set.weight, this_set.face_id]
#        pickle.dump(data_list, fh)

    print("Dataset constructed")
    num_classes = len(train_set.label_to_DBid)

    in_lib_mean = 0.9
    in_lib_std = 0.2
    out_of_lib_mean = 0.3
    out_of_lib_std = 0.2

    num_people = len(train_set.label_to_DBid)

    net = face_classifier.FaceNetwork(num_people, hidden_size, which_features)

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight * 10)
            m.bias.data.fill_(0.01)
        
    net.apply(init_weights)

#    weights = train_set.compute_num_img_per_label()
#    sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
    #train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

#    weights_test = val_set.compute_num_img_per_label()
#    sampler_test = data.sampler.WeightedRandomSampler(weights_test, len(weights))
    test_loader = data.DataLoader(val_set, batch_size = 1, shuffle = True ) # sampler = sampler_test)
    debug_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = False ) # sampler = sampler_test)

    ignored_loader = data.DataLoader(ignored_face_set, batch_size=batch_size, shuffle=False)

    epochs = 25 

    '''
    def compact_loss(output):
        # print(output.shape)
        batch_mean = torch.mean(output, 1, keepdim=True)
        # print(batch_mean.shape)
        diff = output - batch_mean
        # print(diff.shape)

        loss = torch.mean(diff ** 2)

        return loss
    '''

#    ccl = face_classifier.ContrastiveCenterLoss(dim_hidden = hidden_size, num_classes = num_classes)

    optimizer = optim.Adam(net.parameters(), lr = 5e-3, betas=(.9, .999))
#    optimizer_c = optim.SGD(ccl.parameters(), lr= 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = max(8, epochs // 6), gamma = 0.85)

    criterion = nn.CrossEntropyLoss()
    # criterion = [nn.CrossEntropyLoss(), ccl]
    
    for epoch in range(epochs):
        print(f"Training epoch {epoch} with learning rate {optimizer.param_groups[0]['lr']:.2e}")
        total_tr = 0
        correct_tr = 0
        s = time.time()
        net.train()
        softmax_vals_tr = []
        pt_0, pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7, pt_8, pt_9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, batch_data_tr in enumerate(train_loader):
            # if i % 50 == 0: 
            #     print(f"Batch {i}/{len(train_loader)}")
            s1 = time.time()
            inputs_tr, labels_tr, _ = batch_data_tr
#            input_noise = torch.rand(inputs_tr.shape).normal_() * torch.mean(torch.abs(inputs_tr)) * 0.1
            # input_noise = torch.rand(inputs_tr.shape).normal_() * 0.05
#            input_batch = Variable(inputs_tr + input_noise)

            input_batch = (inputs_tr)
            label_batch = (labels_tr)
            el_0 = time.time() - s1
            pt_0 += el_0

            optimizer.zero_grad()
#            optimizer_c.zero_grad()

            s1 = time.time()
            logits = net(input_batch)
            pred = F.softmax(logits, dim=1)
            el_1 = time.time() - s1
            pt_1 += el_1

            s1 = time.time()
            # loss = criterion(outputs_tr, label_batch) + ccl(outputs_tr, label_batch) #  0.15 * compact_loss(outputs_tr, label_batch)

            loss =  criterion(logits, label_batch) #  + 0.08 * criterion[1](label_batch, hidden)#  + 0.15 * compact_loss(logits)
            el_2 = time.time() - s1
            pt_2 += el_2

            if i == 0:
                print(f'Loss done, time was {time.time() - s} sec')
            # sm = nn.Softmax(dim=1)
            # sm_out_tr = sm(out_hidden)

            # print(compact_loss(outputs, label_batch))
            s1 = time.time()
            loss.backward()
            el_3 = time.time() - s1
            pt_3 += el_3

            s1 = time.time()
            optimizer.step()
            el_4 = time.time() - s1
            pt_4 += el_4
            # optimizer_c.step()

            # _, predicted = torch.max(outputs.data, 1)

            s1 = time.time()
            max_softmax_tr, predicted_tr = torch.max(pred.data, 1)
            el_5 = time.time() - s1
            pt_5 += el_5
            # print(max_softmax, predicted.tolist())
            # print(outputs.shape, predicted.shape, max_softmax.shape)
   #         softmax_vals_tr += max_softmax_tr.tolist()

            s1 = time.time()
            batchCorrect_tr = (predicted_tr == label_batch).sum()
            el_6 = time.time() - s1
            pt_6 += el_6
            # print(predicted)
            # print(label_batch)
            # print(predicted == label_batch)
            # print((predicted == label_batch).sum())

            s1 = time.time()
            total_tr += label_batch.size(0)
            el_7 = time.time() - s1
            pt_7 += el_7

            s1 = time.time()
            correct_tr += int(batchCorrect_tr)
            el_8 = time.time() - s1
            pt_8 += el_8

            del loss
            del inputs_tr 
            del labels_tr 
            del input_batch 
            del label_batch 
            del batchCorrect_tr
            del logits
            del pred
        
        print(f'Train done, time was {time.time() - s} sec, len is {len(train_loader)}')
        print(f'{pt_0:.2f}, {pt_1:.2f}, {pt_2:.2f}, {pt_3:.2f}, {pt_4:.2f}, {pt_5:.2f}, {pt_6:.2f}, {pt_7:.2f}, {pt_8:.2f} ')
        print(f'{total_tr}, {correct_tr}')

#        print(f"Epoch {epoch}, Train acc: {correct_tr/total_tr*100:.2f}%, Val acc: {correct_t / total_t*100:.2f}%, ", \
#            f"Top {n_top}: {top_correct / total_t * 100:.2f}%, {time.time() - s:.2f} sec, ", \
#            f" Mean Softmax {in_lib_mean:.2f}|{mean_max_sm_val:.2f}|{out_of_lib_mean:.2f}  || ",\
#            f"{in_lib_std:.2f}|{st2:.2f}|{out_of_lib_std:.2f}")
#
#        scheduler.step()
#
#    if True:
        """
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

#        in_lib_mean = np.mean(np.array(softmax_vals_tr))
#        in_lib_std = np.std(np.array(softmax_vals_tr))
#        mean_max_sm_val = np.mean(np.array(softmax_vals_eval))
#        st2 = np.std(np.array(softmax_vals_eval))
#        out_of_lib_mean = np.mean(np.array(ignore_sm))
#        out_of_lib_std = np.std(np.array(ignore_sm))

        print(f"Epoch {epoch}, Train acc: {correct_tr/total_tr*100:.2f}%, Val acc: {correct_t / total_t*100:.2f}%, ", \
            f"Top {n_top}: {top_correct / total_t * 100:.2f}%, {time.time() - s:.2f} sec, ") #, \
#            f" Mean Softmax {in_lib_mean:.2f}|{mean_max_sm_val:.2f}|{out_of_lib_mean:.2f}  || ",\
#            f"{in_lib_std:.2f}|{st2:.2f}|{out_of_lib_std:.2f}")
        """
        scheduler.step()


    net.eval()
    #exit()

    ignore_sm = []
    for j, batch_i in enumerate(ignored_loader):
        input_i, label_i, _ = batch_i

        # _, logits, preds = net(Variable(input_i))

        logits = net(input_i)
        pred = F.softmax(logits, dim=1)

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
    unassigned_face_set = face_classifier.FaceLabelSet(which_features)
    unassigned_face_set.add_person('ignore', unassigned_filter[0].id)
    for i, un in enumerate(unassigned_faces):
        if which_features == 'short':
            unassigned_face_set.add_datapoint(unassigned_filter[0].id, un.face_encoding, un.id)
        else:
            unassigned_face_set.add_datapoint(unassigned_filter[0].id, un.face_encoding_512, un.id)

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

        input_u = scaler.transform(input_u)

        _, logits, preds = net(Variable(input_u))

        max_softmax_u, pred_u = torch.max(preds.data, 1)
        top5_vals, pred_top5 = preds.topk(5)
        # print(pred_top5.shape)

        face_ids = face_ids.tolist()
        for ii in range(len(face_ids)):
            try:
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

#                top5_class = pred_top5[ii, :].detach().tolist()
#                top5_logits = top5_vals[ii, :].detach().tolist()
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


            except:
                pass
