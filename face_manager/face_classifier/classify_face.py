#! /usr/bin/env python

from django.conf import settings
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from itertools import chain
from PIL import Image  
from random import randint
import numpy as np
from time import sleep
import os
import pickle
import shutil
import time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn

def get_and_resize(in_path):
    im = Image.open(in_path)
    newsize = (224, 224)
    im = im.resize(newsize, resample=Image.LANCZOS)
    return im


def classify_unassigned_faces(do_all=False, do_ignore=False, unassigned_external = None, batch_processing_size=-1):
    
    # Get a list of all unassigned images. 

    criterion1 = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
    criterion2 = Q(poss_ident1=None)

    if do_all:
        unassigned = Face.objects.filter(criterion1)
    elif do_ignore is False:
        unassigned = Face.objects.filter(criterion1 & criterion2)
    else:
        unassigned = unassigned_external

    # if do_all:
    #     for u in unassigned:
    #         print(u)
    #         u.poss_ident1 = None
    #         u.weight_1 = 0
    #         u.poss_ident2 = None
    #         u.weight_2 = 0

    #         u.save()


    # print(unassigned.count())
    # if do_all:
    #     unassigned = unassigned[:200]
    # unassigned = Face.objects.filter(pk=512185)
    # Make the unassigned images into a dataset. Use the same
    # normalizations as during training.

    img_transforms =  transforms.Compose([
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class faceDataset(data.Dataset):
        def __init__(self, transform=None):
            self.transform = transform

            self.images_master = []
            self.image_db_id_master = []

            unassigned_ct = len(unassigned)
            for i, face in enumerate(unassigned):
                self.images_master.extend([face.face_thumbnail.path])
                self.image_db_id_master.extend([face.id])

            self.images = self.images_master.copy()
            self.image_db_id = self.image_db_id_master.copy()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            img = get_and_resize(img_path)
            img = self.transform(img)

            return img

        def subset(self, start, end):
            self.images = self.images_master[start:end]
            self.image_db_id = self.image_db_id_master[start:end]

    dataset = faceDataset(transform=img_transforms)
    img_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    # Load the pickle file that translates between class number and 
    # database face ID.
    with open('/models/id_to_folder_map.pkl', 'rb') as ph:
        n = pickle.load(ph)
        net_class_to_id = {}
        num_classes = len(n)
        # Now, the way the imageLoader works in Pytorch, it 
        # takes the folders in ASCII order instead of numerical order.
        # So we rectify that.
        a = np.arange(num_classes)
        a = [str(x) for x in a]
        a.sort()
        a = [int(x) for x in a]
        for k in n.keys():
            folder_number = int(n[k].replace('/code/MISC_DATA/face_chips/', ''))
            network_label = a.index(folder_number)
            net_class_to_id[network_label] = k

    # print(net_class_to_id)



    def process_network(network_type, model_file, num_classes):

        if network_type == 'densenet':
            model = models.densenet121(num_classes=num_classes)
        elif network_type == 'resnet':
            model = models.wide_resnet101_2(num_classes=num_classes)
        elif network_type == 'shuffle':
            model = models.shufflenet_v2_x2_0(num_classes=num_classes)

        print(network_type)
        model_params = torch.load(model_file, map_location='cpu')
        model.load_state_dict(model_params)
        model.eval()
        model.cpu()

        s = nn.Softmax(dim=-1)

        class_pred_all = None
        conf_all = None
        all_raw_preds = None

        with torch.no_grad():
            start = time.time()
            for i, imgs in enumerate(img_loader):
                if i % 2 == 0:
                    print(f"{i}/{len(img_loader)}")
                pred = model(imgs.cpu())
                pred = s(pred)
                confidence, class_pred = torch.max(pred, 1)
                if conf_all is None:
                    conf_all = confidence
                    class_pred_all = class_pred
                    all_raw_preds = pred
                else:
                    conf_all = torch.cat((conf_all, confidence))
                    class_pred_all = torch.cat((class_pred_all, class_pred))
                    all_raw_preds = torch.cat((all_raw_preds, pred))
            print(time.time() - start)
        return conf_all.numpy(), class_pred_all.numpy(), all_raw_preds.numpy()

    def do_inference():

        n1_confidence, n1_pred, n1_raw_preds = process_network('shuffle', '/models/shuffle.ptw', num_classes)
        n3_confidence, n3_pred, n3_raw_preds = process_network('resnet', '/models/resnet.ptw', num_classes)
        n2_confidence, n2_pred, n2_raw_preds = process_network('densenet', '/models/densenet.ptw', num_classes)

        summed_raw = (n1_raw_preds + n2_raw_preds + n3_raw_preds) / 3

        # Now we do inference. I've found that agreement between 2+ networks is a good 
        # test for whether the image is in-library or not. 
        idcs_in_lib = np.where( (n1_pred == n2_pred) | (n2_pred == n3_pred) | (n1_pred == n3_pred) )[0]
        # And obviously, out-of-library 
        idcs_out_lib = list(set(np.arange(len(n1_pred))) - set(idcs_in_lib))

        ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME)[0]

        def assign_to_ignore(index):
            db_id = dataset.image_db_id[index]
            if do_ignore:
                # print("IGNORED - DO NOTHING: ",  db_id)
                return
            else:
                face_obj = Face.objects.get(pk=db_id)
                # Set all the possibilities to .ignore
                face_obj.poss_ident1 = face_obj.poss_ident3 = face_obj.poss_ident4 = face_obj.poss_ident5 = ignore_person
                face_obj.weight_1 = face_obj.weight_3 = face_obj.weight_4 = face_obj.weight_5 = 1
                likely_class = np.argmax(summed_raw[index, :])
                lookup_db = net_class_to_id[likely_class]
                print("IGN", likely_class, db_id)
                # face_obj.poss_ident2 = Person.objects.get(pk=lookup_db)
                # face_obj.weight_2 = 0.1
                face_obj.save()

        for enum_i, val in enumerate(idcs_out_lib):
            assign_to_ignore(val)


        conf_1 = n1_confidence[idcs_in_lib]
        conf_2 = n2_confidence[idcs_in_lib]
        conf_3 = n3_confidence[idcs_in_lib]

        threshold = 0.95
        cond_1 = np.where( (n1_pred[idcs_in_lib] == n2_pred[idcs_in_lib]) & (conf_1 > threshold) & (conf_2 > threshold))[0]
        cond_2 = np.where( (n1_pred[idcs_in_lib] == n3_pred[idcs_in_lib]) & (conf_1 > threshold) & (conf_3 > threshold))[0]
        cond_3 = np.where( (n2_pred[idcs_in_lib] == n3_pred[idcs_in_lib]) & (conf_2 > threshold) & (conf_3 > threshold))[0]
        intersection = np.union1d(np.union1d(cond_1, cond_2), cond_3)

        NOT_ASSIGNED_PERSON = Person.objects.filter(person_name=settings.BLANK_FACE_NAME)[0]

        for enum_i, val in enumerate(idcs_in_lib):
            if val in intersection:
                # Go ahead and assign. 

                db_id = dataset.image_db_id[val]
                face_obj = Face.objects.get(pk=db_id)
                if do_ignore:
                    face_obj.declared_name = NOT_ASSIGNED_PERSON
                # Set all the possibilities to .ignore
                face_obj.poss_ident2 = face_obj.poss_ident3 = face_obj.poss_ident4 = face_obj.poss_ident5 = ignore_person
                face_obj.weight_2 = face_obj.weight_3 = face_obj.weight_4 = face_obj.weight_5 = 1
                likely_class = np.argmax(summed_raw[val, :])
                # print(likely_class, db_id)
                likelihood = np.max(summed_raw[val, :])
                lookup_db = net_class_to_id[likely_class]
                face_obj.poss_ident1 = Person.objects.get(pk=lookup_db)
                face_obj.weight_1 = likelihood

                face_obj.save()

            else:
                assign_to_ignore(val)

    if batch_processing_size > 0:
        num_to_process = len(dataset)
        num_batches = int(np.ceil(num_to_process / batch_processing_size))
        print('Num: ', num_to_process, num_batches)
        for i in range(num_batches):
            start = i * batch_processing_size
            end = (i + 1) * batch_processing_size
            dataset.subset(start, end)
            
            do_inference()
            # img_loader = data.DataLoader(dataset[start:end], batch_size=4, shuffle=False, num_workers=4)

            # n1_confidence, n1_pred, n1_raw_preds = process_network('shuffle', '/models/shuffle.ptw', num_classes)
            # n2_confidence, n2_pred, n2_raw_preds = process_network('densenet', '/models/densenet.ptw', num_classes)
            # n3_confidence, n3_pred, n3_raw_preds = process_network('resnet', '/models/resnet.ptw', num_classes)

    else:
        do_inference()
    

    #         for cum_thresh in thresholds_list:
    #             cond4 = np.where(cumulative_conf[idcs] >= cum_thresh)
    #             # print(len(cond_1))
    #             # print(len(cond_2))
    #             # print(len(cond_3))
    #             intersection = np.intersect1d(np.union1d(np.union1d(cond_1, cond_2), cond_3), cond4)
    #             print(f"Majority thresh: {cum_thresh:.2f}, {len(intersection)} | {len(intersection) / len(n1_pred_local) * 100:.2f}%")
