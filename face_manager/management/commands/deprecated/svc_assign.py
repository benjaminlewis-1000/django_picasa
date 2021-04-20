#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pickle
from joblib import dump, load
import time
import os

class Command(BaseCommand):
    
    def handle(self, *args, **options):

        # names = Person.objects.annotate(c=Count('face_declared')) \
        #     .filter(c__gt=min_num) \
        #     .filter(~Q(person_name__in=settings.IGNORED_NAMES) )


        # unassigned_filter = face_models.Person.objects.filter(person_name=settings.BLANK_FACE_NAME)
        # ignore_person = face_models.Person.objects.filter(person_name='.ignore')[0]
        # print(ignore_person)
        unassigned_faces = Face.objects.filter(declared_name__person_name=settings.BLANK_FACE_NAME)
        model_path = settings.CLASSIFY_MODEL_PATH

        with open(os.path.join(model_path, 'id_to_class_128.pkl'), 'rb') as fh:
            label_to_DBID_small = pickle.load(fh)
        with open(os.path.join(model_path, 'id_to_class_512.pkl'), 'rb') as fh:
            label_to_DBID_big = pickle.load(fh)

        assert label_to_DBID_small == label_to_DBID_big, 'The two SVCs have different labels...'

        small_model = load(os.path.join(model_path, 'small_svc_all.pkl'))
        big_model = load(os.path.join(model_path, 'big_svc_all.pkl'))

        scaler_small = load(os.path.join(model_path, 'std_scaler_small.bin'))
        scaler_big = load(os.path.join(model_path, 'std_scaler_big.bin'))

        blank_face_id = Person.objects.filter(person_name=settings.BLANK_FACE_NAME)[0].id
        ignore_face_id = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME)[0].id

        for face in unassigned_faces[:1000]: # Dad 
            enc_small = np.array(face.face_encoding).reshape(1, -1)
            enc_big = np.array(face.face_encoding_512).reshape(1, -1)
            enc_small = scaler_small.transform(enc_small)
            enc_big = scaler_big.transform(enc_big)

            small_pred = small_model.predict_proba(enc_small)
            small_label = np.argmax(small_pred)
            small_prob = np.max(small_pred)

            big_pred = big_model.predict_proba(enc_big)
            big_label = np.argmax(big_pred)
            big_prob = np.max(big_pred)

            joint_pred = (small_pred + big_pred)/2
            joint_label = np.argmax(joint_pred)
            joint_prob = np.max(joint_pred)


            prob_thresh = 0.6

            if big_label == small_label:
                print(face.id, big_label, small_label, big_prob, small_prob, label_to_DBID_big[big_label], label_to_DBID_small[small_label])
                print(Person.objects.get(id=label_to_DBID_big[big_label]), Person.objects.get(id=label_to_DBID_small[small_label]))
                print(face.id, joint_label, joint_prob, label_to_DBID_big[joint_label], Person.objects.get(id=label_to_DBID_small[joint_label]))

            if small_label == big_label:
                weighted_prob = np.mean((big_prob, small_prob))
                top_id = label_to_DBID_small[small_label]
            # If agree

            # Case 2 : If not agree

            # Case 3 : If < threshold

            # if face.declared_name.person_name == settings.BLANK_FACE_NAME:
            #     # Make it safe to work on this while the frontend is 
            #     # running. In other words, if it's trained and I assigned
            #     # it a face in the middle of its training, it shouldn't 
            #     # assign any possibilities to it.

            #     face.poss_ident1 = top_id
            #     face.weight_1 = weighted_prob
            #     face.poss_ident2 = blank_face_id
            #     face.weight_2 = 0.0
            #     face.poss_ident3 = blank_face_id
            #     face.weight_3 = 0.0
            #     face.poss_ident4 = blank_face_id
            #     face.weight_4 = 0.0
            #     face.poss_ident5 = blank_face_id
            #     face.weight_5 = 0.0
            #     face.save()

        exit()
        # unassigned_face_set = face_classifier.FaceLabelSet(which_features)
        # unassigned_face_set.add_person('ignore', unassigned_filter[0].id)
        # for i, un in enumerate(unassigned_faces):
        #     if which_features == 'short':
        #         unassigned_face_set.add_datapoint(unassigned_filter[0].id, un.face_encoding, un.id)
        #     else:
        #         unassigned_face_set.add_datapoint(unassigned_filter[0].id, un.face_encoding_512, un.id)

        #     if i > 100 and devel:
        #         break
        #     # if len(unassigned_face_set) > 4000:
        #     #     break
            
        # # After the network is all trained, we can go through and work on the data 
        # # from the unassigned faces. 
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
