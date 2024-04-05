#! /usr/bin/env python

from face_manager import models as face_models
from .face_label_set import FaceLabelSet
from django.db.models import Count
import random

def create_dataset(settings, devel=False, which_features='short'):

    assert which_features in ['short', 'long']
    ignore_filter = face_models.Person.objects.filter(person_name='.realignore')
    ignored_faces = face_models.Face.objects.filter(declared_name=ignore_filter[0].id)

    ignored_face_set = FaceLabelSet(which_features)
    ignored_face_set.add_person('ignore', 1)
    for ignf in ignored_faces:
        if which_features == 'short':
            ignored_face_set.add_datapoint(1, ignf.face_encoding, ignf.id)
        else:
            ignored_face_set.add_datapoint(1, ignf.face_encoding_512, ignf.id)

    # print("Ignored face length: ", len(ignored_face_set))

    train_set = FaceLabelSet(which_features)
    val_set = FaceLabelSet(which_features)
    # Get all the faces that have an assigned name and that
    # have enough faces for us to be interested in training.
    # How to filter on foreign key: 
    # https://stackoverflow.com/a/6205303/3158519
    people_filter = face_models.Person.objects.annotate(num_face=Count('face_declared'))\
        .filter(num_face__gt=settings.FACE_NUM_THRESH)\
        .exclude(person_name__in=settings.IGNORED_NAMES)

    # print(len(people_filter))
  
    # Now to put things in a dataset! 
    for p in people_filter:
        train_set.add_person(p.person_name, p.id)
        val_set.add_person(p.person_name, p.id)

        faces_of_person = face_models.Face.objects.filter(declared_name=p.id)
        # print(p.person_name, p.id, len(faces_of_person))
        # print(type(faces_of_person))

        # nn = 0
        # random.shuffle(faces_of_person)

        indices = list(range(len(faces_of_person)))
        random.shuffle(indices)

        if devel: 
            indices = indices[::10]

        num_train = int(len(indices) * 0.9)
        
        try:
            for ii in range(0, num_train):
                idx = indices[ii]
                if which_features == 'short':
                    train_set.add_datapoint(p.id, faces_of_person[idx].face_encoding, faces_of_person[idx].id)
                else:
                    train_set.add_datapoint(p.id, faces_of_person[idx].face_encoding_512, faces_of_person[idx].id)


            for jj in range(num_train, len(indices)):
                idx = indices[jj]
                if which_features == 'short':
                    val_set.add_datapoint(p.id, faces_of_person[idx].face_encoding, faces_of_person[idx].id)
                else:
                    val_set.add_datapoint(p.id, faces_of_person[idx].face_encoding_512, faces_of_person[idx].id)
        except:
            pass

            # nn += 1
            # if nn > 20:
                # break

    return train_set, val_set, ignored_face_set
