#! /usr/bin/env python

from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from django.core.management.base import BaseCommand
import cv2
import PIL
from django.conf import settings
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ExifTags
from facenet_pytorch import MTCNN, InceptionResnetV1
from io import BytesIO
import torch
import time
import pickle
from django.db.models import Count

class Command(BaseCommand):
    
    def handle(self, *args, **options):

        # Get a list of faces

        min_num = 50

        # faces = Face.objects.annotate(c=Count('declared_name')).filter(c__gt=50)[:100] # .all()
        # print(len(faces))
        names = Person.objects.annotate(c=Count('face_declared')).filter(c__gt=min_num) # .all()

        name_vec = []
        features = []

        for n in names:
            # print(n.person_name, n.id)
            name_vec.append([n.person_name, n.id])

        # print(name_vec)
        with open('/code/names.pkl', 'wb') as fh:
            pickle.dump(name_vec, fh, protocol=pickle.HIGHEST_PROTOCOL)

        feature_file = open('/code/features2.pkl', 'wb')

        for i, n_face in enumerate(names):

            f_subset = Face.objects.filter(declared_name=n_face) # .all()
            print(len(f_subset))

            sub_array = []

            for f in f_subset:
                # print(f)
                label = f.declared_name.id
                # print(label)
                encoding = f.face_encoding
                # print(encoding[:10])
                long_encoding = f.face_encoding_512
                # print(long_encoding[:10])
                instance = [label, encoding, long_encoding]
                sub_array.append(instance)

            # print(f"{idx_end} / {num_faces}, {idx_end / num_faces * 100:.2f}%")
            pickle.dump(sub_array, feature_file, protocol=pickle.HIGHEST_PROTOCOL)

        exit()

        # num_faces = Face.objects.all().count()
        # print(num_faces)
        # batch_size = 500
        # batches = num_faces // batch_size + 1

        # for f in range(batches):
        #     idx_start = f * batch_size
        #     idx_end = (f + 1) * batch_size
        #     f_subset = faces[idx_start:idx_end]
