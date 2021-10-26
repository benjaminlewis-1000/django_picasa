#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Q
from face_manager import face_classifier
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from PIL import Image, ExifTags
from scipy import stats
import copy
import cv2
import face_recognition
import io
import itertools
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import sys
from image_face_extractor import reencoder, ip_finder 


class Command(BaseCommand):
    
    def handle(self, *args, **options):

        start = time.time()
        # Get a list of all faces
        criterion_ign = ~Q(declared_name__person_name__in=['.ignore', '.realignore', '.jessicatodo', '.cutler_tbd'] )
        criterion_2 = Q(reencoded=False)

        all_faces = Face.objects.filter(criterion_ign&criterion_2).order_by('?')

        num_faces = all_faces.count()

        client_ip = ip_finder.server_finder()

        iter_num = 1
        for face in all_faces.iterator():
            print(f'{iter_num}/{num_faces}, {iter_num / num_faces * 100:.4f}%')
            iter_num += 1
            if face.reencoded:
                continue

            face_location = [(face.box_top, face.box_right, face.box_bottom, face.box_right)]
            source_image_file = face.source_image_file.filename

            encoding = reencoder.face_encoding_client(source_image_file, face_location, client_ip)
            # print(source_image_file, face_location, encoding)

            face.encoding = encoding
            face.reencoded = True
            face.save()
#            if time.time() - start > 60 :
#                return

