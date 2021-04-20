#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from random import randint
import numpy as np
import os
import pickle
import shutil
from itertools import chain
import time
from PIL import Image  

def get_and_resize(in_path, out_path):
    im = Image.open(in_path)
    newsize = (224, 224)
    im = im.resize(newsize, resample=Image.LANCZOS)
    im.save(out_path)


class Command(BaseCommand):
    
    def handle(self, *args, **options):

        # ignore_dir = '/code/MISC_DATA/ignore_chips'
        # try:
        #     os.makedirs(ignore_dir)
        # except:
        #     pass

        # ignored_faces = Face.objects.filter(declared_name__person_name__in=['.ignore', '.realignore'])
        # num_to_process = ignored_faces.count()
        # print(num_to_process)

        # ignored_iterator = ignored_faces.iterator()

        # try:
        #     first_face = next(ignored_iterator)
        # except StopIteration:
        #     # No rows were found, so do nothing.
        #     pass
        # else:
        #     # At least one row was found, so iterate over
        #     # all the rows, including the first one.
        #     num = 0
        #     for f in chain([first_face], ignored_iterator):
        #         print(f"{num} / {num_to_process}, {num/num_to_process * 100:.4f}%")
        #         num += 1
        #         img_chip_path = f.face_thumbnail.path
        #         person_id = f.declared_name.id
        #         folder_name = ignore_dir

        #         new_chip_name = os.path.join(folder_name, os.path.basename(img_chip_path))
        #         # shutil.copy(img_chip_path, new_chip_name)
        #         get_and_resize(img_chip_path, new_chip_name)

        # exit()

        outdir = '/code/MISC_DATA/face_chips'
        try:
            os.makedirs(outdir)
        except:
            pass

        min_num = settings.FACE_NUM_THRESH 
        names = Person.objects.annotate(c=Count('face_declared')) \
            .filter(c__gt=min_num) \
            .filter(~Q(person_name__in=settings.IGNORED_NAMES) ) # Exclude people in the IGNORED_NAMES list.

        person_faces = Face.objects.filter(declared_name__in=names)
        num_to_process = person_faces.count()
        print(num_to_process)

        id_to_folder_map = {}

        faces_iterator = person_faces.iterator()

        try:
            first_face = next(faces_iterator)
        except StopIteration:
            # No rows were found, so do nothing.
            pass
        else:
            # At least one row was found, so iterate over
            # all the rows, including the first one.
            num = 0
            for f in chain([first_face], faces_iterator):
                print(f"{num} / {num_to_process}, {num/num_to_process * 100:.4f}%")
                num += 1
                img_chip_path = f.face_thumbnail.path
                person_id = f.declared_name.id
                if person_id in id_to_folder_map.keys():
                    folder_name = id_to_folder_map[person_id]
                else:
                    folder_num = len(id_to_folder_map)
                    folder_name = os.path.join(outdir, str(folder_num))
                    try:
                        os.makedirs(folder_name)
                    except:
                        pass
                    id_to_folder_map[person_id] = folder_name
                print(folder_name)

                new_chip_name = os.path.join(folder_name, os.path.basename(img_chip_path))
                # shutil.copy(img_chip_path, new_chip_name)
                get_and_resize(img_chip_path, new_chip_name)

            with open(os.path.join(outdir, 'id_to_folder_map.pkl'), 'wb') as fh:
                pickle.dump(id_to_folder_map, fh)
