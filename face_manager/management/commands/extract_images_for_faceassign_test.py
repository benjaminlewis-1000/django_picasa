#! /usr/bin/env python

from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from django.core.management.base import BaseCommand
from django.conf import settings
from random import randint
import os
import shutil
import pickle

class Command(BaseCommand):
    
    def handle(self, *args, **options):

        outdir = '/code/DATA_assign'
        
        unassigned_people = Face.objects.order_by('?').filter(written_to_photo_metadata=False).exclude(declared_name__person_name=settings.BLANK_FACE_NAME)[:1500]
        print((unassigned_people[0].declared_name))

        for p in range(len(unassigned_people)):

            person = unassigned_people[p]
            person_name = person.declared_name.person_name
            index = p

            filename = person.source_image_file.filename

            box_top = person.box_top
            box_bottom = person.box_bottom
            box_left = person.box_left
            box_right = person.box_right

            # A field to save the thumbnail. The scripts.py ensures
            # that this is a square thumbnail.
            face_thumbnail = os.path.join('/media/', person.face_thumbnail.path)

            index_dir = os.path.join(outdir, str(index))
            os.mkdir(index_dir)

            shutil.copy(filename, index_dir)
            shutil.copy(face_thumbnail, index_dir)

            pkl_data = {"name": person_name, "top": box_top, "bottom": box_bottom, "left": box_left, "right": box_right}
            print(pkl_data)

            with open(os.path.join(index_dir, 'params.pkl'), 'wb') as fh:
                pickle.dump(pkl_data, fh, protocol=pickle.HIGHEST_PROTOCOL)