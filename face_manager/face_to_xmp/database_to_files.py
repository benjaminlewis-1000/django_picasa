#! /usr/bin/env python


from .add_face_to_xmp import add_face_to_photo_xmp
from celery import shared_task
from datetime import datetime
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from random import randint
import os
import pickle
import re
import shutil

@shared_task(ignore_result=True, name='face_manager.write_to_files')
def main():
    # Only need to do a thousand at a time -- they'll all get written
    # eventually 
    #.order_by('?')\
    num_unassigned = Face.objects.order_by('?') \
        .filter(written_to_photo_metadata=False)\
        .exclude(declared_name__person_name=settings.BLANK_FACE_NAME).count()

    print(f"There are {num_unassigned} faces to write.")

    batch_size = 500
    batches = num_unassigned // batch_size + 1

    for i in range(batches):
        unassigned_batch = Face.objects.order_by('?') \
            .filter(written_to_photo_metadata=False)\
            .exclude(declared_name__person_name=settings.BLANK_FACE_NAME)[:batch_size]

        for face in unassigned_batch:
            try:
                db_to_jpg(face)
            except:
                pass


def db_to_jpg(face):
    # Get the corresponding ImageFile object
    photo_obj = face.source_image_file
    photo_filename = photo_obj.filename
    print(photo_filename)

    # Change the filename to the RW version of the filename. 
    rw_filename = re.sub('^' + settings.PHOTO_ROOT, settings.PHOTO_ROOT_RW, photo_filename)

    photo_mod_date = datetime.fromtimestamp(os.path.getctime(photo_filename))
    # print(photo_filename)
    # print(rw_filename)

    # print(photo_obj.dateModified, photo_obj.dateTaken, photo_mod_date)

    person_name = face.declared_name.person_name

    box_top = face.box_top
    box_bottom = face.box_bottom
    box_left = face.box_left
    box_right = face.box_right

    face_data = {'name': person_name, 'left': box_left, 'right': box_right, 'top': box_top, 'bottom': box_bottom}

    success = add_face_to_photo_xmp(rw_filename, face_data)
    # success = True
    # print("Success was ", success)

    if success:
        # Update data in the database.
        # Change the date modified on the photo object:
        photo_obj.dateModified = timezone.now()
        super(ImageFile, photo_obj).save()
        # print(photo_obj.dateModified)

        # And change the fact that the face was written to the file.
        # print(face.written_to_photo_metadata)
        face.written_to_photo_metadata = True
        super(Face, face).save()
        # print(face.written_to_photo_metadata)

    # # A field to save the thumbnail. The scripts.py ensures
    # # that this is a square thumbnail.
    # face_thumbnail = os.path.join('/media/', person.face_thumbnail.path)

    # index_dir = os.path.join(outdir, str(index))
    # os.mkdir(index_dir)

    # shutil.copy(filename, index_dir)
    # shutil.copy(face_thumbnail, index_dir)

    # pkl_data = {"name": person_name, "top": box_top, "bottom": box_bottom, "left": box_left, "right": box_right}
    # print(pkl_data)

    # with open(os.path.join(index_dir, 'params.pkl'), 'wb') as fh:
    #     pickle.dump(pkl_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
