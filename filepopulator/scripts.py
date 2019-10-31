#! /usr/bin/env python

from PIL import Image
import logging
from PIL.ExifTags import TAGS 
from django.core.files.base import ContentFile
from django.utils import timezone
from django.http import HttpResponse
from django.core.exceptions import ValidationError
from .models import ImageFile, Directory
from datetime import datetime
from django.conf import settings
from django.utils import timezone
import os

logging.basicConfig(level=settings.LOG_LEVEL)

def create_image_file(file_path):

    # print("At start: ", ImageFile.objects.all())
    if not os.path.isfile(file_path):
        logging.debug('File {} is not a file path. Will not insert.'.format(file_path))
        return

    # Check if this photo already exists:
    exist_photo = ImageFile.objects.filter(filename=file_path)
    # print(exist_photo)

    new_photo = ImageFile(filename=file_path)
    new_photo.process_new()
 

    def instance_clean_and_save(instance):
        try:
            instance.full_clean()
        except ValidationError as ve:
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                logging.critical("Did not add JPEG-type photo {}: {}".format(file_path, ve))
            else:
                logging.debug("Did not add photo {}: {}".format(file_path, ve) )
        else:
            instance.save()

            assert os.path.isfile(instance.thumbnail_big.path), \
                'Thumbnail {} wasn''t generated for {}.'.\
                format(instance.thumbnail_big.name, file_path)
            assert os.path.isfile(instance.thumbnail_medium.path), \
                'Thumbnail {} wasn''t generated for {}.'.\
                format(instance.thumbnail_medium.name, file_path)
            assert os.path.isfile(instance.thumbnail_small.path), \
                'Thumbnail {} wasn''t generated for {}.'.\
                format(instance.thumbnail_small.name, file_path)


    # Case 1: photo exists at this location.
    if len(exist_photo):
        if len(exist_photo) > 1:
            raise ValueError('Should only have at most one instance of a file. You have {}'.format(len(exist_photo)))
        else:
            exist_photo = exist_photo[0]

        if exist_photo.pixel_hash == new_photo.pixel_hash:
            if exist_photo.orientation == new_photo.orientation:
            # The photo is already in place, and the pixel hash hasn't changed, and it hasn't rotated
            # Don't want to delete it -- they reference the same picture in distinct locations.
                return
            else:
                exist_photo = new_photo
                exist_photo.orientation = new_photo.orientation
                exist_photo.dateAdded = timezone.now()
                exist_photo.isProcessed = False
                instance_clean_and_save(exist_photo)
                return
        else:
            exist_photo.delete()
            instance_clean_and_save(new_photo)
            return

    # Case 2: No photo exists at this location.
    else:
        # new_photo.process_new()
        exist_with_same_hash = ImageFile.objects.filter(pixel_hash = new_photo.pixel_hash)
        # print("same hash: ", exist_with_same_hash)
        if len(exist_with_same_hash):
            if len(exist_with_same_hash) == 1 and not os.path.exists(exist_with_same_hash[0].filename) :
                # Exactly one other, but it's been deleted or moved.
                # In this case, update the filename and the date added
                # and save it back to the database. 
                instance = exist_with_same_hash[0]
                instance.filename = file_path
                instance.dateAdded = timezone.now()
                instance_clean_and_save(instance)
                return

            elif len(exist_with_same_hash) > 1:
                # raise NotImplementedError('More than one...')
                for each in exist_with_same_hash:
                    if not os.path.exists(each.filename):
                        each.filename = file_path
                        each.dateAdded = timezone.now()
                        instance_clean_and_save(each)
                        return
            elif os.path.exists(exist_with_same_hash[0].filename) and len(exist_with_same_hash) == 1:
                pass
                # Should just create it.
            else:
                raise NotImplementedError('You shouldn''t be here...')

    new_photo.dateAdded = timezone.now()

    instance_clean_and_save(new_photo)


def add_from_root_dir(root_dir):

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            cur_file = os.path.join(root, f)
            create_image_file(cur_file)

def delete_removed_photos():
    all_photos = ImageFile.objects.all()

    for each_photo in all_photos:
        filepath = each_photo.filename
        if not os.path.isfile(filepath):
            each_photo.delete()

    # ImageFile.objects.all().delete()
