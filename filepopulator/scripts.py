#! /usr/bin/env python

from .models import ImageFile, Directory
from datetime import datetime
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.utils import timezone
from django.utils import timezone
from PIL import Image
from PIL.ExifTags import TAGS 
import logging
import os
import re
import time
import traceback

def delete_old_thumbnails(instance):
    os.remove(instance.thumbnail_big.path)
    os.remove(instance.thumbnail_medium.path)
    os.remove(instance.thumbnail_small.path)

def instance_clean_and_save(instance):
    #            print(cur_file)
    file_path = instance.filename
    try:
        instance.full_clean()
    except ValidationError as ve:
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            settings.LOGGER.critical("Did not add JPEG-type photo {}: {}".format(file_path, ve))
        else:
            settings.LOGGER.debug("Did not add photo {}: {}".format(file_path, ve) )
    else:
        try:
            instance.save()
        except ValueError as ve:
            print(dir(instance))
            print(instance.__dict__)
#                for field in dir(instance):
#                    if not field.startwith('_'):
#                    print(field, instance.__dict__[field])
            raise ve
        settings.LOGGER.debug(f"Saved file {file_path} to database")

        assert os.path.isfile(instance.thumbnail_big.path), \
            'Thumbnail {} wasn''t generated for {}.'.\
            format(instance.thumbnail_big.name, file_path)
        assert os.path.isfile(instance.thumbnail_medium.path), \
            'Thumbnail {} wasn''t generated for {}.'.\
            format(instance.thumbnail_medium.name, file_path)
        assert os.path.isfile(instance.thumbnail_small.path), \
            'Thumbnail {} wasn''t generated for {}.'.\
            format(instance.thumbnail_small.name, file_path)

# def add_new_photo(file_path):
#     # This is for images where this file is not in the database.
#     pass


def create_image_file(file_path):

    if not os.path.isfile(file_path):
        settings.LOGGER.debug('File {} is not a file path. Will not insert.'.format(file_path))
        return

    if not re.match(".*\.[j|J][p|P][e|E]?[g|G]$", file_path):
        settings.LOGGER.debug("File {} does not have a jpeg-type ending.".format(file_path))
        return # Success value

    # Check if this photo already exists:
    exist_photo = ImageFile.objects.filter(filename=file_path)

    new_photo = ImageFile(filename=file_path)


    # Case 1: photo exists at this location.
    if len(exist_photo):
        print("Photo exists!")
        if len(exist_photo) > 1:
            settings.LOGGER.critical(f"You have multiple instances of file {file_path} in the database.")
            raise ValueError('Should only have at most one instance of a file {}. You have {}'.format(file_path, len(exist_photo)))
        else:
            exist_photo = exist_photo[0]

        exist_timestamp = exist_photo.dateModified.timestamp()
        new_photo._get_mod_time()
        adding_timestamp = new_photo.dateModified.timestamp()
        print("Check: ", datetime.fromtimestamp(os.path.getctime(file_path)).timestamp(), adding_timestamp)

        # Check the timestamp between the database and the file 
        # under consideration. If they are exactly the same, 
        # then we don't have to change anything in the database.
        # We get the timestamp() value so that we don't have to 
        # deal with some values having a timezone (all the database
        # values) and some not (most pictures). Timestamp simply
        # turns it into a float of UTC seconds. 
        if exist_timestamp == adding_timestamp:
            print(exist_timestamp, adding_timestamp)
            return
        # Only if the files are *not* the same do we compute the
        # md5 hash of the file. This is because reading in the 
        # pixel values of the file is a comparatively expensive
        # operation, taking tenths of a second. If you did that
        # all the time for every one of tens of thousands (or more)
        # pictures, then it would take hours to run through. 
        # Instead, we can process files with no change in ten-thousandths
        # of a second each. Perfect!
        # Small scale test with 200 pictures:
        # With hash every time: ~20 seconds to add all
        # This way with established database (no hashing): ~.5 seconds. 
        # That's a 40x speedup.
        else:
            print(f"Working with {file_path} - photo exists")
            settings.LOGGER.debug(f"Updating file {file_path} in database due to changed timestamp")
            new_photo.process_new_no_md5()
            new_photo._generate_md5_hash()

        if exist_photo.pixel_hash == new_photo.pixel_hash:
            if exist_photo.orientation == new_photo.orientation:
            # The photo is already in place, and the pixel hash hasn't changed, and it hasn't rotated
            # Don't want to delete it -- they reference the same picture in distinct locations.
            # However, our modification timestamps are off, so let's update that. 
                exist_photo.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                instance_clean_and_save(exist_photo)
                return
            else:
                exist_photo = new_photo
                exist_photo.orientation = new_photo.orientation
                exist_photo.dateAdded = timezone.now()
                exist_photo.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                exist_photo.isProcessed = False
                instance_clean_and_save(exist_photo)
                return
        else:
            exist_photo.delete()
            instance_clean_and_save(new_photo)
            return

    # Case 2: No photo exists at this location.
    else:
        print(f"Working with {file_path} - no photo exists")
        settings.LOGGER.debug(f"Adding new file {file_path} to database.")
        new_photo.process_new_no_md5()
        new_photo._generate_md5_hash()
        exist_with_same_hash = ImageFile.objects.filter(pixel_hash = new_photo.pixel_hash)
        if len(exist_with_same_hash):
            if len(exist_with_same_hash) == 1 and not os.path.exists(exist_with_same_hash[0].filename) :
                # Exactly one other, but it's been deleted or moved.
                # In this case, update the filename and the date added
                # and save it back to the database. 
                instance = exist_with_same_hash[0]
                settings.LOGGER.debug(f"Found a file like {file_path} with the same hash. The old file is {instance.filename} .")
                instance.filename = file_path
                instance.dateAdded = timezone.now()
                instance.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                delete_old_thumbnails(instance)
                instance_clean_and_save(instance)
                return

            elif len(exist_with_same_hash) > 1:
                # raise NotImplementedError('More than one...')
                print(f"Same hash: {exist_with_same_hash}")
                # logging.error('This is not how I want it -- I want more matching validation. But getting here was right.')
                for each in exist_with_same_hash:
                    if not os.path.exists(each.filename):
                        print(f"Deleting file {each.filename} since it is no longer in the file path.")
                        each.filename = file_path
                        each.dateAdded = timezone.now()
                        each.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                        delete_old_thumbnails(each)
                        instance_clean_and_save(each)
                        return
            elif os.path.exists(exist_with_same_hash[0].filename) and len(exist_with_same_hash) == 1:
                pass
                # Should just create it.
            else:
                raise NotImplementedError('You shouldn''t be here... the logic should be watertight.')

    new_photo.dateAdded = timezone.now()

    instance_clean_and_save(new_photo)


def add_from_root_dir(root_dir):

    lockfile = settings.LOCKFILE
    
    print(f"Adding lockfile is {lockfile}")
    if os.path.isfile(lockfile):
        print("Locked!")
        return
    else:

        f = open(lockfile, 'w')
        f.close()

        count = 0

        # if False:
        # Under development
        try:
            # Get a list of all images in the root_dir
            actual_file_list = []
            metadata_time = {}
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg')):
                        cur_file = os.path.join(root, f)

                        # Don't try to add files starting with a period - they're often
                        # just system files. 
                        cur_parts = cur_file.split(os.sep)[:-1]
                        filename = cur_file.split(os.sep)[-1]
                        if re.match('\.', filename):
                            continue

                        # Check if a folder starts with '.'. Otherwise, add it in!
                        if not True in set(map(lambda x: x.startswith('.'), cur_parts) ):
                            actual_file_list.append(cur_file)
                            # print(cur_file)

                            metadata = {}
                            mod_time = datetime.fromtimestamp(os.path.getctime(cur_file))
                            metadata_time[cur_file] = mod_time

            # Get a list of files in the database
            db_files = ImageFile.objects.all().values()
            db_file_list = list(db_files.values_list('filename', flat=True))
            db_files = list(db_files)

            # New files: 
            new_files = list(set(actual_file_list) - set(db_file_list))
            print(f"New file length is {len(new_files)}")


            # Now go through and find any that have been modified more recently 
            # than database:
            modded_files = []
            t = 0
            for file in db_files:
                db_mod_time = file['dateModified']
                # if t % 1000 == 0:
                #     print(f"{t} files scanned, {len(modded_files)} mod files found")
                t += 1
                filename = file['filename']
                if filename in actual_file_list:
                    os_mod_time = metadata_time[filename]
                    if db_mod_time.timestamp() >= os_mod_time.timestamp():
                        # No problems -- DB has most up-to-date.
                        pass
                    else:
                        modded_files.append(filename)

                # else:
                    # Don't worry about it -- it's in new_files
                    
            print(f"Mod file length is {len(modded_files)}")
            # Now process the new and modded files. 
            for filename in new_files:
                try:
                    create_image_file(filename)
                except Exception:
                    print(f'{filename} was not processed.')

            for modfile in modded_files:
                try:
                    create_image_file(modfile)
                except Exception:
                    print(f'{filename} was not processed.')

        except Exception as e:
            stack_trace = traceback.format_exc()
            settings.LOGGER.error(type(e).__name__)
            settings.LOGGER.error(e)
            settings.LOGGER.error(cur_file)
            settings.LOGGER.error(stack_trace) 
        finally:
            print("Finished adding from root!")
            try:
                os.remove(lockfile)
            except FileNotFoundError:
                pass






        # try:
        #     for root, dirs, files in os.walk(root_dir):
        #         for f in files:
        #         # Try/catch block only on individual file; lets the rest of the files be added regardless
        #         # of a failure on one. 
        #             count += 1
        #             if count > 0 and count % 10000 == 0:
        #                 print(f"Processed {count:,} images in add from root directory")
        #             try:
        #                 cur_file = os.path.join(root, f)
        #                 cur_parts = cur_file.split(os.sep)[:-1]
        #                 filename = cur_file.split(os.sep)[-1]
        #                 if re.match('\.', filename):
        #                     # Don't try to add files starting with a period - they're often
        #                     # just system files. 
        #                     continue
        #                 # Check if a folder starts with '.'.
        #                 if not True in set(map(lambda x: x.startswith('.'), cur_parts) ):
        #                     create_image_file(cur_file)
        #             except Exception as e:
        #                 stack_trace = traceback.format_exc()
        #                 settings.LOGGER.error(type(e).__name__)
        #                 settings.LOGGER.error(e)
        #                 settings.LOGGER.error(cur_file)
        #                 settings.LOGGER.error(stack_trace) 
        # finally:
        #     try:
        #         os.remove(lockfile)
        #     except FileNotFoundError:
        #         pass

def delete_removed_photos():
    all_photos = ImageFile.objects.all()

    for each_photo in all_photos:
        filepath = each_photo.filename
        if not os.path.isfile(filepath):
            each_photo.delete()

    # ImageFile.objects.all().delete()

def update_dirs_datetime():

    dirs = Directory.objects.all()
    for d in dirs:
    #     print(d)
        d.average_date_taken()
        d.beginning_date_taken()
        d.num_images = d.image_set.count()
        d.save()
