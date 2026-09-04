#! /usr/bin/env python

from .models import ImageFile, Directory, DuplicateFile, FailedImageFile, IMAGE_EXTENSION_REGEX, HEIC_EXTENSIONS
import common
from datetime import datetime
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.utils import timezone
from django.utils import timezone
from picasa import celery_app
from PIL import Image
from PIL.ExifTags import TAGS 
import csv
import logging
import numpy as np
import os
import re
import sys
import time
import traceback

def delete_old_thumbnails(instance):
    os.remove(instance.thumbnail_big.path)
    os.remove(instance.thumbnail_medium.path)
    os.remove(instance.thumbnail_small.path)

def instance_clean_and_save(instance, record_failure=True):
    """Returns (success, error_message); error_message is None on success.

    record_failure=False lets a caller that needs custom failure handling
    (see create_image_file()'s hash-changed branch, which needs to decide
    whether to keep or drop a *different* existing row) opt out of this
    function's own generic OSError bookkeeping and handle the (False,
    error_message) result itself.
    """

    file_path = instance.filename
    try:
        instance.full_clean()
    except ValidationError as ve:
        if file_path.lower().endswith(('.jpg', '.jpeg') + HEIC_EXTENSIONS):
            settings.LOGGER.critical("Did not add photo {}: {}".format(file_path, ve))
        else:
            settings.LOGGER.debug("Did not add photo {}: {}".format(file_path, ve) )
        return False, str(ve)
    else:
        try:
            instance.save()
        except ValueError as ve:
            print(dir(instance))
            print(instance.__dict__)

            raise ve
        except OSError as e:
            # A decode failure can surface here even when
            # _generate_md5_hash() already succeeded (e.g. via its
            # cv2.imread() fallback, which is more tolerant of truncated
            # files than PIL): _generate_thumbnail() re-decodes the image
            # via PIL to resize it and can raise independently.
            settings.LOGGER.error(f"File {file_path} failed to decode during save: {e}")
            if record_failure:
                # Route the same way create_image_file()'s own OSError
                # handling does -- update the existing row if this is one
                # (instance.pk is set), otherwise track it in
                # FailedImageFile since no row exists.
                if instance.pk is not None:
                    ImageFile.objects.filter(pk=instance.pk).update(
                        image_load_failed=True,
                        image_load_error=str(e),
                        dateModified=instance.dateModified,
                    )
                else:
                    FailedImageFile.objects.update_or_create(
                        filename=file_path,
                        defaults={
                            "error_message": str(e),
                            "file_mod_time": os.path.getctime(file_path),
                        },
                    )
            return False, str(e)
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

        # A successful save means this file just decoded fine -- clear any
        # stale FailedImageFile record from an earlier failed attempt.
        FailedImageFile.objects.filter(filename=file_path).delete()
        return True, None

# def add_new_photo(file_path):
#     # This is for images where this file is not in the database.
#     pass

def write_duplicates_csv(not_added_file, in_db_file):

    titles = ['not_added', 'in_database']

    if os.path.exists(settings.DUPLICATES_FILE):        
        with open(settings.DUPLICATES_FILE, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[titles[0]] == not_added_file and row[titles[1]] == in_db_file:
                    print("We've got that one!")
                    return

    if os.path.exists(settings.DUPLICATES_FILE):
        with open(settings.DUPLICATES_FILE, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([not_added_file, in_db_file])
    else:
        with open(settings.DUPLICATES_FILE, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=titles)
            writer.writeheader()
            writer.writerow({titles[0]: not_added_file, titles[1]: in_db_file})


def _pixel_arrays_match(new_photo, candidate_filename):
    """Verifies that two files' actual decoded pixel content is
    identical, not just their pixel_hash MD5 digests. pixel_hash is only
    a 128-bit MD5 of the decoded, flattened pixel array (ImageFile.
    _generate_md5_hash()) -- collisions between genuinely different
    photos are astronomically unlikely in practice, but a synthetic one
    is trivial to construct on purpose (see filepopulator.tests.
    ImageFileTests.test_same_pixel_hash, which does exactly that to
    confirm two different images sharing a pixel_hash are NOT treated as
    duplicates of each other).

    Re-decodes candidate_filename the same way ImageFile._generate_md5_
    hash() does and compares the actual pixel arrays -- only called on
    the rare "pixel_hash already matched" path, not on every file, so
    the extra decode cost here doesn't matter for the common case.
    """
    if not os.path.exists(candidate_filename):
        return False
    candidate = ImageFile(filename=candidate_filename)
    try:
        candidate.process_new_no_md5()
        candidate._generate_md5_hash()
    except OSError:
        return False
    return (
        new_photo.pixels.shape == candidate.pixels.shape
        and bool(np.array_equal(new_photo.pixels, candidate.pixels))
    )


def create_image_file(file_path):

    if not os.path.isfile(file_path):
        settings.LOGGER.debug('File {} is not a file path. Will not insert.'.format(file_path))
        return

    if not re.match(r".*" + IMAGE_EXTENSION_REGEX, file_path):
        settings.LOGGER.debug("File {} does not have a supported image-type ending.".format(file_path))
        return # Success value

    # Check if this photo already exists:
    exist_photo = ImageFile.objects.filter(filename=file_path)
    duplicate_exists = DuplicateFile.objects.filter(filename=file_path)

    if len(duplicate_exists) > 0:
        # Then we just want to get rid of everything and start over... 
        for existing_file in exist_photo:
            existing_file.delete()
        for dupe in duplicate_exists:
            dupe.delete()
        return
    
    new_photo = ImageFile(filename=file_path)

    # Case 1: photo exists at this location.
    if len(exist_photo):
        settings.LOGGER.info(f"Photo exists! {file_path}")
        if len(exist_photo) > 1:
            settings.LOGGER.critical(f"You have multiple instances of file {file_path} in the database.")
            raise ValueError('Should only have at most one instance of a file {}. You have {}'.format(file_path, len(exist_photo)))
        else:
            exist_photo = exist_photo[0]

        exist_timestamp = exist_photo.dateModified.timestamp()
        new_photo._get_mod_time()
        adding_timestamp = new_photo.dateModified.timestamp()
        settings.LOGGER.info(f"Check: {datetime.fromtimestamp(os.path.getctime(file_path)).timestamp()}, {adding_timestamp}")

        # Check the timestamp between the database and the file 
        # under consideration. If they are exactly the same, 
        # then we don't have to change anything in the database.
        # We get the timestamp() value so that we don't have to 
        # deal with some values having a timezone (all the database
        # values) and some not (most pictures). Timestamp simply
        # turns it into a float of UTC seconds. 
        if exist_timestamp == adding_timestamp:
            settings.LOGGER.info(f"Existing timestamp: {exist_timestamp}, Adding timestamp: {adding_timestamp}")
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
            try:
                new_photo.process_new_no_md5()
                new_photo._generate_md5_hash()
            except OSError as e:
                # File was previously ingested fine but is now unreadable
                # (corrupted on disk since, or a bad in-place edit). Record
                # the failure on the *existing* row rather than crashing --
                # ImageFile.objects.update() is used deliberately, not
                # exist_photo.save(), since save() would re-run
                # _generate_md5_hash() and hit this same error again.
                # Bumping dateModified to the file's current mtime is what
                # makes create_image_file()'s "timestamp unchanged, skip"
                # check above stop retrying this file every run -- it'll
                # only be retried again once the file's mtime changes
                # (e.g. someone fixes or replaces it).
                settings.LOGGER.error(f"File {file_path} failed to decode: {e}")
                ImageFile.objects.filter(pk=exist_photo.pk).update(
                    image_load_failed=True,
                    image_load_error=str(e),
                    dateModified=new_photo.dateModified,
                )
                return

        if exist_photo.pixel_hash == new_photo.pixel_hash:
            if exist_photo.orientation == new_photo.orientation:
            # The photo is already in place, and the pixel hash hasn't changed, and it hasn't rotated
            # Don't want to delete it -- they reference the same picture in distinct locations.
            # However, our modification timestamps are off, so let's update that.
                exist_photo.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                # Clear a previous decode-failure flag, if any -- getting
                # this far means _generate_md5_hash() just succeeded.
                exist_photo.image_load_failed = False
                exist_photo.image_load_error = None
                instance_clean_and_save(exist_photo)
                return
            else:
                # Orientation changed (pixel hash didn't) -- any existing
                # Face rows on this image have box coordinates computed
                # against the *old* orientation/rotation, which no longer
                # correspond to the newly-decoded (correctly rotated)
                # pixel data. isProcessed=False below gets them
                # redetected, but the stale old Face rows were never
                # cleared here, so they'd sit alongside fresh detections
                # with incompatible coordinates -- exactly the shape of
                # bug that made find_and_encode_faces() crash on stale
                # pre-existing faces from before this session's EXIF-
                # orientation-consolidation fix (see
                # update_list_of_no_matching_detects() for the defensive
                # fix on that side). Delete them properly (Face.delete(),
                # not a bulk queryset .delete(), so its thumbnail file on
                # disk gets cleaned up too) rather than leaving them to be
                # discovered as stale later.
                from face_manager.models import Face
                for stale_face in Face.objects.filter(source_image_file=exist_photo):
                    stale_face.delete()

                # Regression fix: new_photo was constructed via
                # ImageFile(filename=file_path) with no pk, so
                # `exist_photo = new_photo` followed by .save() performed
                # an INSERT, not an UPDATE -- silently creating a *second*
                # ImageFile row for the same filename (the original stayed
                # untouched, stale, and isProcessed unchanged) instead of
                # updating the one that actually exists. Preserve the
                # original pk so this updates in place.
                old_pk = exist_photo.pk
                exist_photo = new_photo
                exist_photo.pk = old_pk
                # Setting .pk alone isn't enough -- new_photo was
                # constructed fresh (never fetched from the DB), so
                # Django's _state.adding is still True, and full_clean()'s
                # validate_unique() then treats reusing this pk as a
                # collision ("Image file with this ID already exists")
                # instead of recognizing it as the same row being updated.
                exist_photo._state.adding = False
                exist_photo.orientation = new_photo.orientation
                exist_photo.dateAdded = timezone.now()
                exist_photo.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                exist_photo.isProcessed = False
                instance_clean_and_save(exist_photo)
                return
        else:
            # The pixel hash changed -- normally means the file was
            # genuinely replaced with different (valid) content. But if
            # the *new* content turns out to be corrupted, saving it can
            # fail at the thumbnail stage even though _generate_md5_hash()
            # above already succeeded (see instance_clean_and_save()'s own
            # OSError handling). Save the replacement BEFORE deleting the
            # old row -- record_failure=False so a failure here doesn't
            # get treated as "brand new file, never had a row" (it did),
            # and the old good row is kept (flagged) instead of being
            # deleted with nothing to replace it.
            saved, error = instance_clean_and_save(new_photo, record_failure=False)
            if saved:
                exist_photo.delete()
            else:
                settings.LOGGER.error(
                    f"File {file_path} changed but its new content failed to "
                    f"decode: {error}. Keeping the prior row."
                )
                ImageFile.objects.filter(pk=exist_photo.pk).update(
                    image_load_failed=True,
                    image_load_error=error,
                    dateModified=new_photo.dateModified,
                )
            return

    # Case 2: No photo exists at this location.
    else:
        settings.LOGGER.debug(f"Working with {file_path} - no photo exists")
        settings.LOGGER.debug(f"Adding new file {file_path} to database.")
        try:
            new_photo.process_new_no_md5()
            new_photo._generate_md5_hash()
        except OSError as e:
            # Never successfully ingested -- no ImageFile row exists to
            # update (one can't be created without a successful decode:
            # ImageFile.save() needs width/height/thumbnails from it), so
            # track it in FailedImageFile instead. file_mod_time lets
            # add_from_root_dir() skip retrying this file every run while
            # it stays broken, but retry it once its mtime changes.
            settings.LOGGER.error(f"File {file_path} failed to decode: {e}")
            FailedImageFile.objects.update_or_create(
                filename=file_path,
                defaults={
                    "error_message": str(e),
                    "file_mod_time": os.path.getctime(file_path),
                },
            )
            return
        # print(new_photo.pixel_hash)
        exist_with_same_hash = ImageFile.objects.filter(pixel_hash = new_photo.pixel_hash)
        # print("Comparison, same hash: ", exist_with_same_hash, len(exist_with_same_hash), exist_with_same_hash[0])
        # print(exist_with_same_hash[0].filename, file_path, exist_with_same_hash[0].filename == file_path)
        # if len(exist_with_same_hash):
        if len(exist_with_same_hash) == 1 and not os.path.exists(exist_with_same_hash[0].filename) :
            # Exactly one other, but it's been deleted or moved.
            # In this case, update the filename and the date added
            # and save it back to the database. 
            instance = exist_with_same_hash[0]
            print(f"Found a file like {file_path} with the same hash. The old file is {instance.filename} .")
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
            moved_into_existing = False
            for each in exist_with_same_hash:
                if not os.path.exists(each.filename):
                    print(f"Deleting file {each.filename} since it is no longer in the file path.")
                    each.filename = file_path
                    each.dateAdded = timezone.now()
                    each.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))
                    delete_old_thumbnails(each)
                    instance_clean_and_save(each)
                    moved_into_existing = True
                    break
            if moved_into_existing:
                return
            # None of the existing same-hash rows were missing from disk.
            # Verify actual pixel content (not just the pixel_hash digest)
            # against each candidate before trusting it as a real
            # duplicate -- a bare hash match used to be treated as
            # sufficient here, which both (a) missed the possibility of a
            # hash collision between genuinely different photos and (b)
            # had a real bug: even when correctly identified as a
            # duplicate, this branch never returned, so the file got
            # BOTH a DuplicateFile record AND its own redundant
            # ImageFile row, defeating the whole point of tracking
            # duplicates.
            for each in exist_with_same_hash:
                if _pixel_arrays_match(new_photo, each.filename):
                    new_dup = DuplicateFile(filename=file_path)
                    print("File exists (multiple)...", file_path, '. Marking as duplicate.')
                    new_dup.save()
                    return
            settings.LOGGER.warning(
                f"pixel_hash matched {len(exist_with_same_hash)} existing row(s) for "
                f"{file_path}, but none had matching pixel content -- treating as a "
                f"distinct photo (hash collision, not a real duplicate)."
            )
        elif len(exist_with_same_hash) == 1 and os.path.exists(exist_with_same_hash[0].filename):
            # There is a candidate duplicate -- same pixel_hash already
            # ingested at a different path that's still on disk. Verify
            # actual pixel content before trusting the hash (see
            # _pixel_arrays_match's docstring); only if it genuinely
            # matches do we record it as a DuplicateFile and stop -- do
            # NOT fall through to create an ImageFile row too (that was a
            # real bug: this branch never returned, so every duplicate
            # file got correctly recorded as a DuplicateFile AND
            # incorrectly given its own full ImageFile row).
            if _pixel_arrays_match(new_photo, exist_with_same_hash[0].filename):
                new_dup = DuplicateFile(filename=file_path)
                print("File exists...", file_path, exist_with_same_hash[0].filename, '. Marking as duplicate.')
                new_dup.save()
                return
            settings.LOGGER.warning(
                f"pixel_hash matched {exist_with_same_hash[0].filename} for {file_path}, "
                f"but pixel content differs -- treating as a distinct photo (hash "
                f"collision, not a real duplicate)."
            )

        elif len(exist_with_same_hash) == 0:
            settings.LOGGER.info("New photo should be created")
        else:
            pass
            # The photo should be created


    new_photo.dateAdded = timezone.now()

    instance_clean_and_save(new_photo)


def add_from_root_dir(root_dir):

    # Postgres advisory lock (see common/advisory_lock.py), replacing the
    # old settings.LOCKFILE file-based lock: that mechanism was a plain
    # os.path.isfile() check-then-create with no wait/retry/timeout, and
    # a hard kill/OOM/container restart mid-run could leave the file
    # behind forever, silently no-op'ing every future scheduled run
    # ("Locked!" then return) with no alerting. An advisory lock is tied
    # to the database session, so it can't be left stale that way -- it
    # releases automatically the moment the holding connection drops.
    with common.advisory_lock('filepopulator.add_from_root_dir') as acquired:
        if not acquired:
            print("Locked!")
            return

        count = 0

        # if False:
        # Under development
        try:
            # Get a list of all images in the root_dir
            actual_file_list = []
            metadata_time = {}
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg') + HEIC_EXTENSIONS):
                        cur_file = os.path.join(root, f)

                        # Don't try to add files starting with a period - they're often
                        # just system files. 
                        cur_parts = cur_file.split(os.sep)[:-1]
                        filename = cur_file.split(os.sep)[-1]
                        if re.match(r'\.', filename):
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
            # db_files = list(db_files)

            duplicate_files = DuplicateFile.objects.all().values()
            dup_file_list = list(duplicate_files.values_list('filename', flat=True))
            # dup_files = list(dup_file_list)

            # Files that have never successfully decoded, and whose mtime
            # hasn't changed since that failure -- skip these so they
            # aren't re-attempted (and re-fail) on every single run.
            # A file whose mtime DOES differ from what's recorded (fixed or
            # replaced) is left in new_files so it gets retried; a
            # successful retry clears its FailedImageFile row (see
            # instance_clean_and_save()).
            unchanged_failed_file_list = [
                f.filename for f in FailedImageFile.objects.all()
                if f.filename in metadata_time
                and os.path.getctime(f.filename) == f.file_mod_time
            ]

            # New files:
            new_files = list(
                set(actual_file_list) - set(db_file_list) - set(dup_file_list)
                - set(unchanged_failed_file_list)
            )
            print(f"New file length is {len(new_files)}")

            for filename in new_files:
                try:
                    create_image_file(filename)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(f'{filename} was not processed. {e}, {exc_type}, {exc_tb.tb_lineno}')
                    print(traceback.format_exc())

            print("Processed new_files")

        except Exception as e:
            stack_trace = traceback.format_exc()
            settings.LOGGER.error(type(e).__name__)
            settings.LOGGER.error(e)
            settings.LOGGER.error(cur_file)
            settings.LOGGER.error(stack_trace) 
        finally:
            print("Finished adding from root!")

def check_file_mods():
    try:
        # Get a list of files in the database
        db_files = ImageFile.objects.all().values()

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
            if os.path.exists(filename): 
#                os_mod_time = metadata_time[filename]
                os_mod_time = datetime.fromtimestamp(os.path.getctime(filename))
                if db_mod_time.timestamp() >= os_mod_time.timestamp():
                    # No problems -- DB has most up-to-date.
                    pass
                else:
                    modded_files.append(filename)

            # else:
                # Don't worry about it -- it's in new_files
        settings.LOGGER.debug(f"Mod file length is {len(modded_files)}")
                    
        # Now process the new and modded files. 

        for modfile in modded_files:
            try:
                create_image_file(modfile)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f'{modfile} was not processed. {e}, {exc_type}, {exc_tb.tb_lineno}')
                print(traceback.format_exc())

    except Exception as e:
        stack_trace = traceback.format_exc()
        settings.LOGGER.error(type(e).__name__)
        settings.LOGGER.error(e)
#        settings.LOGGER.error(filename)
        settings.LOGGER.error(stack_trace)
    finally:
        print("Finished checking file mod times!")

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
