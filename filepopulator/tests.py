from django.test import TestCase
from django.test import override_settings
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django import forms
from django.conf import settings
from django.db import connection
import psycopg2
import zlib
import hashlib
import json
import os
import binascii
import subprocess
from datetime import datetime, timedelta
from textwrap import wrap # for splitting string
import os
import shutil
import tempfile
import numpy as np
import imageio
import time
from GPSPhoto import gpsphoto
import random
from time import sleep
from PIL import Image
import unittest
from unittest import mock
import pillow_heif

# Create your tests here.

from django.core.management import call_command
from django.core.management.base import CommandError

from .models import ImageFile, Directory, DuplicateFile, FailedImageFile, PhotoCollection, guess_date_from_filename, VideoFile, FailedVideoFile, DuplicateVideoFile
# from .forms import ImageFileForm, DirectoryForm
from .scripts import create_image_file, add_from_root_dir, delete_removed_photos, update_dirs_datetime, check_file_mods, delete_old_thumbnails
from .video_scripts import create_video_file, add_videos_from_root_dir, delete_removed_videos, _parse_exif_date, compute_video_content_hash
from face_manager.models import Face, get_default_blank_person
from common.open_img_oriented import apply_exif_orientation
from . import video_scripts as _video_scripts_module
from .test_video_cache import cached_run_ffprobe, cached_run_exiftool

# Module-level patch (not per-test/per-class): VideoIngestionTests alone
# has dozens of test methods that each call add_videos_from_root_dir()/
# create_video_file(), which internally calls _run_ffprobe()/
# _run_exiftool() against the same small set of real/synthetic fixture
# files, over and over. test_video_cache.py disk-caches those real
# subprocess results (keyed on file content + video_scripts.py's own
# source, so it auto-invalidates if either changes) -- patched here,
# once, for every test in this module, rather than touching each
# individual test method. setUpModule/tearDownModule are standard
# unittest hooks Django's test runner (including under --parallel,
# once per worker process) already calls automatically.
_video_cache_patches = []


def setUpModule():
    _video_cache_patches.extend([
        mock.patch.object(_video_scripts_module, '_run_ffprobe', side_effect=cached_run_ffprobe),
        mock.patch.object(_video_scripts_module, '_run_exiftool', side_effect=cached_run_exiftool),
    ])
    for p in _video_cache_patches:
        p.start()


def tearDownModule():
    for p in _video_cache_patches:
        p.stop()
    _video_cache_patches.clear()


def _tiny_jpeg_bytes(size=(30, 30)):
    import cv2
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    from io import BytesIO
    return BytesIO(buf).read()
# from .views import create_or_get_directory# , create_image_file, add_from_root_dir


class ImageFileTests(TestCase):
    @override_settings(MEDIA_ROOT='/tmp')

    def setUp(self):
        # Get the validation directory, copy it to /tmp so we don't have to worry about destroying it,
        # and get a list of images in it. 
        self.validation_dir = settings.FILEPOPULATOR_VAL_DIRECTORY 

        assert os.path.isdir(self.validation_dir), 'Validation directory in ImageFileTests does not exist.'
        # assert os.path.isdir(self.val_train), 'val_train directory in ImageFileTests does not exist.'
        # assert os.path.isdir(self.val_test), 'val_test directory in ImageFileTests does not exist.'

        self.tmp_valid_dir = '/tmp/img_validation'

        if os.path.exists(self.tmp_valid_dir):
            shutil.rmtree(self.tmp_valid_dir)

        shutil.copytree(self.validation_dir, self.tmp_valid_dir)

        self.test_dir = os.path.join(self.tmp_valid_dir, 'naming')
        self.good_dir = os.path.join(self.test_dir, 'good')
        self.bad_dir = os.path.join(self.test_dir, 'bad')
        self.orientation_dir = os.path.join(self.tmp_valid_dir, 'orientation')

        self.goodFiles = []
        self.badFiles = []
        self.orientFiles = []

        for root, dirs, files in os.walk(self.good_dir):
            for fname in files:
                self.goodFiles.append(os.path.join(root, fname) )

        for root, dirs, files in os.walk(self.orientation_dir):
            for fname in files:
                self.orientFiles.append(os.path.join(root, fname) )

        # List of files that exist but that don't meet the file extension properties.
        for root, dirs, files in os.walk(self.bad_dir):
            for fname in files:
                self.badFiles.append(os.path.join(root, fname) )
            
        # Add file paths that don't exist.
        self.badFiles.append(os.path.join(self.tmp_valid_dir, 'asdf.png'))
        self.badFiles.append(os.path.join('aaa', 'a.png'))
        self.badFiles.append(os.path.join('/images2', 'b.jpg'))
        self.badFiles.append(os.path.join('aaa', 'a.jpg'))
        self.badFiles.append(os.path.join('aaa', 'a.jpg.txt'))

    def tearDown(self):
        # Clean up the objects that were created during these tests. The 
        # delete method also serves to remove the thumbnails. We also remove
        # the files that were copied to /tmp.
        allObjects = ImageFile.objects.all()
        for obj in allObjects:
            obj.delete()

        shutil.rmtree(self.tmp_valid_dir)


    def test_same_pixel_hash(self): ### CHECKED ### 
        # Expected output: construct two images that have the same hash, and see that they 
        # are added to the database with different IDs and different thumbnails. 
        # Also do the same with two completely different images. 
        # In the case with the images with the same hash by construction, they should 
        # have the same pixel_hash, and different pixel_hash in the disparate images.

        # Define a test for two cases: one in which the two files are the same hash but
        # different image, and one where the files are different images and hashes. 
        # is_same_hash will determine which test is running. 
        def test_with_prestrings(str1_pre, str2_pre, is_same_hash, name_suffix):
            # Fill in with identical, random strings until the string is 12k characters long.
            # That lets me do a three channel image that has 2000 pixels (e.g. 50 * 40) by taking
            # every two characters and making them a hex number. 
            mutual_len = 12000 - len(str1_pre)
            # urandom generates random characters, b2a_hex converts them to a nice hex representation.
            mutual = str(binascii.b2a_hex(os.urandom(int(mutual_len // 2))))
            # Remove the "b'" at the beginning and the "'" at the end.
            mutual = mutual[2:-1]

            # Concatenate... not sure why the 'cc'...
            str1 = str1_pre + mutual
            str2 = str2_pre + mutual

            def str_to_numpy(string):
                assert len(string) == 12000
                spl = [int(x, 16) for x in wrap(string, 2)]
                arr = np.array(spl, dtype=np.uint8)
                arr = arr.reshape(50, 40, 3)
                return arr

            # Convert the hex string to a numpy array. 
            array1 = str_to_numpy(str1)
            array2 = str_to_numpy(str2)

            # Quick and dirty way to show that the two numpy arrays
            # aren't the same
            self.assertNotEqual(np.mean(array2 - array1), 0)

            # Save out the numpy arrays to disk so we can run create_image_file on
            # them. 
            # KNOWN TEST BUG (fixed here, not app logic): this test previously
            # reused the literal paths 'outfile1.jpg'/'outfile2.jpg' for both
            # the same-hash and different-hash sub-cases, so the second
            # invocation's create_image_file() calls landed on paths already
            # in the DB from the first invocation ("Case 1: photo exists at
            # this location" in create_image_file) instead of exercising
            # "Case 2: no photo exists" as intended -- which made the
            # different-hash case spuriously get treated as a duplicate of
            # the same-hash case's file. Unique filenames per call fixes it.
            file1 = os.path.join(self.tmp_valid_dir, f'outfile1_{name_suffix}.jpg')
            file2 = os.path.join(self.tmp_valid_dir, f'outfile2_{name_suffix}.jpg')
            png1 = os.path.join(self.tmp_valid_dir, f'outfile1_{name_suffix}.png')
            png2 = os.path.join(self.tmp_valid_dir, f'outfile2_{name_suffix}.png')
            imageio.imsave(png1, array1[:, :, (2, 1, 0)])
            imageio.imsave(png2, array2[:, :, (2, 1, 0)])
            shutil.move(png1, file1)
            shutil.move(png2, file2)

            # Create the two image files. 
            create_image_file(file1)
            create_image_file(file2)

            obj1 = ImageFile.objects.filter(filename=file1)[0]
            obj2 = ImageFile.objects.filter(filename=file2)[0]

            # Assert that two separate files were created, that we have
            # two instances in the database, and that the pixel hashes 
            # are the same (or different) as appropriate. 
            # self.assertEqual(len(allObjects), 2, "The two image files should be same hash but two different instances in DB.")

            if is_same_hash:
                self.assertEqual(obj1.pixel_hash, obj2.pixel_hash )
            else:
                self.assertNotEqual(obj1.pixel_hash, obj2.pixel_hash )

            # Filenames are different and thumbnails were saved properly.
            self.assertNotEqual(obj1.filename, obj2.filename )
            self.assertNotEqual(obj1.id, obj2.id )
            self.assertNotEqual(obj1.thumbnail_small.path, obj2.thumbnail_small.path )

            # Check that the thumbnail was created properly.
            self.assertTrue(os.path.isfile(obj1.thumbnail_small.path))
            self.assertTrue(os.path.isfile(obj2.thumbnail_small.path))

        # Do the two types of test.
        # Two different strings (d55 vs 555) that give the same hash. We can construct
        # two images from this that have the same hash but different values in the image. 
        str1_pre = '4dc968ff0ee35c209572d4777b721587d36fa7b21bdc56b74a3dc0783e7b9518afbfa' + \
            '200a8284bf36e8e4b55b35f427593d849676da0d1555d8360fb5f07fea2'
        str2_pre = '4dc968ff0ee35c209572d4777b721587d36fa7b21bdc56b74a3dc0783e7b9518afbfa' + \
            '202a8284bf36e8e4b55b35f427593d849676da0d1d55d8360fb5f07fea2'
        test_with_prestrings(str1_pre, str2_pre, True, "same")
        rand_1 = str(binascii.b2a_hex(os.urandom(500)))[2:-1]
        rand_2 = str(binascii.b2a_hex(os.urandom(500)))[2:-1]
        test_with_prestrings(rand_1, rand_2, False, "diff")

    def test_duplicate_file_content_does_not_create_a_second_imagefile_row(self):
        # Regression test for a real production bug: create_image_file()'s
        # "pixel_hash matches an existing, still-present file" branch
        # correctly recorded a DuplicateFile for the new path but never
        # returned, so it also fell through and created a full second
        # ImageFile row for the exact same content -- defeating the whole
        # point of tracking duplicates. Confirmed in production: two real
        # ImageFile rows (different paths, one under a bulk-import folder)
        # sharing an identical MD5/pixel_hash.
        original = self.goodFiles[0]
        duplicate_path = os.path.join(os.path.dirname(original), 'duplicate_copy.jpg')
        shutil.copyfile(original, duplicate_path)

        create_image_file(original)
        create_image_file(duplicate_path)

        self.assertEqual(ImageFile.objects.filter(filename=original).count(), 1)
        self.assertEqual(ImageFile.objects.filter(filename=duplicate_path).count(), 0)
        self.assertTrue(DuplicateFile.objects.filter(filename=duplicate_path).exists())

    def test_duplicatefile_original_points_at_the_primary(self):
        original = self.goodFiles[0]
        duplicate_path = os.path.join(os.path.dirname(original), 'duplicate_copy.jpg')
        shutil.copyfile(original, duplicate_path)

        create_image_file(original)
        primary = ImageFile.objects.get(filename=original)
        create_image_file(duplicate_path)

        dup_record = DuplicateFile.objects.get(filename=duplicate_path)
        self.assertEqual(dup_record.original_id, primary.id)

    def test_deleting_the_primary_cascades_to_delete_the_duplicatefile_record(self):
        # The actual point of DuplicateFile.original: if the primary is
        # later deleted (e.g. the file vanished from disk and
        # delete_removed_photos() ran), the stale DuplicateFile record
        # must go with it -- otherwise the surviving duplicate file is
        # permanently blocked from ever being re-ingested as a real
        # photo, with nothing left for it to be "a duplicate of."
        original = self.goodFiles[0]
        duplicate_path = os.path.join(os.path.dirname(original), 'duplicate_copy.jpg')
        shutil.copyfile(original, duplicate_path)

        create_image_file(original)
        primary = ImageFile.objects.get(filename=original)
        create_image_file(duplicate_path)
        self.assertTrue(DuplicateFile.objects.filter(filename=duplicate_path).exists())

        primary.delete()

        self.assertFalse(DuplicateFile.objects.filter(filename=duplicate_path).exists())

    def test_backfill_sets_original_for_a_resolvable_legacy_duplicate(self):
        # Legacy DuplicateFile rows predate the `original` field -- this
        # simulates one (file still on disk, primary still exists) and
        # confirms the backfill command resolves it purely from content,
        # the same way create_image_file() does for new duplicates.
        original = self.goodFiles[0]
        create_image_file(original)
        primary = ImageFile.objects.get(filename=original)

        duplicate_path = os.path.join(os.path.dirname(original), 'legacy_dup.jpg')
        shutil.copyfile(original, duplicate_path)
        legacy_dup = DuplicateFile.objects.create(filename=duplicate_path)  # original left NULL

        call_command('backfill_duplicatefile_original', '--yes')

        legacy_dup.refresh_from_db()
        self.assertEqual(legacy_dup.original_id, primary.id)

    def test_backfill_deletes_a_stale_record_whose_file_no_longer_exists(self):
        legacy_dup = DuplicateFile.objects.create(filename='/tmp/does_not_exist_anymore.jpg')

        call_command('backfill_duplicatefile_original', '--yes')

        self.assertFalse(DuplicateFile.objects.filter(pk=legacy_dup.pk).exists())

    def test_backfill_deletes_a_stale_record_with_no_surviving_primary(self):
        # File still on disk, but nothing in the DB shares its content
        # anymore -- the primary was deleted before this field existed.
        # Deleting frees this now-sole-surviving copy for real
        # re-ingestion on the next scan.
        orphan_path = os.path.join(self.tmp_valid_dir, 'orphaned_dup.jpg')
        shutil.copyfile(self.goodFiles[0], orphan_path)
        legacy_dup = DuplicateFile.objects.create(filename=orphan_path)

        call_command('backfill_duplicatefile_original', '--yes')

        self.assertFalse(DuplicateFile.objects.filter(pk=legacy_dup.pk).exists())

    def test_backfill_leaves_a_corrupted_file_untouched(self):
        corrupted_path = os.path.join(self.tmp_valid_dir, 'corrupted_dup.jpg')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a real jpeg')
        legacy_dup = DuplicateFile.objects.create(filename=corrupted_path)

        call_command('backfill_duplicatefile_original', '--yes')

        legacy_dup.refresh_from_db()
        self.assertIsNone(legacy_dup.original)

    def test_backfill_dry_run_changes_nothing(self):
        legacy_dup = DuplicateFile.objects.create(filename='/tmp/does_not_exist_anymore.jpg')

        call_command('backfill_duplicatefile_original', '--dry-run')

        self.assertTrue(DuplicateFile.objects.filter(pk=legacy_dup.pk).exists())

    def test_backfill_rerun_finds_nothing_more_to_do(self):
        create_image_file(self.goodFiles[0])
        duplicate_path = os.path.join(os.path.dirname(self.goodFiles[0]), 'legacy_dup2.jpg')
        shutil.copyfile(self.goodFiles[0], duplicate_path)
        DuplicateFile.objects.create(filename=duplicate_path)

        call_command('backfill_duplicatefile_original', '--yes')
        call_command('backfill_duplicatefile_original', '--dry-run')

        self.assertEqual(DuplicateFile.objects.filter(original__isnull=True).count(), 0)

    def test_multiple_existing_same_hash_rows_still_recognize_a_genuine_duplicate(self):
        # Exercises the len(exist_with_same_hash) > 1 branch, which has
        # the same missing-return shape of bug as the single-candidate
        # branch above. This scenario (>1 ImageFile already sharing a
        # pixel_hash) shouldn't arise going forward now that the single-
        # candidate case is fixed, but could still exist from data that
        # predates this fix -- a further duplicate of that content should
        # still be recognized, not given yet another row.
        original = self.goodFiles[0]
        create_image_file(original)
        seeded = ImageFile.objects.get(filename=original)

        preexisting_dup_path = os.path.join(os.path.dirname(original), 'preexisting_dup.jpg')
        shutil.copyfile(original, preexisting_dup_path)
        # Bypass create_image_file to seed a second ImageFile row sharing
        # the same pixel_hash, simulating contamination from before this
        # fix existed (bulk_create, like face_manager.tests.
        # make_preexisting_image_row, to avoid ImageFile.save()'s own
        # duplicate-detection running here).
        ImageFile.objects.bulk_create([ImageFile(
            filename=preexisting_dup_path, directory=seeded.directory,
            pixel_hash=seeded.pixel_hash, file_hash=seeded.file_hash,
            width=seeded.width, height=seeded.height, isProcessed=False,
            thumbnail_small='',
        )])

        newest_dup_path = os.path.join(os.path.dirname(original), 'newest_dup.jpg')
        shutil.copyfile(original, newest_dup_path)
        create_image_file(newest_dup_path)

        self.assertTrue(DuplicateFile.objects.filter(filename=newest_dup_path).exists())
        self.assertFalse(ImageFile.objects.filter(filename=newest_dup_path).exists())

    def test_pixel_hash_collision_with_different_content_still_creates_its_own_row(self):
        # The flip side of the two tests above: a pixel_hash MATCH alone
        # must not be trusted as proof of duplicate content (see
        # _pixel_arrays_match's docstring) -- test_same_pixel_hash above
        # already covers the deliberately-constructed-MD5-collision case
        # end to end; this just pins down that the verification helper
        # itself is what's doing the work, directly.
        from filepopulator.scripts import _pixel_arrays_match
        original = self.goodFiles[0]
        different = self.goodFiles[1] if len(self.goodFiles) > 1 else self.orientFiles[0]
        create_image_file(original)
        new_photo = ImageFile(filename=different)
        new_photo.process_new_no_md5()
        new_photo._generate_md5_hash()
        self.assertFalse(_pixel_arrays_match(new_photo, original))

    def test_file_names(self): ### CHECKED ###
        # What we expect to happen: all of the files in goodFiles should be added
        # to the database, none in badFiles should be added, and all the 
        # thumbnails should exist for photos added to the database. 
        
        # Run the create_image_file on both the good files and the bad files. 
        for good in self.goodFiles:
            create_image_file(good)

        for bad in self.badFiles:
            create_image_file(bad)

        # Get a list of all objects that are in the database. All of the good
        # files should be there and none of the bad files should be there. 
        allObjects = ImageFile.objects.all()
        # Keep a list of the files that were retrieved. 
        allFiles = []
        for num in range(len(allObjects) ):
            fullname = allObjects[num].filename
            # Assert: thumbnails created, filename is in the good files list. 
            self.assertTrue(allObjects[num].filename in self.goodFiles)
            self.assertTrue(os.path.isfile(allObjects[num].thumbnail_small.path))
            self.assertTrue(os.path.isfile(allObjects[num].thumbnail_small.path))
            allFiles.append(fullname)

        settings.LOGGER.debug("All files in test_file_names is: {}".format(allFiles))

        # Reverse test -- make sure that all of the good files made it into the database, and 
        # none of the bad files did. 
        for eachGood in self.goodFiles:
            self.assertTrue(eachGood in allFiles, 'File {} has a name that is valid but Django thinks is not.'.format(eachGood) )

        for eachBad in self.badFiles:
            self.assertFalse(eachBad in allFiles, 'File {} has a name that Django thinks is valid but is not.'.format(eachBad))

        # Test that there were only two directories created, since that's how many there 
        # are currently in the good files. The bad files should not have created a 
        # directory.
        dirs = Directory.objects.all()
        self.assertEqual(dirs.count(), 2) 

    def test_image_pixels_change(self): ### CHECKED ### 
        # We test this by putting a completely different picture at the same path. 
        # What do we expect to happen? We want the ID to change and the thumbnail
        # path *and* contents to change. The old thumbnail also should be deleted.

        file1 = self.goodFiles[0]
        # Move the first file to a destination file, then create with
        # that destination file. 
        dest_file = os.path.join(self.tmp_valid_dir, 'tmpmv.jpg')
        shutil.copy(file1, dest_file)
        create_image_file(dest_file)

        # Get the data from the first addition of the dest_file
        first_item = ImageFile.objects.filter(filename=dest_file)
        thumbnail1 = first_item[0].thumbnail_small.path
        id1 = first_item[0].id
        self.assertTrue(os.path.isfile(thumbnail1))

        file2 = self.goodFiles[1]
        shutil.copy(file2, dest_file)
        create_image_file(dest_file)

        # Get the data from the second data addition. 
        second_item = ImageFile.objects.filter(filename=dest_file)
        thumbnail2 = second_item[0].thumbnail_small.path
        id2 = second_item[0].id

        # Thumbnail paths aren't the same, IDs aren't the same,
        # and the first thumbnail has been removed. 
        self.assertNotEqual(thumbnail1, thumbnail2)
        self.assertNotEqual(id1, id2)

        self.assertFalse(os.path.isfile(thumbnail1))
        self.assertTrue(os.path.isfile(thumbnail2))

    def test_repeat_adds(self): ### CHECKED ### 
        # Test to see that adding a file twice does not create a duplicate file.
        # What we expect: the file should be in the database only once, and the
        # ID shouldn't change. 
        goodFile = self.goodFiles[0]
        create_image_file(goodFile)
        file_result = ImageFile.objects.filter(filename = goodFile)[0]
        id_first = file_result.id
        create_image_file(goodFile)
        file_result = ImageFile.objects.filter(filename = goodFile)[0]
        id_second = file_result.id

        self.assertEqual(id_first, id_second)
        num_goodfiles = ImageFile.objects.filter(filename = goodFile).count()
        self.assertEqual(num_goodfiles, 1)
        num_files = ImageFile.objects.all().count()
        self.assertEqual(num_files, 1)

    def test_save_skips_rehash_when_pixel_hash_already_set(self):
        # ImageFile.save() used to unconditionally call
        # _generate_md5_hash() (a real, measured ~15ms-per-call cost:
        # pixel decode + MD5 + a perceptual hash + a DB query) even when
        # the caller had already computed and verified a correct
        # pixel_hash moments earlier -- e.g. create_image_file()'s
        # "mtime changed but pixel hash confirmed unchanged" path, which
        # calls _generate_md5_hash() itself and THEN save(), redoing the
        # exact same work twice for the same file. save() now only
        # recomputes it when pixel_hash isn't already a real value.
        goodFile = self.goodFiles[0]
        img = ImageFile(filename=goodFile)
        img.process_new_no_md5()
        img._generate_md5_hash()
        real_hash = img.pixel_hash
        self.assertNotIn(real_hash, (None, '', '-1', -1))

        with mock.patch.object(ImageFile, '_generate_md5_hash') as mock_hash:
            img.save()
        mock_hash.assert_not_called()
        self.assertEqual(img.pixel_hash, real_hash)

    def test_save_computes_hash_when_pixel_hash_not_yet_set(self):
        goodFile = self.goodFiles[0]
        img = ImageFile(filename=goodFile)
        img.process_new_no_md5()
        self.assertEqual(img.pixel_hash, -1)  # field default, never computed

        with mock.patch.object(
            ImageFile, '_generate_md5_hash', wraps=img._generate_md5_hash
        ) as mock_hash:
            img.save()
        mock_hash.assert_called_once()
        self.assertNotIn(img.pixel_hash, (None, '', '-1', -1))

    def test_save_refreshes_file_hash_even_when_pixel_hash_reused(self):
        # file_hash is derived from self.filename, not pixel content --
        # it must stay correct even on the skip-rehash path, e.g. after a
        # "file moved to a new path" update reuses an existing pixel_hash.
        goodFile = self.goodFiles[0]
        img = ImageFile(filename=goodFile)
        img.process_new_no_md5()
        img._generate_md5_hash()
        img.save()

        new_path = os.path.join(self.tmp_valid_dir, 'moved_for_hash_test.jpg')
        shutil.move(goodFile, new_path)
        img.filename = new_path
        img.save()

        expected = hashlib.md5(new_path.encode('utf-8')).hexdigest()
        self.assertEqual(img.file_hash, expected)

    def test_image_path_changes(self): ### CHECKED ###
        # Case: we have an image that is already in the database, but it is then
        # moved somewhere else in the filesystem and the original file is no
        # longer in place.
        # Expected outcome: The database detects that the file has been moved 
        # and updates the record to show that it's the same ID. The path to the
        # thumbnail should change, though (mostly for ease of doing business)
        # and the old thumbnail shouldn't exist.

        # Are we testing when the same file is encountered elsewhere? 

        file_orig = self.goodFiles[0]
        create_image_file(file_orig)
        orig_data = ImageFile.objects.filter(filename=file_orig)[0]
        # Test default of isProcessed
        self.assertFalse(orig_data.isProcessed)
        orig_data.isProcessed = True
        orig_data.save()
        orig_data = ImageFile.objects.filter(filename=file_orig)[0]
        # Test that isProcessed was saved to database
        self.assertTrue(orig_data.isProcessed)
        new_path = os.path.join(self.tmp_valid_dir, 'tmpmv.jpg')
        shutil.move(file_orig, new_path )
        create_image_file(new_path)

        # Move the file, show that moving it doesn't affect anything that's
        # in the database (until create_image_file is called again)
        shutil.move(new_path, file_orig)

        orig_ref = ImageFile.objects.filter(filename = file_orig)
        new_ref = ImageFile.objects.filter(filename = new_path)
        # Test that in this case, isProcessed was preserved. 
        self.assertTrue(new_ref[0].isProcessed)

        self.assertTrue(os.path.isfile(new_ref[0].thumbnail_small.path))
        self.assertTrue(os.path.isfile(new_ref[0].thumbnail_small.path))
        self.assertFalse(os.path.isfile(orig_data.thumbnail_small.path))
        self.assertFalse(os.path.isfile(orig_data.thumbnail_small.path))

        self.assertEqual(orig_data.id, new_ref[0].id)
        self.assertEqual(orig_ref.count(), 0)
        self.assertEqual(new_ref.count(), 1)
        # Check that the date added changed.
        self.assertNotEqual(orig_data.dateAdded, new_ref[0].dateAdded)

    def test_image_path_changes_two_instances(self): ### CHECKED ###
        # Case: the same pixel image is added at one path, then a genuine
        # duplicate (same content, different path) is added while the
        # original is still in place. UPDATED for the create_image_file()
        # duplicate-detection fix: the duplicate copy is now correctly
        # recorded as a DuplicateFile rather than given its own ImageFile
        # row (previously a real bug -- see CLAUDE.md -- meant it got
        # BOTH a DuplicateFile record and a redundant row). Moving the
        # ORIGINAL file afterward should still be tracked correctly via
        # the normal "moved" detection, unaffected by the unrelated
        # duplicate bookkeeping.

        file_orig = self.goodFiles[0]
        create_image_file(file_orig)
        f1_data = ImageFile.objects.filter(filename=file_orig)[0]
        f1_data.isProcessed = True
        f1_data.save()
        f1_data = ImageFile.objects.filter(filename=file_orig)[0]
        new_path = os.path.join(self.tmp_valid_dir, 'tmpmv.jpg')
        shutil.copy(file_orig, new_path )
        create_image_file(new_path)

        # The duplicate copy is recorded, not given its own row.
        self.assertEqual(ImageFile.objects.filter(filename=new_path).count(), 0)
        self.assertTrue(DuplicateFile.objects.filter(filename=new_path).exists())
        total_records = ImageFile.objects.all()
        self.assertEqual(len(total_records), 1)

        # Move the original file -- should still be tracked as the same
        # row via ordinary move detection.
        f3_path = os.path.join(self.tmp_valid_dir, 'f3.jpg')
        shutil.move(file_orig, f3_path)
        create_image_file(f3_path)
        f3_data = ImageFile.objects.filter(filename=f3_path)[0]
        f1_update = ImageFile.objects.filter(filename=file_orig)
        self.assertEqual(len(f1_update), 0)

        # Check the processing
        self.assertTrue(f3_data.isProcessed)
        # Check that IDs are the same
        self.assertEqual(f1_data.id, f3_data.id)
        self.assertTrue(os.path.isfile(f3_data.thumbnail_small.path))
        self.assertFalse(os.path.isfile(f1_data.thumbnail_small.path))

        self.assertTrue(os.path.isfile(f3_data.filename))
        self.assertFalse(os.path.isfile(f1_data.filename))

        self.assertNotEqual(f1_data.dateAdded, f3_data.dateAdded)

    def test_same_picture_two_paths(self): ### CHECKED ###
        # Case: We have the exact same picture (same pixels) in two different
        # file locations at the same time. We add both images to the database.
        # UPDATED for the create_image_file() duplicate-detection fix: the
        # second path is now correctly recorded as a DuplicateFile rather
        # than given its own ImageFile row (previously a real production
        # bug -- confirmed via two real ImageFile rows sharing an
        # identical MD5 hash -- see CLAUDE.md).

        # Copy the same image to two places and add both to the database.
        src_file = self.goodFiles[0]
        path1 = os.path.join(self.tmp_valid_dir, 'tmp1.jpg')
        shutil.copy(src_file, path1)
        path2 = os.path.join(self.tmp_valid_dir, 'tmp2.jpg')
        shutil.copy(src_file, path2)
        create_image_file(path1)
        create_image_file(path2)

        # The first path gets a real row; the second is a duplicate.
        first_item = ImageFile.objects.filter(filename=path1)
        self.assertEqual(len(first_item), 1)
        self.assertEqual(ImageFile.objects.filter(filename=path2).count(), 0)
        self.assertTrue(DuplicateFile.objects.filter(filename=path2).exists())

        self.assertTrue(os.path.isfile(first_item[0].thumbnail_small.path))
        self.assertFalse(first_item[0].isProcessed)

    def test_delete_photos(self): ### CHECKED ###

        # Case: We want to delete random photos from the file system, then run
        # the function that cleans that up (delete_removed_photos) and
        # check that they were, in fact, completely removed from the database
        # -- once MISSING_FILE_GRACE_PERIOD has elapsed (see
        # DeleteRemovedPhotosGracePeriodTests for the two-phase behavior
        # itself; this test just confirms the eventual end state, for
        # continuity with its own long-standing name/intent).
        for good in self.goodFiles:
            create_image_file(good)

        # Get number of images
        all_files = ImageFile.objects.all()
        before_len = len(all_files)

        num_to_remove = 3
        # Remove a couple files from the disk
        for n in range(num_to_remove):
            os.remove(self.goodFiles[n])

        delete_removed_photos()
        # First pass only flags them as missing -- not deleted yet.
        self.assertEqual(ImageFile.objects.count(), before_len)
        # Backdate past the grace period, as if they'd been missing a while.
        ImageFile.objects.filter(filename__in=self.goodFiles[:num_to_remove]).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_photos()

        # Test length of database
        updated_files = ImageFile.objects.all()
        self.assertTrue(before_len - len(updated_files) == num_to_remove)
        # Test that the removed files aren't in the DB
        for n in range(num_to_remove):
            f = self.goodFiles[n]
            in_db = ImageFile.objects.filter(filename=f)
            self.assertEqual(len(in_db), 0)

        # And test that the files that weren't removed still are good. 
        for m in range(3, len(self.goodFiles)):
            f = self.goodFiles[m]
            in_db = ImageFile.objects.filter(filename=f)
            self.assertEqual(len(in_db), 1)

        # Make sure that adding again doesn't create more.
        for good in self.goodFiles:
            create_image_file(good)

        updated_files = ImageFile.objects.all()
        self.assertTrue(before_len - len(updated_files) == num_to_remove)

        # And test that the files that weren't removed still are good. 
        for m in range(3, len(self.goodFiles)):
            f = self.goodFiles[m]
            in_db = ImageFile.objects.filter(filename=f)
            self.assertEqual(len(in_db), 1)


    def test_metadata_gps(self): ### CHECKED ### 
        # Case: the database processing should be able to detect 
        # GPS data adequately in files and put that in the database.
        # If a file doesn't have GPS data, it should get a -999 in 
        # both latitude and longitude. 
        # Expected outcome: pretty much that. I'm randomizing 
        # GPS values in ~half of the images, writing to the JPG,
        # and expecting to get that same data back and put it in
        # the database. gpsphoto seems to have small bugs in a couple
        # things, so I'm catching OS and Key errors for that. 

        for file in self.goodFiles:
            # Random select if the photo will have GPS data
            rv = random.randint(0, 1)
            photo = gpsphoto.GPSPhoto(file)
            if rv == 1:
                try:
                    # Create GPSInfo Data Object
                    lat = (random.random() - 0.5) * 180
                    lon = (random.random() - 0.5) * 360
                    info = gpsphoto.GPSInfo((lat, lon))
                    # Modify GPS Data
                    photo.modGPSData(info, file)
                    create_image_file( file )
                    fdata = ImageFile.objects.filter(filename=file)
                    # Due to rounding precision in GPS, not every decimal
                    # is represented, so we assert that the stored and 
                    # actual values are within a small margin.
                    self.assertTrue(abs(fdata[0].gps_lat_decimal - lat) < 0.01)
                    self.assertTrue(abs(fdata[0].gps_lon_decimal - lon) < 0.01)
                except OSError:
                    pass
                except KeyError:
                    pass
            else:
                # Get the file. Some files have GPS built in already, so I either
                # check that there was no GPS there and both fields are set to 
                # default, or make the not-unreasonable assumption that the 
                # lat and lon are different and assert such.
                create_image_file( file )
                fdata = ImageFile.objects.filter(filename=file)
                if fdata[0].gps_lat_decimal == -999:
                    self.assertEqual(fdata[0].gps_lon_decimal, -999)
                else:
                    self.assertNotEqual(fdata[0].gps_lat_decimal, fdata[0].gps_lon_decimal)

    def test_thumbnails(self): ### CHECKED ### 
    # Explicit test of thumbnails. Nothing crazy, and covered by other tests,
    # but it's good to have atomicity of tests.
        file = self.goodFiles[0]
        create_image_file(file)
        f_data = ImageFile.objects.filter(filename = file)[0]
        self.assertTrue(os.path.isfile(f_data.thumbnail_small.path))
        self.assertTrue(os.path.isfile(f_data.thumbnail_small.path))

    def test_move_id_stay_same(self): ### CHECKED ###
        # Case: if a file moves to another location but otherwise stays
        # the same (no edits), it should keep the same ID and keep its 
        # isProcessed status. The path to the thumbnail should change. 
        # The original path should be removed as well. 

        # Add all the files. We're going to move a bunch of them around. 
        for good in self.goodFiles:
            create_image_file(good)

        # Get a list of all files that were initially added. 
        items = ImageFile.objects.all()
        item_files_init = [x.filename for x in items]

        # Move the first file around. Set its isProcessed bit. 
        src_file = self.goodFiles[0]
        first_item = ImageFile.objects.filter(filename=src_file)[0]
        first_item.isProcessed = True
        first_item.save()
        first_item = ImageFile.objects.filter(filename=src_file)[0]
        self.assertTrue(first_item.isProcessed)
        ident1 = first_item.id
        ph = first_item.pixel_hash
        date_add = first_item.dateAdded

        path1 = os.path.join(self.tmp_valid_dir, 'tmp1.jpg')

        shutil.move(src_file, path1)
        # This path (src_file) will be removed from the database,
        # so in the final comparison of the initial files and the 
        # files that are in the database at the end, it should be
        # removed from this list. (Done later)
        create_image_file(path1)
        # Having just moved the file, the isProcessed should be saved
        # in the new database entry. 
        path1_item = ImageFile.objects.filter(filename=path1)[0]
        print(path1_item.id, first_item.id)
        self.assertTrue(path1_item.isProcessed)
        # Identity should be different because it moved to a different
        # file location. 
        self.assertEqual(path1_item.id, first_item.id)
        self.assertNotEqual(path1_item.dateAdded, date_add)
        # src_file should no longer be in the database.
        i1_tmp = ImageFile.objects.filter(filename=src_file)
        self.assertEqual(len(i1_tmp), 0)
        date_add = path1_item.dateAdded

        # Have two of same input -- add another copy of path1 at path2
        # while path1 is still present. UPDATED for the create_image_
        # file() duplicate-detection fix: path2 is now correctly recorded
        # as a DuplicateFile rather than given its own ImageFile row
        # (previously a real bug -- see CLAUDE.md). Moving path2 to path3
        # afterward and rescanning finds path1 still present with the
        # same content, so path3 is recognized as a further duplicate too
        # -- path1 remains the sole real row throughout.
        path2 = os.path.join(self.tmp_valid_dir, 'tmp2.jpg')
        path3 = os.path.join(self.tmp_valid_dir, 'tmp3.jpg')
        shutil.copy(path1, path2)
        create_image_file(path2)
        self.assertEqual(ImageFile.objects.filter(filename=path2).count(), 0)
        self.assertTrue(DuplicateFile.objects.filter(filename=path2).exists())

        shutil.move(path2, path3)
        create_image_file(path3)
        self.assertEqual(ImageFile.objects.filter(filename=path3).count(), 0)
        self.assertTrue(DuplicateFile.objects.filter(filename=path3).exists())
        self.assertFalse(os.path.exists(path2))
        self.assertTrue(os.path.exists(path1_item.filename))
        # Path1 should still be in the database
        p1_tmp = ImageFile.objects.filter(filename=path1)
        self.assertEqual(len(p1_tmp), 1)

        items = ImageFile.objects.all()
        item_files = [x.filename for x in items]
        # UPDATED for the create_image_file() duplicate-detection fix:
        # path3 is a duplicate (see above), not a real row, so only path1
        # was actually added -- src_file removed from the set (renamed to
        # path1), path2/path3 never get their own rows at all.
        self.assertEqual(set(item_files), set(item_files_init + [path1]) - set([src_file]))

        # Should gracefully handle a genuinely NEW (non-duplicate) file
        # being written into a path that previously only held a
        # DuplicateFile record (no ImageFile row to "overwrite" here,
        # unlike the old behavior this test predates). Uses an
        # orientation fixture rather than another "good" fixture -- the
        # goodFiles set includes deliberate same-content repeats under
        # different extensions, which would just create another
        # duplicate here instead of exercising the "new content" path.
        shutil.move(self.orientFiles[0], path3)
        self.assertEqual(ImageFile.objects.filter(filename=path3).count(), 0)

        # path3 already has a stale DuplicateFile record from the move
        # above -- create_image_file()'s very first check wipes any
        # existing DuplicateFile/ImageFile bookkeeping for a path and
        # bails out ("start over"), rather than ingesting in the same
        # call. A second call (matching how the next add_from_root_dir
        # scan would naturally pick it up) then genuinely ingests it.
        create_image_file(path3)
        create_image_file(path3)
        path3_item_aft = ImageFile.objects.filter(filename=path3)[0]
        self.assertEqual(path3_item_aft.filename, path3)

    # # Tested adding bad file names? Not there?
    def test_bogus_file(self): ### CHECKED ### 
        bogus = '/tmp/asdfafdadsf.jpg'
        create_image_file(bogus)
        items = ImageFile.objects.all()
        self.assertEqual(len(items), 0)

    def test_rotated_image_update(self):
        # Case: an image stays in the same location, but is rotated and then
        # re-processed. 
        # Expected outcomes:
        # - ID should change -- we essentially treat it like a brand-new addition.
        #    This makes sense -- we would have to reprocess faces anyway. 
        # - At least the pixels of the thumbnail should change to reflect
        #       the rotation
        # - isProcessed should be reset
        # - dateAdded should update

        for good in self.goodFiles:
            create_image_file(good)

        file1 = self.goodFiles[0]

        first_file_data = ImageFile.objects.filter(filename=file1)[0]
        first_file_data.isProcessed = True
        first_file_data.save()
        first_file_data = ImageFile.objects.filter(filename=file1)[0]
        self.assertTrue(first_file_data.isProcessed)
        initial_date = first_file_data.dateAdded
        image_id = first_file_data.id

        pil_file = Image.open(file1)
        rotated = pil_file.rotate(90)
        rotated.save(file1)

        create_image_file(file1)
        rot_data = ImageFile.objects.filter(filename = file1)[0]
        self.assertNotEqual(first_file_data.id, rot_data.id)
        self.assertEqual(first_file_data.filename, rot_data.filename)
        self.assertFalse(rot_data.isProcessed)
        self.assertNotEqual(first_file_data.pixel_hash, rot_data.pixel_hash)
        self.assertNotEqual(first_file_data.thumbnail_small.path, rot_data.thumbnail_small.path)


    # def test_directories(self):
        # Somehow I'm getting the same directory added twice. Yikes!

    #     # Move to another, unused path: should be same id, but
    #     # a different time.
    #     shutil.move(file1, path1)
    #     create_image_file(path1)
    #     data = ImageFile.objects.filter(filename=path1)
    #     self.assertNotEqual(initial_date, data[0].dateAdded)
    #     self.assertEqual(image_id, data[0].id)
    #     # Initial date has now updated
    #     initial_date = data[0].dateAdded

    #     # # Same test as above, just with new file 
    #     # create_image_file(path1)
    #     # data = ImageFile.objects.filter(filename=path1)
    #     # self.assertEqual(initial_date, data[0].dateAdded)
    #     # self.assertEqual(image_id, data[0].id)

    #     # raise NotImplementedError('Rotation test')

    # def test_make_request_of_image(self):
    #     raise NotImplementedError('Request image')
    #     # We're not able to get to the thumbnail via the admin page.
    #     # Look into that. 
    #     # May also need to look into the media root. 

    # def test_exif_metadata(self):
    #     raise NotImplementedError('Metadata test')

    # def test_rotation_dir(self):
    #     raise NotImplementedError('Need to test out the directory called "orientation".')


    def test_bulk_add(self):

        add_from_root_dir(self.tmp_valid_dir)
        add_from_root_dir(self.tmp_valid_dir)

        valid_files = []

        for root, dirs, files in os.walk(self.tmp_valid_dir):
            for f in files:
                cur_file = os.path.join(root, f)
                if cur_file.lower().endswith( ('.jpg', '.jpeg', ) ):
                    valid_files.append(cur_file)

        files_in_db = ImageFile.objects.all()
        files_in_db = [x.filename for x in files_in_db]
        # The real validation fixture directory intentionally contains a
        # genuine duplicate (tmpmv.jpg, same content as naming/good/space
        # in filename.jpg) -- UPDATED for the create_image_file()
        # duplicate-detection fix: a duplicate is correctly recorded as a
        # DuplicateFile, not given its own ImageFile row (see CLAUDE.md),
        # so "every file is accounted for" now means either one.
        duplicate_files = [d.filename for d in DuplicateFile.objects.all()]

        for vf in valid_files:
            self.assertTrue(vf in files_in_db or vf in duplicate_files)

        dir_objs = Directory.objects.all()
        # print(dir_objs)
        directory_list = [x.dir_path for x in dir_objs]
        self.assertEqual(len(directory_list), len(set(directory_list)))

    def test_add_from_root_dir_skips_entirely_when_lock_already_held(self):
        # Regression test for the settings.LOCKFILE -> Postgres advisory
        # lock change (common/advisory_lock.py): while another instance
        # holds the lock, this call must do nothing at all (not create
        # any ImageFile/Directory rows), same as the old "Locked!" no-op,
        # but via a real held advisory lock instead of a file on disk.
        #
        # Postgres advisory locks are reentrant PER SESSION -- acquiring
        # the same key again from the same connection would just succeed
        # again, not block -- so the "other holder" here has to be a
        # genuinely separate connection, not another `with advisory_lock`
        # on Django's own (shared, single) test connection.
        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        self.addCleanup(other.close)
        key = zlib.crc32(b'filepopulator.add_from_root_dir')
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
            self.assertTrue(cur.fetchone()[0])

            add_from_root_dir(self.tmp_valid_dir)
            self.assertEqual(ImageFile.objects.count(), 0)

            cur.execute("SELECT pg_advisory_unlock(%s)", [key])

        # Once free again, a normal run still works.
        add_from_root_dir(self.tmp_valid_dir)
        self.assertGreater(ImageFile.objects.count(), 0)

    def test_tracked_root_going_empty_is_recorded_as_an_application_error(self):
        # Real, exceptional signal (dropped mount, bad permissions, a
        # renamed root) -- distinct from a brand-new, genuinely-empty
        # root (next test), which must NOT be flagged.
        from common.models import ApplicationError
        add_from_root_dir(self.tmp_valid_dir)
        self.assertGreater(ImageFile.objects.count(), 0)

        shutil.rmtree(self.tmp_valid_dir)
        os.makedirs(self.tmp_valid_dir)

        add_from_root_dir(self.tmp_valid_dir)

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'filepopulator.scripts.add_from_root_dir')
        self.assertIn(self.tmp_valid_dir, error.message)

    def test_brand_new_empty_root_is_not_recorded(self):
        from common.models import ApplicationError
        empty_root = '/tmp/brand_new_empty_root'
        if os.path.exists(empty_root):
            shutil.rmtree(empty_root)
        os.makedirs(empty_root)
        self.addCleanup(shutil.rmtree, empty_root, ignore_errors=True)

        add_from_root_dir(empty_root)

        self.assertEqual(ApplicationError.objects.count(), 0)


class DeleteRemovedPhotosGracePeriodTests(TestCase):
    """ImageFile.missing_since / delete_removed_photos()'s grace period,
    added 2026-10-05 after a real incident: a bulk move (hundreds of
    files, source deleted before the destination copies were all visible
    to the same scan cycle) raced the old immediate-delete behavior,
    permanently losing rows -- and their tagged Face rows -- before a
    later scan's own move-detection (pixel_hash match + old file gone)
    ever got a chance to run. See CLAUDE.md's 2026-10-05 write-up."""

    def setUp(self):
        self.validation_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        self.tmp_valid_dir = '/tmp/img_validation_grace'
        if os.path.exists(self.tmp_valid_dir):
            shutil.rmtree(self.tmp_valid_dir)
        shutil.copytree(self.validation_dir, self.tmp_valid_dir)

        self.good_dir = os.path.join(self.tmp_valid_dir, 'naming', 'good')
        self.goodFiles = []
        for root, dirs, files in os.walk(self.good_dir):
            for fname in files:
                self.goodFiles.append(os.path.join(root, fname))
        assert len(self.goodFiles) >= 2, 'Need at least 2 real fixture files for these tests.'

    def tearDown(self):
        shutil.rmtree(self.tmp_valid_dir, ignore_errors=True)

    def test_first_missing_scan_flags_but_does_not_delete(self):
        gone_path = self.goodFiles[0]
        create_image_file(gone_path)
        self.assertIsNone(ImageFile.objects.get(filename=gone_path).missing_since)

        os.remove(gone_path)
        delete_removed_photos()

        row = ImageFile.objects.get(filename=gone_path)
        self.assertIsNotNone(row.missing_since)

    def test_still_missing_past_grace_period_is_deleted(self):
        gone_path = self.goodFiles[0]
        create_image_file(gone_path)
        os.remove(gone_path)

        ImageFile.objects.filter(filename=gone_path).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_photos()

        self.assertFalse(ImageFile.objects.filter(filename=gone_path).exists())

    def test_missing_within_grace_period_is_not_yet_deleted(self):
        gone_path = self.goodFiles[0]
        create_image_file(gone_path)
        os.remove(gone_path)

        ImageFile.objects.filter(filename=gone_path).update(
            missing_since=timezone.now() - timedelta(hours=1)
        )
        delete_removed_photos()

        self.assertTrue(ImageFile.objects.filter(filename=gone_path).exists())

    def test_file_reappearing_clears_missing_since(self):
        # Simulates the real race this was built for: a file goes missing
        # (e.g. mid-move), gets flagged, then the destination copy is
        # recognized as a move and this row's path/content is updated --
        # at which point the file backing this row exists again, and the
        # next scan should self-heal the flag rather than leave it stale.
        path = self.goodFiles[0]
        create_image_file(path)
        ImageFile.objects.filter(filename=path).update(missing_since=timezone.now())

        delete_removed_photos()

        row = ImageFile.objects.get(filename=path)
        self.assertIsNone(row.missing_since)


class MovedFileFaceContinuityTests(TestCase):
    """Real incident, 2026-10-08: a photo that moved several times over
    months ended up with its validated Face classifications permanently
    separated from the file's current location. Investigation found the
    pixel_hash-based "same photo, just moved" reuse branch in
    create_image_file() never once linked any of this file's relocations,
    across its whole history -- every move created a brand-new row with
    fresh, unvalidated detections, while the old row (and its validated
    faces) was left behind at a now-nonexistent path. The exact root cause
    of why the hash lookup never matched in that specific case was never
    fully pinned down (no filesystem-history data available to check what
    was really on disk at each past point in time) -- these tests instead
    cover the full range of ways create_image_file()/delete_removed_photos()
    *should* behave for a move, so a regression here is caught going
    forward even without a confirmed repro of the original failure."""

    def setUp(self):
        self.validation_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        self.tmp_valid_dir = '/tmp/img_validation_move_continuity'
        if os.path.exists(self.tmp_valid_dir):
            shutil.rmtree(self.tmp_valid_dir)
        shutil.copytree(self.validation_dir, self.tmp_valid_dir)

        self.good_dir = os.path.join(self.tmp_valid_dir, 'naming', 'good')
        self.goodFiles = []
        for root, dirs, files in os.walk(self.good_dir):
            for fname in files:
                self.goodFiles.append(os.path.join(root, fname))
        assert len(self.goodFiles) >= 2, 'Need at least 2 real fixture files for these tests.'

    def tearDown(self):
        shutil.rmtree(self.tmp_valid_dir, ignore_errors=True)

    def _attach_validated_face(self, img, person_name='Moved File Continuity Test Person'):
        from face_manager.models import Person
        person, _ = Person.objects.get_or_create(person_name=person_name)
        face = Face(
            source_image_file=img,
            declared_name=person,
            validated=True,
            box_left=1, box_top=1,
            box_right=min(40, img.width - 1), box_bottom=min(40, img.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_clean_move_in_one_scan_cycle_preserves_validated_face(self):
        # The simple, working case: old file gone and new file present by
        # the time a single create_image_file() call discovers the new
        # path -- the single-match reuse branch should rename the existing
        # row in place, carrying its validated Face along with it.
        old_path = self.goodFiles[0]
        create_image_file(old_path)
        img = ImageFile.objects.get(filename=old_path)
        old_id = img.id
        face = self._attach_validated_face(img)

        new_path = os.path.join(self.tmp_valid_dir, 'moved.jpg')
        shutil.move(old_path, new_path)
        create_image_file(new_path)

        self.assertFalse(ImageFile.objects.filter(filename=old_path).exists())
        moved = ImageFile.objects.get(filename=new_path)
        self.assertEqual(moved.id, old_id)
        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, moved.id)
        self.assertTrue(face.validated)
        self.assertEqual(face.declared_name.person_name, 'Moved File Continuity Test Person')

    def test_move_discovered_while_old_row_still_flagged_missing_preserves_face(self):
        # The row was already flagged missing_since by an earlier
        # delete_removed_photos() pass (even one well past the grace
        # period -- a scan simply hasn't gotten around to deleting it yet)
        # but hasn't actually been deleted. The new path is then
        # discovered -- create_image_file()'s hash lookup doesn't filter
        # on missing_since at all, so this should still reuse the row and
        # preserve the face, regardless of how overdue the grace period
        # is. This is the behavior that matters most going forward: as
        # long as add_from_root_dir() (which discovers new paths) runs
        # before delete_removed_photos() actually deletes the row in the
        # same or an earlier cycle, a move stays linked.
        old_path = self.goodFiles[0]
        create_image_file(old_path)
        img = ImageFile.objects.get(filename=old_path)
        old_id = img.id
        face = self._attach_validated_face(img)

        new_path = os.path.join(self.tmp_valid_dir, 'moved.jpg')
        shutil.move(old_path, new_path)

        # Simulate an earlier scan having already flagged this row missing,
        # well past the grace period, without yet deleting it.
        ImageFile.objects.filter(pk=old_id).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(days=30)
        )

        create_image_file(new_path)

        moved = ImageFile.objects.get(filename=new_path)
        self.assertEqual(moved.id, old_id)
        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, moved.id)
        self.assertTrue(face.validated)

        # And the stale missing_since should self-heal on the next pass,
        # now that the row's file genuinely exists again at the new path.
        delete_removed_photos()
        moved.refresh_from_db()
        self.assertIsNone(moved.missing_since)

    def test_old_row_already_deleted_before_new_path_discovered_loses_face_history(self):
        # The known, accepted gap: if delete_removed_photos() actually
        # deletes the old row (grace period genuinely expired) in a scan
        # cycle that completes *before* the new path is ever discovered --
        # i.e. the two halves of a move are more than one full cycle
        # apart, each handled by a separate add_from_root_dir()/
        # delete_removed_photos() pairing -- the link is gone for good.
        # The next discovery of the new path creates a brand-new row with
        # no face history. This test exists to make that limitation
        # explicit and regression-checked, not to assert it's acceptable
        # forever.
        old_path = self.goodFiles[0]
        create_image_file(old_path)
        img = ImageFile.objects.get(filename=old_path)
        old_id = img.id
        self._attach_validated_face(img)

        new_path = os.path.join(self.tmp_valid_dir, 'moved.jpg')
        shutil.move(old_path, new_path)

        # Old row already past its grace period, and a scan cycle actually
        # runs delete_removed_photos() -- with the new path NOT yet
        # discovered (simulating it landing in a later, separate cycle).
        ImageFile.objects.filter(pk=old_id).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_photos()
        self.assertFalse(ImageFile.objects.filter(pk=old_id).exists())
        self.assertFalse(Face.objects.filter(source_image_file_id=old_id).exists())

        # Only now does a later cycle discover the new path.
        create_image_file(new_path)
        moved = ImageFile.objects.get(filename=new_path)
        self.assertNotEqual(moved.id, old_id)
        self.assertEqual(Face.objects.filter(source_image_file=moved).count(), 0)

    def test_two_rows_share_hash_one_missing_one_present_reuses_only_the_missing_one(self):
        # Reproduces the contaminated state actually found in production:
        # two live ImageFile rows ended up sharing one pixel_hash (one
        # orphaned at a dead path, one real). This isn't something
        # create_image_file() can produce on its own -- its own duplicate
        # check prevents it -- so it's constructed directly here, the same
        # way the real contamination must have arisen outside this
        # function's own normal control flow. A third path with the same
        # content should correctly reuse/rename the row whose file is
        # missing, and must never touch the row whose file is still
        # genuinely present.
        present_path = self.goodFiles[0]
        create_image_file(present_path)
        present_row = ImageFile.objects.get(filename=present_path)
        present_face = self._attach_validated_face(present_row, person_name='Present Row Person')

        orphan_path = os.path.join(self.tmp_valid_dir, 'orphan_source.jpg')
        shutil.copy(present_path, orphan_path)
        orphan_row = ImageFile.objects.create(
            filename=orphan_path,
            directory=present_row.directory,
            pixel_hash=present_row.pixel_hash,
            file_hash=present_row.file_hash,
            width=present_row.width,
            height=present_row.height,
            orientation=present_row.orientation,
            dateAdded=timezone.now(),
            dateModified=timezone.now(),
        )
        orphan_face = self._attach_validated_face(orphan_row, person_name='Orphan Row Person')
        # The orphan's own file is now gone -- it was only ever a stand-in
        # to get two live rows sharing a hash; what matters is that its
        # *file* is missing while the other row's file remains present.
        os.remove(orphan_path)

        new_path = os.path.join(self.tmp_valid_dir, 'reunited.jpg')
        shutil.copy(present_path, new_path)
        create_image_file(new_path)

        # The orphan row should have been renamed/reused -- its face
        # travels with it.
        self.assertFalse(ImageFile.objects.filter(pk=orphan_row.pk, filename=orphan_path).exists())
        reused = ImageFile.objects.get(filename=new_path)
        self.assertEqual(reused.id, orphan_row.id)
        orphan_face.refresh_from_db()
        self.assertEqual(orphan_face.source_image_file_id, reused.id)

        # The genuinely-present row must be completely untouched.
        present_row.refresh_from_db()
        self.assertEqual(present_row.filename, present_path)
        self.assertTrue(os.path.exists(present_path))
        present_face.refresh_from_db()
        self.assertEqual(present_face.source_image_file_id, present_row.id)

    def test_duplicate_recorded_not_reused_when_old_file_still_present(self):
        # A true copy (not a move): the original file is still on disk at
        # scan time. The new path should be recorded as a DuplicateFile
        # pointing at the original, not given its own ImageFile row --
        # confirms this case (what the investigation suspected might
        # explain the "overwriting a subset" comment) stays safe and
        # doesn't fabricate a second real row either.
        original_path = self.goodFiles[0]
        create_image_file(original_path)
        original = ImageFile.objects.get(filename=original_path)
        face = self._attach_validated_face(original)

        copy_path = os.path.join(self.tmp_valid_dir, 'copy.jpg')
        shutil.copy(original_path, copy_path)
        create_image_file(copy_path)

        self.assertFalse(ImageFile.objects.filter(filename=copy_path).exists())
        self.assertTrue(DuplicateFile.objects.filter(filename=copy_path, original=original).exists())
        self.assertTrue(os.path.exists(original_path))
        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, original.id)


@override_settings(MEDIA_ROOT='/tmp/cleanup_orphaned_media_test')
class CleanupOrphanedMediaFilesTests(TestCase):
    """cleanup_orphaned_media_files, added 2026-10-08 alongside the
    thumbnail_big/thumbnail_medium removal -- those two directories are
    now entirely dead (every file in them is an orphan by definition,
    since no field references them anymore), and checking confirmed the
    *other* thumbnail directories (thumbnails_small, shared by ImageFile/
    VideoFile; face_thumbnails; face_highlights) have real, substantial
    pre-existing orphan backlogs too (re-processing/restore incidents
    leaving behind files a row no longer points to)."""

    def setUp(self):
        self.media_root = settings.MEDIA_ROOT
        if os.path.exists(self.media_root):
            shutil.rmtree(self.media_root)
        os.makedirs(self.media_root)
        self.old_time = time.time() - 48 * 3600  # comfortably past the 24h default

    def tearDown(self):
        shutil.rmtree(self.media_root, ignore_errors=True)

    def _write_file(self, rel_path, content=b'x', mtime=None):
        full_path = os.path.join(self.media_root, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(content)
        if mtime is not None:
            os.utime(full_path, (mtime, mtime))
        return full_path

    def test_thumbnails_big_and_med_are_wiped_unconditionally(self):
        # No field references these directories at all anymore -- every
        # file in them is orphaned by construction, regardless of any DB
        # state.
        big_path = self._write_file('thumbnails_big/ab/c/old.jpg', mtime=self.old_time)
        med_path = self._write_file('thumbnails_med/ab/c/old.jpg', mtime=self.old_time)

        call_command('cleanup_orphaned_media_files')

        self.assertFalse(os.path.exists(big_path))
        self.assertFalse(os.path.exists(med_path))

    def test_thumbnails_small_referenced_by_imagefile_is_kept(self):
        rel_path = 'thumbnails_small/ab/c/kept.jpg'
        full_path = self._write_file(rel_path, mtime=self.old_time)
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/co_test')
        ImageFile.objects.bulk_create([ImageFile(
            filename='/tmp/co_test/a.jpg', directory=directory,
            thumbnail_small=rel_path,
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
        )])

        call_command('cleanup_orphaned_media_files')

        self.assertTrue(os.path.exists(full_path))

    def test_thumbnails_small_orphan_is_deleted(self):
        orphan_path = self._write_file('thumbnails_small/ab/c/orphan.jpg', mtime=self.old_time)

        call_command('cleanup_orphaned_media_files')

        self.assertFalse(os.path.exists(orphan_path))

    def test_thumbnails_small_referenced_by_videofile_is_kept(self):
        rel_path = 'thumbnails_small/ab/c/kept_video.jpg'
        full_path = self._write_file(rel_path, mtime=self.old_time)
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/co_test_video')
        VideoFile.objects.bulk_create([VideoFile(
            filename='/tmp/co_test_video/a.mp4', directory=directory,
            thumbnail_small=rel_path,
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
        )])

        call_command('cleanup_orphaned_media_files')

        self.assertTrue(os.path.exists(full_path))

    def test_face_thumbnail_referenced_kept_orphan_deleted(self):
        from face_manager.models import Person

        kept_rel = 'face_thumbnails/ab/c/kept.jpg'
        orphan_rel = 'face_thumbnails/ab/c/orphan.jpg'
        kept_path = self._write_file(kept_rel, mtime=self.old_time)
        orphan_path = self._write_file(orphan_rel, mtime=self.old_time)

        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/co_test_face')
        img = ImageFile(
            filename='/tmp/co_test_face/a.jpg', directory=directory,
            thumbnail_small='',
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
        )
        ImageFile.objects.bulk_create([img])
        person, _ = Person.objects.get_or_create(person_name='Cleanup Orphan Test Person')
        Face.objects.bulk_create([Face(
            source_image_file=img, declared_name=person,
            face_thumbnail=kept_rel,
            box_left=1, box_top=1, box_right=5, box_bottom=5,
        )])

        call_command('cleanup_orphaned_media_files')

        self.assertTrue(os.path.exists(kept_path))
        self.assertFalse(os.path.exists(orphan_path))

    def test_person_highlight_referenced_kept_orphan_deleted(self):
        from face_manager.models import Person

        kept_rel = 'face_highlights/a/kept.jpg'
        orphan_rel = 'face_highlights/a/orphan.jpg'
        kept_path = self._write_file(kept_rel, mtime=self.old_time)
        orphan_path = self._write_file(orphan_rel, mtime=self.old_time)

        Person.objects.filter(person_name='Cleanup Highlight Test Person').delete()
        Person.objects.bulk_create([Person(
            person_name='Cleanup Highlight Test Person', highlight_img=kept_rel,
        )])

        call_command('cleanup_orphaned_media_files')

        self.assertTrue(os.path.exists(kept_path))
        self.assertFalse(os.path.exists(orphan_path))

    def test_recently_modified_orphan_is_not_deleted(self):
        # Guards against the real race: a thumbnail file lands on disk
        # moments before its owning row's save() actually commits -- a
        # file that looks unreferenced in a query that ran a moment ago
        # could just be mid-flight, not a genuine orphan.
        recent_orphan = self._write_file('thumbnails_small/ab/c/brand_new.jpg')  # mtime = now

        call_command('cleanup_orphaned_media_files')

        self.assertTrue(os.path.exists(recent_orphan))

    def test_dry_run_deletes_nothing(self):
        big_path = self._write_file('thumbnails_big/ab/c/old.jpg', mtime=self.old_time)
        orphan_path = self._write_file('thumbnails_small/ab/c/orphan.jpg', mtime=self.old_time)

        call_command('cleanup_orphaned_media_files', '--dry-run')

        self.assertTrue(os.path.exists(big_path))
        self.assertTrue(os.path.exists(orphan_path))

    def test_scheduled_task_deletes_orphans_and_respects_the_lock(self):
        from filepopulator.tasks import cleanup_orphaned_media

        orphan_path = self._write_file('thumbnails_big/ab/c/old.jpg', mtime=self.old_time)
        cleanup_orphaned_media()
        self.assertFalse(os.path.exists(orphan_path))

        # While another session holds the advisory lock, the task must
        # do nothing at all -- same contract as add_from_root_dir()'s own
        # lock (see test_add_from_root_dir_skips_entirely_when_lock_already_held
        # for why a genuinely separate connection is needed here, not
        # another `with advisory_lock` on Django's own test connection).
        second_orphan = self._write_file('thumbnails_big/ab/c/second.jpg', mtime=self.old_time)
        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        self.addCleanup(other.close)
        key = zlib.crc32(b'filepopulator.cleanup_orphaned_media')
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
            self.assertTrue(cur.fetchone()[0])

            cleanup_orphaned_media()
            self.assertTrue(os.path.exists(second_orphan))

            cur.execute("SELECT pg_advisory_unlock(%s)", [key])

        # Once free again, a normal run still works.
        cleanup_orphaned_media()
        self.assertFalse(os.path.exists(second_orphan))


class DirectoryTests(TestCase):
    @override_settings(MEDIA_ROOT='/tmp')

    def setUp(self):
        pass

    @classmethod
    def setUpTestData(cls):

        cls.validation_dir = settings.FILEPOPULATOR_VAL_DIRECTORY 

        assert os.path.isdir(cls.validation_dir), 'Validation directory in FaceManageTests does not exist.'

        # Copy the validation files to the /tmp directory
        cls.tmp_valid_dir = '/tmp/img_validation'

        if os.path.exists(cls.tmp_valid_dir):
            shutil.rmtree(cls.tmp_valid_dir)

        shutil.copytree(cls.validation_dir, cls.tmp_valid_dir)

        cls.face_file = os.path.join(cls.tmp_valid_dir, 'has_face_tags.jpg')

        create_image_file(cls.face_file)
        add_from_root_dir(cls.tmp_valid_dir)

    def test_top_name(self):
        
        dirs = Directory.objects.all()
        for d in dirs:
            tln = d.dir_path.split('/')[-1]
            print(tln, d.top_level_name())
            self.assertEqual(tln, d.top_level_name())

    def test_get_average_age(self):
        # Regression test for a fixed bug: average_date_taken()/
        # beginning_date_taken() used to call `timezone.utc`, an attribute
        # removed from django.utils.timezone in the Django version this app
        # now runs (6.0), raising AttributeError on every scheduled
        # filepopulator.update_dir_dates run in production. Now uses
        # pytz.utc (datetime.timezone.utc doesn't work here either, since
        # this module's `from datetime import datetime` shadows the
        # `datetime` module name with the class).
        dirs = Directory.objects.all()

        for d in dirs:
            print("Average before: ", d.mean_datesec)
            self.assertEqual(d.mean_datesec, -1)

        update_dirs_datetime()

        dirs = Directory.objects.all()
        for d in dirs:
            print("Average after: ", d.mean_datesec)
            self.assertNotEqual(d.mean_datesec, -1)


class DirectoryDateFallbackTests(TestCase):
    """Regression test for a real bug found via a user report: a directory
    whose images all have dateTakenValid=False (no trustworthy EXIF date,
    so dateTaken fell back to file-processing time) used to make
    average_date_taken()/beginning_date_taken() fall back to datetime.now()
    -- so every scheduled filepopulator.update_dir_dates run kept
    overwriting the directory's date to "whenever the task last ran"
    instead of anything related to the photos themselves. Now falls back
    to the images' actual (if EXIF-unconfirmed) dateTaken values first,
    only using now() when a directory has no images with any dateTaken at
    all."""

    def _make_bare_image_file(self, filename, date_taken, date_taken_valid):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            dateTaken=date_taken, dateTakenValid=date_taken_valid,
        )
        ImageFile.objects.bulk_create([obj])
        return directory

    def test_falls_back_to_unconfirmed_date_taken_instead_of_now(self):
        from django.utils import timezone as dj_timezone
        stale_date = dj_timezone.make_aware(datetime(2018, 6, 15))

        directory = self._make_bare_image_file(
            "/tmp/dir_date_fallback_test/a.jpg", stale_date, False,
        )
        self._make_bare_image_file(
            "/tmp/dir_date_fallback_test/b.jpg", stale_date, False,
        )

        directory.average_date_taken()
        directory.beginning_date_taken()

        self.assertEqual(directory.mean_datetime.date(), stale_date.date())
        self.assertEqual(directory.first_datetime.date(), stale_date.date())
        now = dj_timezone.now()
        self.assertNotEqual(directory.mean_datetime.year, now.year)

    def test_still_falls_back_to_now_when_no_dates_at_all(self):
        from django.utils import timezone as dj_timezone
        directory, _ = Directory.objects.get_or_create(dir_path="/tmp/dir_date_fallback_test_empty")

        directory.average_date_taken()
        directory.beginning_date_taken()

        now = dj_timezone.now()
        self.assertEqual(directory.mean_datetime.date(), now.date())
        self.assertEqual(directory.first_datetime.date(), now.date())

    def test_prefers_valid_dates_over_unconfirmed_ones(self):
        from django.utils import timezone as dj_timezone
        stale_date = dj_timezone.make_aware(datetime(2018, 6, 15))
        valid_date = dj_timezone.make_aware(datetime(2019, 3, 1))

        directory = self._make_bare_image_file(
            "/tmp/dir_date_fallback_test_mixed/a.jpg", stale_date, False,
        )
        self._make_bare_image_file(
            "/tmp/dir_date_fallback_test_mixed/b.jpg", valid_date, True,
        )

        directory.average_date_taken()

        self.assertEqual(directory.mean_datetime.date(), valid_date.date())


@override_settings(MEDIA_ROOT='/tmp/filepopulator_test_media')
class CheckFileModsTests(TestCase):
    def setUp(self):
        self.tmp_dir = '/tmp/filepop_mod_test'
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        src = os.path.join(settings.FILEPOPULATOR_VAL_DIRECTORY, 'naming', 'good', '1.JPG')
        self.file_path = os.path.join(self.tmp_dir, '1.JPG')
        shutil.copy(src, self.file_path)
        create_image_file(self.file_path)

    def tearDown(self):
        for obj in ImageFile.objects.all():
            obj.delete()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_modification_leaves_record_unchanged(self):
        before = ImageFile.objects.get(filename=self.file_path)
        check_file_mods()
        after = ImageFile.objects.get(filename=self.file_path)
        self.assertEqual(before.id, after.id)
        self.assertEqual(before.pixel_hash, after.pixel_hash)

    def test_modified_file_gets_reprocessed(self):
        before = ImageFile.objects.get(filename=self.file_path)
        before.isProcessed = True
        before.save()

        # Overwrite with different content and push the mtime forward so
        # check_file_mods() considers it modified.
        different_src = os.path.join(settings.FILEPOPULATOR_VAL_DIRECTORY, 'naming', 'good', '2.jpg')
        shutil.copy(different_src, self.file_path)
        future = time.time() + 3600
        os.utime(self.file_path, (future, future))

        check_file_mods()

        after = ImageFile.objects.get(filename=self.file_path)
        self.assertNotEqual(before.pixel_hash, after.pixel_hash)
        # Reprocessing a changed file resets isProcessed -- see
        # instance_clean_and_save()/ImageFile.save() cleaning out stale
        # per-image state whenever the pixel content actually changes.
        self.assertFalse(after.isProcessed)

    def test_missing_file_is_left_alone_by_check_file_mods(self):
        # check_file_mods() only looks at files that still exist
        # (`if os.path.exists(filename)`) -- a deleted file is a job for
        # delete_removed_photos(), not this function. Documenting that
        # boundary explicitly since it's easy to assume this function
        # handles deletions too.
        os.remove(self.file_path)
        check_file_mods()
        self.assertTrue(ImageFile.objects.filter(filename=self.file_path).exists())


@override_settings(MEDIA_ROOT='/tmp/filepopulator_test_media')
class OrientationChangeReprocessTests(TestCase):
    """Regression test for a fixed bug: when create_image_file() detects
    the same pixel hash but a different orientation than what's stored, it
    reset isProcessed=False to trigger redetection but never cleared the
    image's existing Face rows. Those faces' box coordinates were computed
    against the *old* orientation/rotation and don't correspond to the
    newly (correctly) rotated pixel data, so they'd sit alongside fresh
    detections with incompatible coordinates -- exactly the shape of stale
    data that made find_and_encode_faces() crash in production (see
    face_manager.face_extract_encode's update_list_of_no_matching_detects()
    for the matching defensive fix on that side)."""

    def test_orientation_change_clears_stale_faces_and_marks_unprocessed(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        create_image_file(path)
        img = ImageFile.objects.get(filename=path)

        face = Face(
            source_image_file=img,
            box_left=1, box_top=1, box_right=min(40, img.width - 1), box_bottom=min(40, img.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()

        # Simulate "orientation as computed now differs from what's
        # stored" without touching the file itself. Also back-date the
        # stored dateModified (rather than the read-only fixture file's
        # actual mtime) so create_image_file()'s "timestamp unchanged,
        # skip" short-circuit doesn't apply and it actually recomputes.
        from datetime import timedelta
        ImageFile.objects.filter(pk=img.pk).update(
            orientation=img.orientation + 1,
            isProcessed=True,
            dateModified=img.dateModified - timedelta(days=1),
        )

        create_image_file(path)

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())
        # Regression check for a second, related fixed bug: this branch
        # used to reassign exist_photo to an unsaved instance (no pk),
        # so .save() inserted a *second* row for the same filename
        # instead of updating the existing one.
        self.assertEqual(ImageFile.objects.filter(filename=path).count(), 1)
        img.refresh_from_db()
        self.assertFalse(img.isProcessed)

    def test_orientation_change_deletes_the_old_thumbnail_file(self):
        # Real incident, 2026-10-08: this branch reassigns exist_photo to
        # a freshly-constructed instance (no thumbnail_small set), so
        # save() generates a brand-new thumbnail file under a new random
        # name -- the OLD one was never deleted, silently orphaning it.
        # Found while root-causing a real 436K-file orphaned-thumbnail
        # backlog; this exact branch was the single biggest source of it.
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        create_image_file(path)
        img = ImageFile.objects.get(filename=path)
        old_thumbnail_path = img.thumbnail_small.path
        self.assertTrue(os.path.isfile(old_thumbnail_path))

        from datetime import timedelta
        ImageFile.objects.filter(pk=img.pk).update(
            orientation=img.orientation + 1,
            isProcessed=True,
            dateModified=img.dateModified - timedelta(days=1),
        )

        create_image_file(path)

        self.assertFalse(os.path.isfile(old_thumbnail_path))
        new_img = ImageFile.objects.get(filename=path)
        self.assertTrue(os.path.isfile(new_img.thumbnail_small.path))
        self.assertNotEqual(new_img.thumbnail_small.path, old_thumbnail_path)


class DeleteOldThumbnailsTests(TestCase):
    """Real incident, 2026-10-06: a restored-from-backup database can
    reference thumbnail files that no longer exist on disk (regenerated
    or cleaned up since the backup was taken). os.remove() raising
    FileNotFoundError used to abort delete_old_thumbnails() entirely --
    this is purely opportunistic cleanup, so a missing file should be a
    no-op, not a crash that blocks the caller's own move/reprocess logic
    from ever running."""

    def test_missing_thumbnail_file_does_not_raise(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        create_image_file(path)
        img = ImageFile.objects.get(filename=path)

        # Simulate exactly the real scenario: the field still points at a
        # name, but the file itself is already gone.
        os.remove(img.thumbnail_small.path)

        delete_old_thumbnails(img)  # must not raise

        self.assertIsNone(img.thumbnail_small.name)

    def test_still_deletes_files_that_do_exist(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        create_image_file(path)
        img = ImageFile.objects.get(filename=path)
        small_path = img.thumbnail_small.path
        self.assertTrue(os.path.isfile(small_path))

        delete_old_thumbnails(img)

        self.assertFalse(os.path.isfile(small_path))


@override_settings(MEDIA_ROOT='/tmp/filepopulator_test_media')
class CorruptedImageIngestionTests(TestCase):
    """Uses the 5 real corrupted JPEGs pulled from production logs (see
    /mnt/fast_storage/appdata/django_picasa/test_suite/corrupted_images/NOTES.md),
    mounted read-only under /photos/corrupted."""

    CORRUPTED_DIR = '/photos/corrupted'

    def tearDown(self):
        for obj in ImageFile.objects.all():
            obj.delete()
        FailedImageFile.objects.all().delete()

    def test_create_image_file_tracks_never_ingested_corrupted_file(self):
        # Regression test for a fixed bug: ImageFile._generate_md5_hash()
        # used to only catch TypeError/PIL.Image.DecompressionBombError,
        # not the OSError ("image file is truncated" / "broken data
        # stream") a corrupted JPEG actually raises -- so
        # create_image_file() crashed outright instead of degrading
        # gracefully. Now the OSError is caught and, since this file was
        # never successfully ingested (no ImageFile row exists to record
        # the failure on), it's tracked in FailedImageFile instead.
        # Pick a fixture that's actually unrecoverable rather than a
        # hardcoded real-production filename -- the real fixture set (5
        # files, see NOTES.md) mounted locally has a mix of recoverable and
        # unrecoverable files (see the next test's docstring), while CI's
        # synthetic ci_fixtures/corrupted/ (2 files) are both deliberately
        # deep-truncated and always unrecoverable. Try each present file
        # until one actually lands in FailedImageFile.
        path = None
        for filename in os.listdir(self.CORRUPTED_DIR):
            candidate = os.path.join(self.CORRUPTED_DIR, filename)
            create_image_file(candidate)
            if FailedImageFile.objects.filter(filename=candidate).exists():
                path = candidate
                break
        self.assertIsNotNone(path, "no unrecoverable fixture found in CORRUPTED_DIR")

        self.assertFalse(ImageFile.objects.filter(filename=path).exists())
        failed = FailedImageFile.objects.get(filename=path)
        self.assertTrue(failed.error_message)

    def test_add_from_root_dir_tracks_and_stops_retrying_corrupted_files(self):
        # add_from_root_dir()'s per-file try/except already meant one
        # corrupted file didn't take down the whole batch. The bug was
        # that these files also never got any record at all, so they were
        # re-attempted (and re-failed) on every single scheduled
        # populate_files_from_root run, forever. Now confirms both halves
        # of the fix: every file ends up accounted for (either a real
        # ImageFile or a FailedImageFile -- some of these "corrupted"
        # fixtures are truncated shallowly enough that PIL/cv2's fallback
        # decoding actually recovers them, which is fine, not every
        # corrupted file is unrecoverable), and a second run doesn't
        # re-attempt whichever ones did fail (last_attempted_at is
        # untouched -- if it *had* retried, FailedImageFile.
        # objects.update_or_create() would have bumped it via auto_now).
        add_from_root_dir(self.CORRUPTED_DIR)

        corrupted_files = os.listdir(self.CORRUPTED_DIR)
        # >=5 locally (the real fixture set); CI's synthetic
        # ci_fixtures/corrupted/ only has 2 -- just confirm there's at least
        # something to test against, not a specific count.
        self.assertGreaterEqual(len(corrupted_files), 1)
        self.assertEqual(
            ImageFile.objects.count() + FailedImageFile.objects.count(),
            len(corrupted_files),
        )
        self.assertGreater(FailedImageFile.objects.count(), 0)
        first_pass_timestamps = {
            f.filename: f.last_attempted_at for f in FailedImageFile.objects.all()
        }

        add_from_root_dir(self.CORRUPTED_DIR)

        second_pass_timestamps = {
            f.filename: f.last_attempted_at for f in FailedImageFile.objects.all()
        }
        self.assertEqual(first_pass_timestamps, second_pass_timestamps)

    def test_previously_good_photo_that_becomes_corrupted_is_flagged_not_crashed(self):
        # The other half of the bug: a photo that was ingested fine and
        # later becomes unreadable on disk (bit rot, a bad in-place edit)
        # hits the same OSError, but through the "photo already exists"
        # branch of create_image_file() -- there's already a real
        # ImageFile row for it. That row is flagged via
        # image_load_failed/image_load_error (added for the face_manager
        # retry-forever fix) rather than losing its prior good data, and
        # dateModified is bumped so it isn't re-attempted every run either.
        tmp_dir = '/tmp/filepop_corrupt_existing_test'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        try:
            src = os.path.join(settings.FILEPOPULATOR_VAL_DIRECTORY, 'naming', 'good', '1.JPG')
            file_path = os.path.join(tmp_dir, '1.JPG')
            shutil.copy(src, file_path)
            create_image_file(file_path)
            before = ImageFile.objects.get(filename=file_path)
            self.assertFalse(before.image_load_failed)

            # Corrupt it in place (truncate) and push the mtime forward so
            # create_image_file() treats it as changed, same technique
            # ci_fixtures/generate_fixtures.py uses to build corrupted
            # fixtures.
            with open(file_path, 'rb') as f:
                data = f.read()
            with open(file_path, 'wb') as f:
                f.write(data[: len(data) - 200])
            future = time.time() + 3600
            os.utime(file_path, (future, future))

            create_image_file(file_path)

            after = ImageFile.objects.get(filename=file_path)
            self.assertEqual(before.id, after.id)
            self.assertTrue(after.image_load_failed)
            self.assertTrue(after.image_load_error)
            # Prior good data (pixel_hash, thumbnails, etc.) is untouched.
            self.assertEqual(before.pixel_hash, after.pixel_hash)
            self.assertFalse(FailedImageFile.objects.filter(filename=file_path).exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@override_settings(MEDIA_ROOT='/tmp/filepopulator_test_media')
class HeicIngestionTests(TestCase):
    """HEIC (Apple's default photo format since iOS 11) support.

    HEIC_DIR mirrors real fixture data locally: 8 real iPhone photos
    (models 12 through 17 Pro) at
    /mnt/fast_storage/appdata/django_picasa/test_suite/heic_images/,
    mounted read-only under /photos/heic_stub. CI only has the single
    synthetic no-EXIF stub from ci_fixtures/heic_stub/ (see
    generate_fixtures.py) -- tests here work with "whatever's present"
    rather than assuming a specific count or specific metadata, so they
    hold in both environments.

    Verified empirically (see CLAUDE.md) against all 8 real samples:
    pillow_heif/libheif always decodes to plain RGB, and usually reports
    EXIF orientation 1 regardless of the photo's actual portrait/
    landscape framing -- meaning it usually auto-applies any container-
    level rotation transform during decode and resets the tag to match.
    A real production file (IMG_9370.HEIC, orientation 8) showed that
    doesn't always hold, so a non-1 orientation is now rotated via the
    same apply_exif_orientation() JPEG uses rather than rejected. The
    multi-frame (Live Photo/burst) guard below still fails loudly
    (recorded via FailedImageFile/image_load_failed, not silently) since
    that case is still genuinely unsupported.
    """

    HEIC_DIR = '/photos/heic_stub'

    def tearDown(self):
        for obj in ImageFile.objects.all():
            obj.delete()
        FailedImageFile.objects.all().delete()

    def _heic_files(self):
        return [
            f for f in sorted(os.listdir(self.HEIC_DIR))
            if f.lower().endswith(('.heic', '.heif'))
        ]

    def test_ingests_every_heic_fixture_successfully(self):
        heic_files = self._heic_files()
        self.assertGreaterEqual(len(heic_files), 1)
        for filename in heic_files:
            path = os.path.join(self.HEIC_DIR, filename)
            create_image_file(path)
            img = ImageFile.objects.filter(filename=path).first()
            self.assertIsNotNone(img, f"{filename} was not ingested")
            self.assertEqual(img.orientation, 1)
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)
            self.assertTrue(os.path.isfile(img.thumbnail_small.path))

    def test_add_from_root_dir_discovers_heic_files(self):
        # Regression test for a fixed bug: add_from_root_dir() used to
        # filter to `.jpg`/`.jpeg` before even building its file list, so
        # .heic files were invisible to the whole ingestion pipeline from
        # the start, not just rejected.
        add_from_root_dir(self.HEIC_DIR)
        self.assertEqual(ImageFile.objects.count(), len(self._heic_files()))

    def test_gps_decimal_conversion_is_sane_when_present(self):
        # Not every fixture (or CI's synthetic stub) has GPS data -- only
        # assert against ones that actually do.
        found_any = False
        for filename in self._heic_files():
            path = os.path.join(self.HEIC_DIR, filename)
            create_image_file(path)
            img = ImageFile.objects.get(filename=path)
            if img.gps_lat_decimal == -999 and img.gps_lon_decimal == -999:
                # -999 is ImageFile's own "no GPS data" sentinel default.
                continue
            found_any = True
            self.assertGreaterEqual(img.gps_lat_decimal, -90)
            self.assertLessEqual(img.gps_lat_decimal, 90)
            self.assertGreaterEqual(img.gps_lon_decimal, -180)
            self.assertLessEqual(img.gps_lon_decimal, 180)
        if not found_any:
            self.skipTest("no fixture in HEIC_DIR has GPS data (expected for CI's synthetic stub)")

    def test_non_one_orientation_is_rotated_and_ingests_successfully(self):
        # Regression test for a fixed bug: a non-1 HEIC orientation used
        # to raise OSError and be recorded as a FailedImageFile instead
        # of being ingested, on the (usually true, but not always, see
        # IMG_9370.HEIC in CLAUDE.md) assumption that libheif always
        # normalizes this itself. Now it's rotated via
        # apply_exif_orientation(), same as JPEG.
        path = os.path.join(self.HEIC_DIR, self._heic_files()[0])
        raw_width, raw_height = Image.open(path).size

        class FakeExif(dict):
            def get_ifd(self, tag):
                return {}

        fake_exif = FakeExif({274: 6})  # Orientation tag id = 274; 6 = 90deg CW
        with mock.patch("PIL.Image.Image.getexif", return_value=fake_exif):
            create_image_file(path)

        self.assertFalse(FailedImageFile.objects.filter(filename=path).exists())
        img = ImageFile.objects.get(filename=path)
        self.assertEqual(img.orientation, 6)
        # Orientation 6 is a 90-degree rotation, so width/height swap
        # relative to the raw (unrotated) decode.
        self.assertEqual(img.width, raw_height)
        self.assertEqual(img.height, raw_width)
        self.assertTrue(os.path.isfile(img.thumbnail_small.path))

    def test_all_orientation_codes_round_trip_correctly_on_real_photos(self):
        # Broader validation for the same fix as
        # test_non_one_orientation_is_rotated_and_ingests_successfully,
        # using real photo pixel data instead of a single synthetic
        # rotation and a single orientation code: for every real HEIC
        # fixture (all naturally orientation 1 -- already upright) and
        # every non-identity EXIF orientation code, simulate what that
        # code's "as captured" raw pixels would look like (the exact
        # mathematical inverse of apply_exif_orientation()'s own
        # transform for that code), then confirm apply_exif_orientation()
        # restores the original upright pixels exactly. This is the same
        # helper JPEG has always used -- this test exists to give HEIC's
        # newly-relaxed guard (see CLAUDE.md, IMG_9370.HEIC) the same
        # confidence for every orientation value, not just the one
        # production actually hit.
        def simulate_raw(image, orientation):
            if orientation == 2:
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            if orientation == 3:
                return image.transpose(Image.ROTATE_180)
            if orientation == 4:
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 5:
                return image.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
            if orientation == 6:
                return image.transpose(Image.ROTATE_90)
            if orientation == 7:
                return image.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 8:
                return image.transpose(Image.ROTATE_270)

        for filename in self._heic_files():
            path = os.path.join(self.HEIC_DIR, filename)
            original = Image.open(path).convert('RGB')
            orig_arr = np.array(original)
            for code in range(2, 9):
                raw = simulate_raw(original, code)
                restored = apply_exif_orientation(raw, code)
                restored_arr = np.array(restored)
                self.assertEqual(
                    restored_arr.shape, orig_arr.shape,
                    f"{filename} orientation={code}: shape mismatch after round-trip",
                )
                self.assertTrue(
                    np.array_equal(restored_arr, orig_arr),
                    f"{filename} orientation={code}: pixels did not round-trip exactly",
                )

    def test_multi_frame_guard_fails_loudly(self):
        path = os.path.join(self.HEIC_DIR, self._heic_files()[0])

        with mock.patch.object(
            pillow_heif.HeifImageFile, "n_frames",
            new_callable=mock.PropertyMock, return_value=3,
        ):
            create_image_file(path)

        self.assertFalse(ImageFile.objects.filter(filename=path).exists())
        failed = FailedImageFile.objects.get(filename=path)
        self.assertIn("frames", failed.error_message.lower())

    def test_rgba_heic_thumbnail_does_not_crash(self):
        # Regression test for a fixed bug: some real-world HEIC files
        # decode to RGBA mode (an alpha channel) even for a plain photo --
        # found via a genuine production file after HEIC support shipped,
        # unlike any of the all-RGB samples this was originally tested
        # against. _generate_thumbnail() tried to save that straight to
        # JPEG, which can't encode alpha, raising
        # "cannot write mode RGBA as JPEG".
        os.makedirs("/tmp/heic_rgba_test", exist_ok=True)
        path = "/tmp/heic_rgba_test/rgba.heic"
        arr = np.zeros((150, 200, 4), dtype=np.uint8)
        arr[:, :, 0] = 90
        arr[:, :, 1] = 160
        arr[:, :, 2] = 210
        arr[:, :, 3] = 255  # fully opaque, same as a real photo would be
        img = Image.fromarray(arr, mode="RGBA")
        heif_file = pillow_heif.from_pillow(img)
        heif_file.save(path, quality=80)
        self.addCleanup(shutil.rmtree, "/tmp/heic_rgba_test", ignore_errors=True)

        # Confirm the fixture is actually RGBA before testing our handling.
        self.assertEqual(Image.open(path).mode, "RGBA")

        create_image_file(path)

        self.assertFalse(FailedImageFile.objects.filter(filename=path).exists())
        obj = ImageFile.objects.get(filename=path)
        self.assertTrue(os.path.isfile(obj.thumbnail_small.path))

    def test_save_does_not_redecode_when_image_already_populated(self):
        # Real perf bug found 2026-09-18 while profiling why this test
        # class was so slow: ImageFile.save() used to call _init_image()
        # unconditionally, even when process_new_no_md5() (the normal
        # create_image_file() path) had already called it once on the
        # same instance -- silently doubling every HEIC decode (~0.89s
        # each, confirmed via pillow_heif.heif.BaseImage.load() tracing
        # against real fixtures; negligible for cheap formats like JPEG,
        # which is why it went unnoticed). self.image is only ever
        # assigned inside _init_image() itself, so guarding on
        # self.image already being set can never skip a genuinely-needed
        # decode -- verified separately (see CLAUDE.md) against all 8
        # real HEIC fixtures + 5 real JPEGs with byte-identical results.
        # This test targets the specific mechanism (a second
        # _init_image() call is skipped), not full-pipeline correctness.
        # Use whichever real fixture actually ingests successfully --
        # same "work with whatever's present" convention as the rest of
        # this class (a local dev fixture may include a multi-frame Live
        # Photo that's expected to fail loudly instead; CI's single
        # synthetic stub is expected to succeed).
        for filename in self._heic_files():
            path = os.path.join(self.HEIC_DIR, filename)
            call_count = {'n': 0}
            real_init_image = ImageFile._init_image

            def counting_init_image(self):
                call_count['n'] += 1
                return real_init_image(self)

            with mock.patch.object(ImageFile, '_init_image', counting_init_image):
                create_image_file(path)

            if ImageFile.objects.filter(filename=path).exists():
                self.assertEqual(
                    call_count['n'], 1,
                    "_init_image() should run exactly once per real ingestion, not twice",
                )
                return
        self.skipTest("no fixture in HEIC_DIR ingested successfully")


class NormalizeNullIslandGpsTests(TestCase):
    """Regression test for the normalize_null_island_gps management
    command, which cleans up ImageFile rows left with GPS (0, 0) by a
    fixed bug in _init_image(): a NaN GPS decimal conversion used to fall
    back to 0 instead of the -999 "no GPS" sentinel used everywhere else,
    so those rows look like real (if geographically absurd) GPS data
    rather than "no GPS" under a straight equality check."""

    def _make_bare_image_file(self, filename, lat, lon):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            gps_lat_decimal=lat, gps_lon_decimal=lon,
        )
        ImageFile.objects.bulk_create([obj])
        return ImageFile.objects.get(filename=filename)

    def test_normalizes_null_island_rows(self):
        null_island = self._make_bare_image_file("/tmp/null_island_test/a.jpg", 0, 0)
        real_gps = self._make_bare_image_file("/tmp/null_island_test/b.jpg", 47.62, -122.33)
        no_gps = self._make_bare_image_file("/tmp/null_island_test/c.jpg", -999, -999)

        call_command("normalize_null_island_gps", "--yes")

        null_island.refresh_from_db()
        real_gps.refresh_from_db()
        no_gps.refresh_from_db()
        self.assertEqual((null_island.gps_lat_decimal, null_island.gps_lon_decimal), (-999, -999))
        self.assertEqual((real_gps.gps_lat_decimal, real_gps.gps_lon_decimal), (47.62, -122.33))
        self.assertEqual((no_gps.gps_lat_decimal, no_gps.gps_lon_decimal), (-999, -999))

    def test_dry_run_writes_nothing(self):
        null_island = self._make_bare_image_file("/tmp/null_island_test/d.jpg", 0, 0)

        call_command("normalize_null_island_gps", "--dry-run")

        null_island.refresh_from_db()
        self.assertEqual((null_island.gps_lat_decimal, null_island.gps_lon_decimal), (0, 0))


class FindNearestMetroTests(unittest.TestCase):
    """find_nearest_metro() is pure/offline (no DB, no network) -- a plain
    unittest.TestCase rather than Django's TestCase is enough."""

    def test_suburb_maps_to_nearby_major_city(self):
        from filepopulator.geocode import find_nearest_metro
        name, distance_km = find_nearest_metro(47.7623, -122.2054)  # Bothell, WA
        self.assertIsNotNone(name)
        self.assertIn(name, ("Seattle", "Bellevue"))
        self.assertLess(distance_km, 25)

    def test_remote_location_has_no_match(self):
        from filepopulator.geocode import find_nearest_metro
        # Rural central Montana (Fergus County / Lewistown area) -- the
        # nearest sizeable city (Billings, Great Falls) is well over
        # SEARCH_RADIUS_BANDS_KM's widest band away.
        name, distance_km = find_nearest_metro(47.0623, -109.4280)
        self.assertIsNone(name)
        self.assertIsNone(distance_km)


class FindNearestNamedPlaceTests(unittest.TestCase):
    """find_nearest_named_place() is the last-resort, population-blind
    fallback used when Nominatim has nothing at all - pure/offline, same
    as FindNearestMetroTests above."""

    def test_returns_nearest_regardless_of_population(self):
        from filepopulator.geocode import find_nearest_named_place
        # Bothell itself may not be "major" enough for find_nearest_metro
        # to prefer it over Seattle/Bellevue (population-ranked within
        # the radius band) - this function doesn't rank by population at
        # all, so the literal nearest place wins even if it's small.
        name, country_code, distance_km = find_nearest_named_place(47.7623, -122.2054)
        self.assertIsNotNone(name)
        self.assertEqual(country_code, 'US')
        self.assertLess(distance_km, 25)

    def test_nothing_within_radius_returns_none(self):
        from filepopulator.geocode import find_nearest_named_place
        # Middle of the Pacific, nowhere near any place in the gazetteer
        # even at this function's generous default radius.
        name, country_code, distance_km = find_nearest_named_place(0.0, -150.0)
        self.assertIsNone(name)
        self.assertIsNone(country_code)
        self.assertIsNone(distance_km)


class NominatimRateLimiterReuseTests(unittest.TestCase):
    """Regression test for a real bug (found 2026-09-10): a fresh
    RateLimiter used to be constructed on every single call - min_delay_
    seconds is enforced via *instance* state, so a fresh instance had
    never made a call before and every coordinate in a backfill run got
    waved through with zero delay, never actually throttling anything."""

    def setUp(self):
        import filepopulator.geocode as geocode_module
        self._geocode_module = geocode_module
        self._orig_reverse = geocode_module._reverse_rate_limiter
        self._orig_forward = geocode_module._forward_rate_limiter
        geocode_module._reverse_rate_limiter = None
        geocode_module._forward_rate_limiter = None

    def tearDown(self):
        self._geocode_module._reverse_rate_limiter = self._orig_reverse
        self._geocode_module._forward_rate_limiter = self._orig_forward

    def test_reverse_limiter_is_a_reused_singleton(self):
        from filepopulator.geocode import _get_nominatim_geocode
        self.assertIs(_get_nominatim_geocode(), _get_nominatim_geocode())

    def test_forward_limiter_is_a_reused_singleton(self):
        from filepopulator.geocode import _get_nominatim_forward_geocode
        self.assertIs(_get_nominatim_forward_geocode(), _get_nominatim_forward_geocode())


class ReverseGeocodePreciseExtractionTests(unittest.TestCase):
    """reverse_geocode_precise()'s locality extraction, exercised against
    a mocked Nominatim client (patched at _get_nominatim_geocode, one
    level below reverse_geocode_precise itself) rather than mocking the
    function wholesale like BackfillGeocodingTests does - this is
    specifically testing the extraction logic those tests take as given."""

    def _location(self, address, display_name='Somewhere'):
        location = mock.Mock()
        location.raw = {'address': address, 'name': ''}
        location.address = display_name
        return location

    def test_settlement_tag_wins_over_weaker_tags(self):
        from filepopulator.geocode import reverse_geocode_precise
        address = {'city': 'Bothell', 'county': 'King County', 'country': 'United States', 'state': 'Washington'}
        with mock.patch('filepopulator.geocode._get_nominatim_geocode',
                         return_value=lambda *a, **k: self._location(address)):
            result = reverse_geocode_precise(47.7623, -122.2054)
        self.assertEqual(result['locality'], 'Bothell')

    def test_falls_back_to_county_when_no_settlement_tag(self):
        # Real production case (2026-09-10): a rural attraction/landmark
        # address has no city/town/village/hamlet/suburb tag at all, but
        # does have a county - a real, useful answer that was previously
        # discarded entirely in favor of leaving locality None.
        from filepopulator.geocode import reverse_geocode_precise
        address = {'county': 'Honolulu County', 'country': 'United States', 'state': 'Hawaii'}
        with mock.patch('filepopulator.geocode._get_nominatim_geocode',
                         return_value=lambda *a, **k: self._location(address)):
            result = reverse_geocode_precise(21.5258, -158.038)
        self.assertEqual(result['locality'], 'Honolulu County')

    def test_no_address_tags_at_all_leaves_locality_none(self):
        # Nothing left to fall back to at this level - run_geocoding_
        # backfill's own offline nearest-named-place fallback is what
        # handles this case, not reverse_geocode_precise itself.
        from filepopulator.geocode import reverse_geocode_precise
        with mock.patch('filepopulator.geocode._get_nominatim_geocode',
                         return_value=lambda *a, **k: self._location({})):
            result = reverse_geocode_precise(1.0, 1.0)
        self.assertIsNone(result['locality'])


class BackfillGeocodingTests(TestCase):
    """Exercises the backfill_geocoding management command against a
    mocked Nominatim lookup (no real network calls in tests) -- covers
    coordinate deduplication (multiple images at the same rounded spot
    share one GeocodeCache row), --dry-run, and that a failure on one
    coordinate is recorded (lookup_failed) rather than aborting the whole
    backfill for every other coordinate queued behind it."""

    def _make_bare_image_file(self, filename, lat, lon):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            gps_lat_decimal=lat, gps_lon_decimal=lon,
        )
        ImageFile.objects.bulk_create([obj])
        return ImageFile.objects.get(filename=filename)

    def test_shared_coordinate_reuses_one_cache_entry(self):
        from filepopulator.models import GeocodeCache
        img_a = self._make_bare_image_file("/tmp/geocode_test/a.jpg", 47.6062, -122.3321)
        img_b = self._make_bare_image_file("/tmp/geocode_test/b.jpg", 47.6062, -122.3321)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch(
            'filepopulator.geocode.reverse_geocode_precise',
            return_value=fake_result,
        ):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 1)
        img_a.refresh_from_db()
        img_b.refresh_from_db()
        self.assertIsNotNone(img_a.geocode)
        self.assertEqual(img_a.geocode_id, img_b.geocode_id)
        self.assertEqual(img_a.geocode.locality, 'Seattle')

    def test_dry_run_writes_nothing(self):
        from filepopulator.models import GeocodeCache
        self._make_bare_image_file("/tmp/geocode_test/c.jpg", 47.6062, -122.3321)

        call_command('backfill_geocoding', '--dry-run')

        self.assertEqual(GeocodeCache.objects.count(), 0)

    def test_failed_lookup_recorded_and_does_not_block_other_coordinates(self):
        from filepopulator.models import GeocodeCache
        img_fail = self._make_bare_image_file("/tmp/geocode_test/d.jpg", 47.6062, -122.3321)
        img_ok = self._make_bare_image_file("/tmp/geocode_test/e.jpg", 45.5152, -122.6784)

        fake_result = {
            'locality': 'Portland', 'county': 'Multnomah', 'state': 'Oregon',
            'country': 'United States', 'display_name': 'Portland, OR, USA',
            'raw_response': {'address': {}},
        }

        def flaky_geocode(lat, lon):
            if round(lat, 2) == 47.61:
                raise RuntimeError("simulated Nominatim failure")
            return fake_result

        with mock.patch(
            'filepopulator.geocode.reverse_geocode_precise',
            side_effect=flaky_geocode,
        ):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 2)
        img_fail.refresh_from_db()
        img_ok.refresh_from_db()
        self.assertTrue(img_fail.geocode.lookup_failed)
        self.assertFalse(img_ok.geocode.lookup_failed)
        self.assertEqual(img_ok.geocode.locality, 'Portland')

    def test_empty_locality_falls_back_to_nearest_named_place(self):
        # Real production bug (2026-09-10): Nominatim genuinely finding
        # nothing usable (not a failure) used to leave locality
        # permanently None. run_geocoding_backfill now falls back to the
        # offline nearest-named-place lookup and marks it approximate.
        from filepopulator.models import GeocodeCache
        img = self._make_bare_image_file("/tmp/geocode_test/f.jpg", 47.7623, -122.2054)  # near Bothell, WA

        fake_result = {
            'locality': None, 'county': None, 'state': None, 'country': None,
            'display_name': None, 'raw_response': None,
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result):
            call_command('backfill_geocoding')

        img.refresh_from_db()
        self.assertIsNotNone(img.geocode.locality)
        self.assertTrue(img.geocode.locality_is_approximate)
        self.assertFalse(img.geocode.lookup_failed)  # a genuine empty result, not a failure

    def test_concurrent_duplicate_coordinate_does_not_crash_the_batch(self):
        # Regression test for a real production bug (2026-08-27): the
        # recurring hourly geocode_new_images task and the one-time
        # backfill_geocoding command can run concurrently, and both
        # compute "what's missing" from the same snapshot -- if both
        # decide to geocode the same coordinate, whichever saves second
        # used to hit GeocodeCache's unique (lat, lon) constraint with no
        # handling at all, crashing the entire run for every other
        # coordinate queued behind it. Simulates the race by inserting a
        # competing row *during* our own reverse_geocode_precise() call,
        # the actual vulnerable window in production.
        from filepopulator.models import GeocodeCache
        img_race = self._make_bare_image_file("/tmp/geocode_test/race_a.jpg", 47.6062, -122.3321)
        img_ok = self._make_bare_image_file("/tmp/geocode_test/race_b.jpg", 45.5152, -122.6784)

        winner_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        portland_result = {
            'locality': 'Portland', 'county': 'Multnomah', 'state': 'Oregon',
            'country': 'United States', 'display_name': 'Portland, OR, USA',
            'raw_response': {'address': {}},
        }

        def racy_geocode(lat, lon):
            if round(lat, 2) == 47.61:
                GeocodeCache.objects.create(lat=lat, lon=lon, locality='Concurrent Winner')
                return winner_result
            return portland_result

        with mock.patch('filepopulator.geocode.reverse_geocode_precise', side_effect=racy_geocode):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 2)
        img_race.refresh_from_db()
        img_ok.refresh_from_db()
        self.assertEqual(img_race.geocode.locality, 'Concurrent Winner')
        self.assertEqual(img_ok.geocode.locality, 'Portland')

    def test_nearby_uncached_coordinate_reuses_existing_precise_geocode_without_a_network_call(self):
        # A real production scenario (2026-09-10): a photo library has a
        # lot of "wandered the same block across visits" GPS noise -- two
        # coordinates ~50m apart round to different ~11m GeocodeCache
        # cells (ROUND_DECIMALS=4) but are still clearly the same place.
        # An already-cached PRECISE result within GEOCODE_REUSE_RADIUS_KM
        # should be copied directly, with zero Nominatim calls.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        # ~50m north of the cached coordinate -- well within the 150m
        # reuse radius, but a different rounded (lat, lon) cache key.
        img = self._make_bare_image_file("/tmp/geocode_test/g.jpg", 47.60665, -122.3321)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise') as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_not_called()

        self.assertEqual(GeocodeCache.objects.count(), 2)
        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Seattle')
        self.assertEqual(img.geocode.state, 'Washington')
        self.assertFalse(img.geocode.locality_is_approximate)

    def test_far_uncached_coordinate_still_calls_nominatim(self):
        # The reuse radius is tight (150m) -- a coordinate genuinely in a
        # different area must not silently inherit a nearby cache entry's
        # locality.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/h.jpg", 45.5152, -122.6784)

        fake_result = {
            'locality': 'Portland', 'county': 'Multnomah', 'state': 'Oregon',
            'country': 'United States', 'display_name': 'Portland, OR, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Portland')

    def test_approximate_existing_entry_is_not_reused_as_a_source(self):
        # An offline nearest-named-place fallback result
        # (locality_is_approximate=True) is itself already a guess -- it
        # must not be propagated to a second, different coordinate as if
        # it were a genuine precise answer.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Some Distant Named Place',
            country='United States', locality_is_approximate=True,
        )
        img = self._make_bare_image_file("/tmp/geocode_test/i.jpg", 47.60665, -122.3321)

        fake_result = {
            'locality': 'Real Nearby Place', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Real Nearby Place, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Real Nearby Place')

    def test_reuse_chains_within_the_same_run(self):
        # A coordinate resolved earlier in THIS run (not predating it)
        # should also be usable as a reuse source for a later coordinate
        # in the same run, not just rows that already existed beforehand.
        from filepopulator.models import GeocodeCache
        img_seed = self._make_bare_image_file("/tmp/geocode_test/j.jpg", 47.6062, -122.3321)
        img_chain = self._make_bare_image_file("/tmp/geocode_test/k.jpg", 47.60665, -122.3321)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img_seed.refresh_from_db()
        img_chain.refresh_from_db()
        self.assertEqual(img_seed.geocode.locality, 'Seattle')
        self.assertEqual(img_chain.geocode.locality, 'Seattle')
        self.assertFalse(img_chain.geocode.locality_is_approximate)

    def test_previously_failed_coordinate_is_retried_and_recovers(self):
        # Real production scenario (2026-09-10): 80 coordinates failed
        # with a genuine 429 during the rate-limit storm this session
        # diagnosed and fixed -- before this fix, a failed row looked
        # permanently "already cached" and was never retried, even once
        # the underlying rate-limit issue was resolved.
        from filepopulator.models import GeocodeCache
        img = self._make_bare_image_file("/tmp/geocode_test/l.jpg", 47.6062, -122.3321)
        stale = GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, lookup_failed=True,
            lookup_error='Non-successful status code 429',
        )
        # A plain .update() rather than img.save() -- ImageFile.save()
        # unconditionally re-decodes the file for its pixel hash, and this
        # bare test fixture has no real file on disk.
        ImageFile.objects.filter(pk=img.pk).update(geocode=stale)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result):
            result = call_command('backfill_geocoding')

        # GeocodeCache row is updated in place, not duplicated.
        self.assertEqual(GeocodeCache.objects.count(), 1)
        stale.refresh_from_db()
        self.assertFalse(stale.lookup_failed)
        self.assertIsNone(stale.lookup_error)
        self.assertEqual(stale.locality, 'Seattle')
        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Seattle')

    def test_previously_failed_coordinate_that_fails_again_stays_flagged(self):
        from filepopulator.models import GeocodeCache
        self._make_bare_image_file("/tmp/geocode_test/m.jpg", 47.6062, -122.3321)
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, lookup_failed=True,
            lookup_error='Non-successful status code 429',
        )

        def still_failing(lat, lon):
            raise RuntimeError("still rate-limited")

        with mock.patch('filepopulator.geocode.reverse_geocode_precise', side_effect=still_failing):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 1)
        entry = GeocodeCache.objects.get(lat=47.6062, lon=-122.3321)
        self.assertTrue(entry.lookup_failed)
        self.assertEqual(entry.lookup_error, 'still rate-limited')

    def test_reuse_does_not_chain_through_a_previously_reused_result(self):
        # Real concern raised 2026-09-10: if a reused (not independently
        # Nominatim-confirmed) result could itself seed further reuse, a
        # sequence of points spaced just under GEOCODE_REUSE_RADIUS_KM
        # apart -- a realistic shape for a GPS trace along a street or
        # across a metro area -- could drift arbitrarily far from the one
        # real geocode the whole chain traces back to, each hop
        # individually "within radius" of the previous one. A must reuse
        # B, and B must reuse C, but C should never inherit A's answer if
        # C is actually outside the reuse radius of A itself.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        # ~100m north of the real anchor -- within the 150m reuse radius,
        # should reuse 'Seattle' with no Nominatim call.
        img_b = self._make_bare_image_file("/tmp/geocode_test/n.jpg", 47.6071, -122.3321)
        # ~100m further north again -- ~200m from the real anchor, so
        # OUTSIDE the reuse radius of it. If B (itself only reused, never
        # independently confirmed) were allowed to seed reuse, this would
        # wrongly inherit 'Seattle' from B. It must not.
        img_c = self._make_bare_image_file("/tmp/geocode_test/o.jpg", 47.6080, -122.3321)

        tacoma_result = {
            'locality': 'Tacoma', 'county': 'Pierce', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Tacoma, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=tacoma_result) as mock_reverse:
            call_command('backfill_geocoding')
            # Exactly one real call -- for C, which is genuinely out of
            # every real anchor's reuse radius. B needed none at all.
            mock_reverse.assert_called_once()

        img_b.refresh_from_db()
        img_c.refresh_from_db()
        self.assertEqual(img_b.geocode.locality, 'Seattle')
        self.assertEqual(img_c.geocode.locality, 'Tacoma')
        self.assertNotEqual(img_c.geocode.locality, img_b.geocode.locality)

    def test_image_at_already_cached_coordinate_is_relinked_with_no_nominatim_call(self):
        # Real production bug (2026-10-21): an ImageFile whose (rounded)
        # coordinate is ALREADY fully resolved in GeocodeCache (e.g. a
        # second photo from an already-geocoded place, ingested after the
        # first photo from that spot already cached+linked it) used to
        # never get its own geocode FK set at all -- the per-coordinate
        # loop only ever revisits a coordinate that's missing from the
        # cache or lookup_failed=True, so an already-resolved coordinate
        # was never looked at again by any future run. Now a standing,
        # unconditional relink pass fixes this on every call, at zero
        # Nominatim cost.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/relink1.jpg", 47.6062, -122.3321)
        self.assertIsNone(img.geocode)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise') as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_not_called()

        img.refresh_from_db()
        self.assertIsNotNone(img.geocode)
        self.assertEqual(img.geocode.locality, 'Seattle')

    def test_relink_dry_run_writes_nothing(self):
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/relink2.jpg", 47.6062, -122.3321)

        call_command('backfill_geocoding', '--dry-run')

        img.refresh_from_db()
        self.assertIsNone(img.geocode)

    def test_image_at_a_failed_coordinate_is_not_relinked_goes_through_retry_instead(self):
        # A lookup_failed=True cache row is a known-bad, not a resolved
        # answer -- it must go through the normal retry path (a real
        # reverse_geocode_precise call), not be silently "relinked" to a
        # failed, contentless entry.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, lookup_failed=True, lookup_error='simulated prior failure',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/relink3.jpg", 47.6062, -122.3321)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img.refresh_from_db()
        self.assertIsNotNone(img.geocode)
        self.assertFalse(img.geocode.lookup_failed)
        self.assertEqual(img.geocode.locality, 'Seattle')

    def test_high_failure_rate_this_run_is_recorded_as_an_application_error(self):
        # Run-level, not per-coordinate -- a real outage/rate-limit event
        # (confirmed real, 2026-09-10) fails most or all of a run's real
        # calls at once, unlike ordinary single-coordinate noise.
        from common.models import ApplicationError
        for i in range(6):
            self._make_bare_image_file(f"/tmp/geocode_test/fail{i}.jpg", 10.0 + i, 10.0 + i)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise',
                         side_effect=RuntimeError("simulated Nominatim outage")):
            call_command('backfill_geocoding')

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'filepopulator.geocode.run_geocoding_backfill')
        self.assertIn('6/6', error.message)

    def test_low_failure_rate_is_not_recorded(self):
        from common.models import ApplicationError
        for i in range(6):
            self._make_bare_image_file(f"/tmp/geocode_test/ok{i}.jpg", 20.0 + i, 20.0 + i)

        fake_result = {
            'locality': 'Somewhere', 'county': None, 'state': None,
            'country': 'United States', 'display_name': 'Somewhere, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result):
            call_command('backfill_geocoding')

        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_small_batch_below_min_attempts_is_not_recorded(self):
        # Even a 100% failure rate shouldn't flag a tiny batch -- too
        # noisy a signal below GEOCODE_FAILURE_RATE_MIN_ATTEMPTS.
        from common.models import ApplicationError
        self._make_bare_image_file("/tmp/geocode_test/tiny0.jpg", 30.0, 30.0)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise',
                         side_effect=RuntimeError("simulated failure")):
            call_command('backfill_geocoding')

        self.assertEqual(ApplicationError.objects.count(), 0)


class SimilarityTests(TestCase):
    """Exercises phash-based near-duplicate detection
    (filepopulator/similarity.py) against bare ImageFile rows with
    manually-set phash values, rather than real image files -- the
    comparison logic itself (Hamming distance + threshold + bookkeeping)
    doesn't depend on how the hash was produced."""

    def _make_bare_image_file(self, filename, phash, similarity_checked=False):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            phash=phash, similarity_checked=similarity_checked,
        )
        ImageFile.objects.bulk_create([obj])
        return ImageFile.objects.get(filename=filename)

    def test_finds_close_pair_and_skips_distant_pair(self):
        from filepopulator.models import SimilarImagePair
        from filepopulator.similarity import run_similarity_check

        base = 0x0F0F0F0F0F0F0F0F
        close = base ^ 0b111  # 3 bits different -- within default threshold
        far = base ^ ((1 << 40) - 1)  # 40 bits different -- well outside it

        img_base = self._make_bare_image_file("/tmp/similarity_test/base.jpg", base)
        img_close = self._make_bare_image_file("/tmp/similarity_test/close.jpg", close)
        img_far = self._make_bare_image_file("/tmp/similarity_test/far.jpg", far)

        result = run_similarity_check()

        self.assertEqual(result['checked'], 3)
        self.assertEqual(SimilarImagePair.objects.count(), 1)
        pair = SimilarImagePair.objects.get()
        self.assertEqual({pair.image_a_id, pair.image_b_id}, {img_base.id, img_close.id})
        self.assertEqual(pair.hamming_distance, 3)

        for img in (img_base, img_close, img_far):
            img.refresh_from_db()
            self.assertTrue(img.similarity_checked)

    def test_pair_stored_with_canonical_ordering_regardless_of_which_side_records_it(self):
        from filepopulator.models import SimilarImagePair

        img_a = self._make_bare_image_file("/tmp/similarity_test/order_a.jpg", 1)
        img_b = self._make_bare_image_file("/tmp/similarity_test/order_b.jpg", 2)
        lo, hi = sorted((img_a.id, img_b.id))

        SimilarImagePair.record(img_b.id, img_a.id, 1)
        SimilarImagePair.record(img_a.id, img_b.id, 1)

        self.assertEqual(SimilarImagePair.objects.count(), 1)
        pair = SimilarImagePair.objects.get()
        self.assertEqual(pair.image_a_id, lo)
        self.assertEqual(pair.image_b_id, hi)

    def test_dry_run_writes_nothing(self):
        from filepopulator.models import SimilarImagePair

        self._make_bare_image_file("/tmp/similarity_test/dry_a.jpg", 1)
        self._make_bare_image_file("/tmp/similarity_test/dry_b.jpg", 1)

        call_command("backfill_similarity", "--dry-run")

        self.assertEqual(SimilarImagePair.objects.count(), 0)
        self.assertFalse(ImageFile.objects.filter(similarity_checked=True).exists())

    def test_already_checked_images_are_not_recompared(self):
        from filepopulator.similarity import run_similarity_check

        img_a = self._make_bare_image_file("/tmp/similarity_test/prechecked.jpg", 1, similarity_checked=True)
        img_b = self._make_bare_image_file("/tmp/similarity_test/newcomer.jpg", 1)

        result = run_similarity_check()

        # Only the unchecked image counts as "checked" by this run, even
        # though it necessarily also compared against img_a to find it.
        self.assertEqual(result['checked'], 1)
        self.assertEqual(result['already_checked'], 1)

    def test_incremental_run_still_finds_match_against_previously_checked_image(self):
        from filepopulator.models import SimilarImagePair
        from filepopulator.similarity import run_similarity_check

        img_old = self._make_bare_image_file("/tmp/similarity_test/old.jpg", 0x1234, similarity_checked=True)
        img_new = self._make_bare_image_file("/tmp/similarity_test/new.jpg", 0x1234)

        run_similarity_check()

        self.assertEqual(SimilarImagePair.objects.count(), 1)
        pair = SimilarImagePair.objects.get()
        self.assertEqual({pair.image_a_id, pair.image_b_id}, {img_old.id, img_new.id})
        self.assertEqual(pair.hamming_distance, 0)


class CrossCollectionDedupClassifyTests(TestCase):
    """classify_pair(): pure function, exactly reproduces the 4-bucket
    logic validated against real production data 2026-10-21 (see
    filepopulator/cross_collection_dedup.py's own docstring and
    CLAUDE.md for the full investigation)."""

    def _bare_image(self, name, date_taken, make, model):
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/classify_test')
        path = f'/photos/classify_test/{name}.jpg'
        img = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100, dateTaken=date_taken,
            camera_make=make, camera_model=model,
        )
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def test_exact_metadata_match(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        when = timezone.now()
        a = self._bare_image('a', when, 'samsung', 'SM-A326W')
        b = self._bare_image('b', when, 'samsung', 'SM-A326W')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_EXACT_METADATA)

    def test_one_side_missing_camera_info(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._bare_image('a', timezone.now(), 'samsung', 'SM-A326W')
        b = self._bare_image('b', timezone.now() + timedelta(days=30), '', '')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_STRIPPED_EXIF)

    def test_same_camera_different_datetaken_is_burst_signature(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._bare_image('a', timezone.now(), 'NIKON CORPORATION', 'NIKON D3100')
        b = self._bare_image('b', timezone.now() + timedelta(seconds=4), 'NIKON CORPORATION', 'NIKON D3100')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_BURST_SIGNATURE)

    def test_differing_camera_make_is_a_conflict(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._bare_image('a', timezone.now(), 'samsung', 'SM-A326W')
        b = self._bare_image('b', timezone.now(), 'OLYMPUS IMAGING CORP.', 'C960Z')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_CAMERA_CONFLICT)


class CrossCollectionFaceMatchingTests(TestCase):
    """match_cross_collection_faces(): pure function, no DB. Mirrors the
    same optimal-assignment-then-threshold-filter pattern already tested
    for the duplicate-video case (merge_duplicate_videofiles.py)."""

    def test_rescaled_overlapping_boxes_match(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        primary_faces = [{'box': (100, 100, 200, 200)}]
        # Same relative position, but secondary is half the resolution --
        # needs scale_x=scale_y=2 to land back on the primary's box.
        secondary_faces = [{'box': (50, 50, 100, 100)}]
        matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x=2.0, scale_y=2.0)
        self.assertEqual(matches, [(0, 0)])

    def test_non_overlapping_boxes_do_not_match(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        primary_faces = [{'box': (0, 0, 50, 50)}]
        secondary_faces = [{'box': (500, 500, 550, 550)}]
        matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x=1.0, scale_y=1.0)
        self.assertEqual(matches, [])

    def test_picks_the_globally_optimal_pairing_not_a_greedy_one(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        # Two faces each side, at (roughly) swapped positions --
        # confirms this uses real assignment, not "first match wins".
        primary_faces = [{'box': (0, 0, 50, 50)}, {'box': (200, 200, 250, 250)}]
        secondary_faces = [{'box': (200, 200, 250, 250)}, {'box': (0, 0, 50, 50)}]
        matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x=1.0, scale_y=1.0)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_empty_either_side_returns_no_matches(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        self.assertEqual(match_cross_collection_faces([], [{'box': (0, 0, 10, 10)}], 1.0, 1.0), [])
        self.assertEqual(match_cross_collection_faces([{'box': (0, 0, 10, 10)}], [], 1.0, 1.0), [])


@override_settings(MEDIA_ROOT="/tmp/filepopulator_cross_collection_test_media")
class ResolveCrossCollectionPairTests(TestCase):
    """resolve_pair(): the real end-to-end orchestration -- real
    PhotoCollection/Directory/ImageFile/Face rows, real IOU matching and
    DB writes (or not, for dry_run/needs-review)."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.first_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.first_root, ignore_errors=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Second Collection', root_directory=self.second_root,
        )
        # Shared across both sides of a pair by default -- two separate
        # timezone.now() calls would never land on the same microsecond,
        # which would wrongly push an otherwise-identical-metadata pair
        # into the BURST_SIGNATURE bucket instead of EXACT_METADATA.
        self.shared_date_taken = timezone.now()

    def _make_image(self, name, collection, size_bytes, width=100, height=100,
                     date_taken=None, make='samsung', model='SM-A326W', is_processed=False):
        root = self.second_root if collection.pk == self.collection_b.pk else self.first_root
        directory, _ = Directory.objects.get_or_create(dir_path=root, defaults={'collection': collection})
        if directory.collection_id != collection.pk:
            directory.collection = collection
            directory.save()
        path = os.path.join(root, f'{name}.jpg')
        with open(path, 'wb') as f:
            f.write(os.urandom(size_bytes))
        img = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=width, height=height, dateTaken=date_taken or self.shared_date_taken,
            camera_make=make, camera_model=model, isProcessed=is_processed,
        )
        ImageFile.objects.bulk_create([img])
        img = ImageFile.objects.get(filename=path)
        # Matches what the real ingestion pipeline does at save time
        # (instance_clean_and_save()) -- bulk_create() bypasses it, so
        # this fixture replicates it explicitly rather than leaving
        # .collections permanently empty like a real ingested row never is.
        img.collections.add(collection)
        return img

    def _make_face(self, image, box, person=None, validated=False, kps=None):
        bl, bt, br, bb = box
        face = Face(
            source_image_file=image, declared_name=person or get_default_blank_person(),
            box_left=bl, box_top=bt, box_right=br, box_bottom=bb, validated=validated, kps=kps,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def _make_pair(self, image_a, image_b, distance=0):
        from filepopulator.models import SimilarImagePair
        SimilarImagePair.record(image_a.id, image_b.id, distance)
        return SimilarImagePair.objects.get(
            image_a_id=min(image_a.id, image_b.id), image_b_id=max(image_a.id, image_b.id),
        )

    def test_auto_merge_collapses_matched_faces_onto_the_larger_file(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        small = self._make_image('small', self.collection_a, size_bytes=100)
        big = self._make_image('big', self.collection_b, size_bytes=5000)
        small_face = self._make_face(small, (10, 10, 30, 30))
        big_face = self._make_face(big, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['status'], CrossCollectionImageDuplicate.STATUS_AUTO_MERGED)
        self.assertEqual(stats['primary_id'], big.id)
        self.assertEqual(stats['faces_matched'], 1)
        # Matched, both bare/unvalidated -- lowest id wins as the tiebreak.
        survivor_id = min(small_face.id, big_face.id)
        loser_id = max(small_face.id, big_face.id)
        self.assertTrue(Face.objects.filter(pk=survivor_id).exists())
        self.assertFalse(Face.objects.filter(pk=loser_id).exists())
        self.assertEqual(Face.objects.get(pk=survivor_id).source_image_file_id, big.id)

        small.refresh_from_db()
        self.assertTrue(small.isProcessed)
        self.assertIn(self.collection_b, big.collections.all())

        record = CrossCollectionImageDuplicate.objects.get(image_a_id=min(small.id, big.id), image_b_id=max(small.id, big.id))
        self.assertEqual(record.status, CrossCollectionImageDuplicate.STATUS_AUTO_MERGED)
        self.assertEqual(record.primary_image_id, big.id)

    def test_unmatched_secondary_face_is_transferred_not_deleted(self):
        from filepopulator.cross_collection_dedup import resolve_pair

        small = self._make_image('small2', self.collection_a, size_bytes=100)
        big = self._make_image('big2', self.collection_b, size_bytes=5000)
        # Only the secondary (small, since it's the smaller file) has a
        # face at all -- a detection only one side's extraction run found.
        secondary_only_face = self._make_face(small, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['faces_transferred'], 1)
        self.assertEqual(stats['faces_deleted'], 0)
        secondary_only_face.refresh_from_db()
        self.assertEqual(secondary_only_face.source_image_file_id, big.id)

    def test_validated_face_always_survives_a_match(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from face_manager.models import Person

        person = Person.objects.create(person_name='Real Person')
        small = self._make_image('small3', self.collection_a, size_bytes=100)
        big = self._make_image('big3', self.collection_b, size_bytes=5000)
        # The validated face is on the SMALLER (non-primary) file -- it
        # must still win and end up on the primary.
        validated_face = self._make_face(small, (10, 10, 30, 30), person=person, validated=True)
        unvalidated_face = self._make_face(big, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        self.assertTrue(Face.objects.filter(pk=validated_face.pk).exists())
        self.assertFalse(Face.objects.filter(pk=unvalidated_face.pk).exists())
        validated_face.refresh_from_db()
        self.assertEqual(validated_face.source_image_file_id, big.id)

    def test_needs_review_pair_touches_nothing(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._make_image(
            'burst_a', self.collection_a, size_bytes=100,
            date_taken=timezone.now(), make='NIKON CORPORATION', model='NIKON D3100',
        )
        b = self._make_image(
            'burst_b', self.collection_b, size_bytes=5000,
            date_taken=timezone.now() + timedelta(seconds=4), make='NIKON CORPORATION', model='NIKON D3100',
        )
        face_a = self._make_face(a, (10, 10, 30, 30))
        face_b = self._make_face(b, (10, 10, 30, 30))
        pair = self._make_pair(a, b)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['status'], CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW)
        self.assertEqual(stats['reason'], CrossCollectionImageDuplicate.REASON_BURST_SIGNATURE)
        # Both faces untouched, both images untouched.
        self.assertTrue(Face.objects.filter(pk=face_a.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face_b.pk).exists())
        a.refresh_from_db()
        b.refresh_from_db()
        self.assertFalse(a.isProcessed)
        self.assertFalse(b.isProcessed)
        self.assertNotIn(self.collection_b, a.collections.all())

        record = CrossCollectionImageDuplicate.objects.get(image_a_id=min(a.id, b.id), image_b_id=max(a.id, b.id))
        self.assertEqual(record.status, CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW)

    def test_dry_run_writes_nothing(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        small = self._make_image('small4', self.collection_a, size_bytes=100)
        big = self._make_image('big4', self.collection_b, size_bytes=5000)
        face_small = self._make_face(small, (10, 10, 30, 30))
        face_big = self._make_face(big, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=True)

        self.assertEqual(stats['status'], CrossCollectionImageDuplicate.STATUS_AUTO_MERGED)
        self.assertEqual(stats['faces_matched'], 1)
        # But nothing was actually written.
        self.assertTrue(Face.objects.filter(pk=face_small.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face_big.pk).exists())
        self.assertFalse(CrossCollectionImageDuplicate.objects.exists())

    def test_rescales_boxes_for_different_resolutions(self):
        from filepopulator.cross_collection_dedup import resolve_pair

        # big is 2x the resolution of small -- a face at the same
        # relative position should still match once rescaled.
        small = self._make_image('small5', self.collection_a, size_bytes=100, width=100, height=100)
        big = self._make_image('big5', self.collection_b, size_bytes=5000, width=200, height=200)
        self._make_face(small, (10, 10, 30, 30))  # 10% of width, relative
        self._make_face(big, (20, 20, 60, 60))  # same relative position at 2x scale
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)
        self.assertEqual(stats['faces_matched'], 1)

    def test_unmatched_transferred_face_box_is_rescaled_into_primarys_space(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # Real bug, found 2026-10-21 (see CLAUDE.md): a face transferred
        # from secondary onto primary without rescaling its box kept
        # secondary's own pixel coordinates -- silently wrong once the
        # two images have different resolutions, and a real risk of the
        # face being clamped/deleted the next time primary goes through
        # face_extract_encode.py's "existing face, no matching fresh
        # detection" reconciliation path.
        small = self._make_image('resc_small', self.collection_a, size_bytes=100, width=100, height=200)
        big = self._make_image('resc_big', self.collection_b, size_bytes=5000, width=400, height=800)  # 4x both axes
        # Only on the secondary (small) -- nothing on primary to match
        # against, so this is definitely the "unmatched, transfer" path.
        unmatched_face = self._make_face(small, (10, 20, 30, 50))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['primary_id'], big.id)
        self.assertEqual(stats['faces_transferred'], 1)
        unmatched_face.refresh_from_db()
        self.assertEqual(unmatched_face.source_image_file_id, big.id)
        # Rescaled by the real 4x width/4x height ratio between the pair.
        self.assertEqual(
            (unmatched_face.box_left, unmatched_face.box_top, unmatched_face.box_right, unmatched_face.box_bottom),
            (40, 80, 120, 200),
        )
        # And the rescaled box is a real, valid, non-degenerate box that
        # genuinely fits inside primary's actual dimensions -- the
        # concrete guarantee this fix is for.
        self.assertTrue(0 <= unmatched_face.box_left < unmatched_face.box_right <= big.width)
        self.assertTrue(0 <= unmatched_face.box_top < unmatched_face.box_bottom <= big.height)

    def test_matched_but_secondary_wins_face_box_is_also_rescaled(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # The OTHER path a transferred face can take -- matched to a
        # primary face, but the secondary's own copy is judged the
        # better survivor (e.g. it's validated) -- needs the exact same
        # rescaling as the unmatched-transfer path, not just a special
        # case of it.
        from face_manager.models import Person
        person = Person.objects.create(person_name='Rescale Survivor')

        small = self._make_image('resc_small2', self.collection_a, size_bytes=100, width=100, height=100)
        big = self._make_image('resc_big2', self.collection_b, size_bytes=5000, width=300, height=300)  # 3x both axes
        # Secondary's copy is validated -- it must win and be transferred
        # (rescaled) onto primary; primary's own (unvalidated) copy at
        # the matching relative position must be deleted.
        survivor = self._make_face(small, (10, 10, 30, 30), person=person, validated=True)
        loser = self._make_face(big, (30, 30, 90, 90))  # same relative position, 3x scale
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['faces_matched'], 1)
        self.assertFalse(Face.objects.filter(pk=loser.pk).exists())
        survivor.refresh_from_db()
        self.assertEqual(survivor.source_image_file_id, big.id)
        self.assertEqual(
            (survivor.box_left, survivor.box_top, survivor.box_right, survivor.box_bottom),
            (30, 30, 90, 90),
        )

    def test_transferred_face_kps_is_rescaled_not_discarded(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        small = self._make_image('resc_small3', self.collection_a, size_bytes=100, width=100, height=200)
        big = self._make_image('resc_big3', self.collection_b, size_bytes=5000, width=200, height=400)  # 2x both axes
        kps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # Box widened to actually bound these kps (x in [10,90], y in
        # [20,100]) within face_kps_within_box_margin's 25% margin --
        # the original (10,20,30,50) box was a geometrically-inconsistent
        # placeholder from before that constraint existed.
        face = self._make_face(small, (5, 15, 95, 105), kps=kps)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, big.id)
        # Even indices (x) scaled by scale_x=2.0, odd indices (y) by
        # scale_y=2.0 -- same 2x on both axes here, but computed via the
        # real per-axis logic, not just "double everything".
        self.assertEqual(face.kps, [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0])

    def test_null_kps_stays_null_after_transfer(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        small = self._make_image('resc_small4', self.collection_a, size_bytes=100, width=100, height=100)
        big = self._make_image('resc_big4', self.collection_b, size_bytes=5000, width=200, height=200)
        face = self._make_face(small, (10, 10, 30, 30), kps=None)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        face.refresh_from_db()
        self.assertIsNone(face.kps)

    def test_primary_marked_processed_when_secondary_already_was(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # Real bug, found 2026-10-21: if secondary already completed its
        # own real face-extraction pass, every real face for this
        # content is now on primary too (matched + transferred) -- but
        # primary's own isProcessed flag was never updated, so the
        # normal scheduled face_extraction task would later run a
        # wasteful (and, pre-rescaling-fix, risky) redundant detection
        # pass on it.
        small = self._make_image('proc_small', self.collection_a, size_bytes=100, is_processed=True)
        big = self._make_image('proc_big', self.collection_b, size_bytes=5000, is_processed=False)
        self._make_face(small, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        big.refresh_from_db()
        self.assertTrue(big.isProcessed)

    def test_primary_left_unprocessed_when_secondary_never_was(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # The opposite case: secondary was NEVER processed (no real
        # detection pass has ever found whatever faces might be in this
        # content) -- primary must NOT be force-marked done, since it
        # may still need its own genuine first detection pass.
        small = self._make_image('proc_small2', self.collection_a, size_bytes=100, is_processed=False)
        big = self._make_image('proc_big2', self.collection_b, size_bytes=5000, is_processed=False)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        big.refresh_from_db()
        self.assertFalse(big.isProcessed)

    def test_primary_already_processed_stays_processed_regardless(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        small = self._make_image('proc_small3', self.collection_a, size_bytes=100, is_processed=False)
        big = self._make_image('proc_big3', self.collection_b, size_bytes=5000, is_processed=True)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        big.refresh_from_db()
        self.assertTrue(big.isProcessed)


class FindUnresolvedCrossCollectionPairsTests(TestCase):
    """find_unresolved_cross_collection_pairs(): the candidate-selection
    query shared by the management command and the recurring task."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Find Unresolved Test', root_directory=self.second_root,
        )

    def _make_image(self, name, collection):
        root = self.second_root if collection.pk == self.collection_b.pk else '/photos/find_unresolved_test'
        directory, _ = Directory.objects.get_or_create(dir_path=root, defaults={'collection': collection})
        if directory.collection_id != collection.pk:
            directory.collection = collection
            directory.save()
        path = os.path.join(root, f'{name}.jpg')
        img = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def test_only_yields_cross_collection_exact_phash_pairs(self):
        from filepopulator.models import SimilarImagePair
        from filepopulator.cross_collection_dedup import find_unresolved_cross_collection_pairs

        cross = self._make_image('cross_a', self.collection_a)
        cross_other = self._make_image('cross_b', self.collection_b)
        SimilarImagePair.record(cross.id, cross_other.id, 0)

        same = self._make_image('same_a', self.collection_a)
        same_other = self._make_image('same_b', self.collection_a)
        SimilarImagePair.record(same.id, same_other.id, 0)

        far = self._make_image('far_a', self.collection_a)
        far_other = self._make_image('far_b', self.collection_b)
        SimilarImagePair.record(far.id, far_other.id, 4)  # not exact -- excluded

        results = list(find_unresolved_cross_collection_pairs())
        result_id_sets = [{p.image_a_id, p.image_b_id} for p in results]
        self.assertIn({cross.id, cross_other.id}, result_id_sets)
        self.assertNotIn({same.id, same_other.id}, result_id_sets)
        self.assertNotIn({far.id, far_other.id}, result_id_sets)

    def test_already_decided_pairs_are_excluded(self):
        from filepopulator.models import SimilarImagePair, CrossCollectionImageDuplicate
        from filepopulator.cross_collection_dedup import find_unresolved_cross_collection_pairs

        a = self._make_image('decided_a', self.collection_a)
        b = self._make_image('decided_b', self.collection_b)
        SimilarImagePair.record(a.id, b.id, 0)
        lo, hi = sorted((a.id, b.id))
        CrossCollectionImageDuplicate.objects.create(
            image_a_id=lo, image_b_id=hi,
            status=CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW,
            reason=CrossCollectionImageDuplicate.REASON_CAMERA_CONFLICT,
        )

        results = list(find_unresolved_cross_collection_pairs())
        result_id_sets = [{p.image_a_id, p.image_b_id} for p in results]
        self.assertNotIn({a.id, b.id}, result_id_sets)


class MergeCrossCollectionImageDuplicatesCommandTests(TestCase):
    """The management command itself -- thin wrapper over
    find_unresolved_cross_collection_pairs()/resolve_pair(), already
    covered individually above; this just confirms the command wires
    them together and reports sane totals."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Command Test Collection', root_directory=self.second_root,
        )

    def test_command_resolves_a_real_candidate(self):
        from filepopulator.models import SimilarImagePair, CrossCollectionImageDuplicate

        directory_a, _ = Directory.objects.get_or_create(
            dir_path='/photos/merge_command_test', defaults={'collection': self.collection_a},
        )
        img_a = ImageFile(
            filename='/photos/merge_command_test/a.jpg', directory=directory_a, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_a])
        img_a = ImageFile.objects.get(filename='/photos/merge_command_test/a.jpg')

        directory_b = Directory.objects.create(dir_path=self.second_root, collection=self.collection_b)
        path_b = os.path.join(self.second_root, 'b.jpg')
        with open(path_b, 'wb') as f:
            f.write(os.urandom(2000))
        img_b = ImageFile(
            filename=path_b, directory=directory_b, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_b])
        img_b = ImageFile.objects.get(filename=path_b)

        SimilarImagePair.record(img_a.id, img_b.id, 0)

        call_command('merge_cross_collection_image_duplicates')

        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 1)

    def test_dry_run_leaves_no_decision_records(self):
        from filepopulator.models import SimilarImagePair, CrossCollectionImageDuplicate

        directory_a, _ = Directory.objects.get_or_create(
            dir_path='/photos/merge_command_dry_test', defaults={'collection': self.collection_a},
        )
        img_a = ImageFile(
            filename='/photos/merge_command_dry_test/a.jpg', directory=directory_a, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_a])
        img_a = ImageFile.objects.get(filename='/photos/merge_command_dry_test/a.jpg')

        directory_b = Directory.objects.create(dir_path=self.second_root, collection=self.collection_b)
        img_b = ImageFile(
            filename=os.path.join(self.second_root, 'b.jpg'), directory=directory_b, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_b])
        img_b = ImageFile.objects.get(filename=os.path.join(self.second_root, 'b.jpg'))

        SimilarImagePair.record(img_a.id, img_b.id, 0)

        call_command('merge_cross_collection_image_duplicates', '--dry-run')

        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 0)


class BackfillMergedPrimaryIsProcessedTests(TestCase):
    """backfill_merged_primary_isprocessed: catch-up for merges that ran
    before resolve_pair() started marking primary isProcessed=True when
    secondary already held real faces at merge time (2026-10-21, see
    CLAUDE.md)."""

    def _make_record(self, primary_is_processed, faces_matched, faces_transferred, status='auto_merged'):
        from filepopulator.models import CrossCollectionImageDuplicate
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/backfill_isprocessed_test')
        import uuid
        suffix = uuid.uuid4().hex[:8]
        primary = ImageFile(
            filename=f'/photos/backfill_isprocessed_test/primary_{suffix}.jpg', directory=directory,
            thumbnail_small="", pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(), width=100, height=100,
            isProcessed=primary_is_processed,
        )
        secondary = ImageFile(
            filename=f'/photos/backfill_isprocessed_test/secondary_{suffix}.jpg', directory=directory,
            thumbnail_small="", pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(), width=100, height=100,
            isProcessed=True,  # always True post-merge, regardless of history -- see module docstring
        )
        ImageFile.objects.bulk_create([primary, secondary])
        primary = ImageFile.objects.get(filename=primary.filename)
        secondary = ImageFile.objects.get(filename=secondary.filename)
        lo, hi = sorted((primary.id, secondary.id))
        record = CrossCollectionImageDuplicate.objects.create(
            image_a_id=lo, image_b_id=hi, status=status,
            reason=CrossCollectionImageDuplicate.REASON_EXACT_METADATA,
            primary_image=primary, faces_matched=faces_matched, faces_transferred=faces_transferred,
        )
        return primary, record

    def test_marks_primary_processed_when_merge_record_shows_real_faces(self):
        primary, _ = self._make_record(primary_is_processed=False, faces_matched=1, faces_transferred=0)

        call_command('backfill_merged_primary_isprocessed')

        primary.refresh_from_db()
        self.assertTrue(primary.isProcessed)

    def test_leaves_primary_unprocessed_when_merge_record_shows_no_faces(self):
        # Proxy limitation, accepted: a secondary that was processed but
        # genuinely found zero faces looks identical to "never
        # processed" here -- primary is correctly left alone (a real,
        # if likely-also-zero-yielding, redetection pass later, not a
        # correctness risk).
        primary, _ = self._make_record(primary_is_processed=False, faces_matched=0, faces_transferred=0)

        call_command('backfill_merged_primary_isprocessed')

        primary.refresh_from_db()
        self.assertFalse(primary.isProcessed)

    def test_already_processed_primary_is_left_alone(self):
        primary, _ = self._make_record(primary_is_processed=True, faces_matched=1, faces_transferred=1)

        call_command('backfill_merged_primary_isprocessed')  # must not error re-marking an already-True row

        primary.refresh_from_db()
        self.assertTrue(primary.isProcessed)

    def test_needs_review_records_are_not_touched(self):
        from filepopulator.models import CrossCollectionImageDuplicate
        primary, _ = self._make_record(
            primary_is_processed=False, faces_matched=0, faces_transferred=0,
            status=CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW,
        )
        # needs_review records never have a primary_image in real usage,
        # but confirm the status filter itself excludes them regardless.
        CrossCollectionImageDuplicate.objects.filter(primary_image=primary).update(
            faces_matched=5, faces_transferred=5,
        )

        call_command('backfill_merged_primary_isprocessed')

        primary.refresh_from_db()
        self.assertFalse(primary.isProcessed)

    def test_dry_run_does_not_write(self):
        primary, _ = self._make_record(primary_is_processed=False, faces_matched=1, faces_transferred=0)

        call_command('backfill_merged_primary_isprocessed', '--dry-run')

        primary.refresh_from_db()
        self.assertFalse(primary.isProcessed)


class MergeCrossCollectionImageDuplicatesTaskTests(TestCase):
    """The recurring Celery task -- same underlying functions as the
    command, already covered individually above; this just confirms the
    task wires them together and respects its own advisory lock, same
    contract as cleanup_orphaned_media's own lock test."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Task Test Collection', root_directory=self.second_root,
        )

    def _make_pair(self, suffix):
        directory_a, _ = Directory.objects.get_or_create(
            dir_path='/photos/merge_task_test', defaults={'collection': self.collection_a},
        )
        img_a = ImageFile(
            filename=f'/photos/merge_task_test/a{suffix}.jpg', directory=directory_a, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_a])
        img_a = ImageFile.objects.get(filename=f'/photos/merge_task_test/a{suffix}.jpg')

        directory_b, _ = Directory.objects.get_or_create(
            dir_path=self.second_root, defaults={'collection': self.collection_b},
        )
        path_b = os.path.join(self.second_root, f'b{suffix}.jpg')
        img_b = ImageFile(
            filename=path_b, directory=directory_b, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_b])
        img_b = ImageFile.objects.get(filename=path_b)

        from filepopulator.models import SimilarImagePair
        SimilarImagePair.record(img_a.id, img_b.id, 0)
        return img_a, img_b

    def test_task_resolves_pending_pairs_and_respects_the_lock(self):
        from filepopulator.models import CrossCollectionImageDuplicate
        from filepopulator.tasks import merge_cross_collection_image_duplicates

        self._make_pair('1')
        merge_cross_collection_image_duplicates()
        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 1)

        # While another session holds the lock, the task must do nothing.
        self._make_pair('2')
        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        self.addCleanup(other.close)
        key = zlib.crc32(b'filepopulator.merge_cross_collection_image_duplicates')
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
            self.assertTrue(cur.fetchone()[0])

            merge_cross_collection_image_duplicates()
            self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 1)  # unchanged

            cur.execute("SELECT pg_advisory_unlock(%s)", [key])

        # Once free again, the second pair resolves too.
        merge_cross_collection_image_duplicates()
        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 2)


class BackfillOwnTracksGpsTaskTests(TestCase):
    """The recurring Celery task -- backfill_camera_gps() itself is
    already thoroughly tested directly (see filepopulator/tests.py's
    OwnTracks GPS backfill tests); this just confirms the task wires it
    up correctly (right default args, respects its own advisory lock,
    doesn't crash the whole task on OwnTracksConfigError)."""

    def test_calls_backfill_with_a_run_limit_and_no_camera_override(self):
        from filepopulator.tasks import backfill_owntracks_gps

        with mock.patch('filepopulator.owntracks_gps.backfill_camera_gps') as mock_backfill:
            mock_backfill.return_value = {'matched': 3, 'no_offset_tag': 1, 'no_nearby_point': 2}
            backfill_owntracks_gps()

        # No camera_models= override -- defaults to every camera currently
        # enabled in CameraGpsBackfillConfig, so a camera toggled on via
        # the frontend is picked up on the very next scheduled run.
        mock_backfill.assert_called_once()
        _, kwargs = mock_backfill.call_args
        self.assertNotIn('camera_models', kwargs)
        self.assertEqual(kwargs['limit'], 1000)

    def test_concurrent_run_is_skipped(self):
        from filepopulator.tasks import backfill_owntracks_gps

        with mock.patch('filepopulator.tasks.advisory_lock') as mock_lock, \
             mock.patch('filepopulator.owntracks_gps.backfill_camera_gps') as mock_backfill:
            mock_lock.return_value.__enter__.return_value = False
            backfill_owntracks_gps()

        mock_backfill.assert_not_called()

    def test_not_configured_is_a_no_op_not_a_crash(self):
        from filepopulator.owntracks_gps import OwnTracksConfigError
        from filepopulator.tasks import backfill_owntracks_gps

        with mock.patch('filepopulator.owntracks_gps.backfill_camera_gps') as mock_backfill:
            mock_backfill.side_effect = OwnTracksConfigError("not configured")
            backfill_owntracks_gps()  # must not raise

        mock_backfill.assert_called_once()


class PhashComputationTests(TestCase):
    """Confirms phash is actually computed and stored during real image
    ingestion (not just exercised against bare rows above), and that two
    genuinely different real photos land far apart."""

    @override_settings(MEDIA_ROOT='/tmp')
    def test_phash_populated_on_ingestion(self):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', '1.JPG')
        tmp_path = '/tmp/phash_ingestion_test/1.JPG'
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        shutil.copy(src, tmp_path)

        create_image_file(tmp_path)

        obj = ImageFile.objects.get(filename=tmp_path)
        self.assertIsNotNone(obj.phash)
        self.assertFalse(obj.similarity_checked)


@override_settings(MEDIA_ROOT='/tmp')
class PhashBackfillTests(TestCase):
    """Regression coverage for backfill_phash: images ingested before
    phash computation existed (or where it's manually cleared here to
    simulate that) get a phash computed directly from the file on disk,
    without going through the expensive full ImageFile.save() pipeline
    (which would also rehash pixel_hash/regenerate thumbnails)."""

    def _ingest_real_image(self, name):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', name)
        tmp_path = f'/tmp/phash_backfill_test/{name}'
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        shutil.copy(src, tmp_path)
        create_image_file(tmp_path)
        return ImageFile.objects.get(filename=tmp_path)

    def test_backfill_computes_phash_for_existing_image_missing_one(self):
        obj = self._ingest_real_image('1.JPG')
        # Simulate a pre-phash-era row: has real pixel_hash/thumbnails
        # already, just never got a phash.
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash')

        obj.refresh_from_db()
        self.assertIsNotNone(obj.phash)

    def test_dry_run_writes_nothing(self):
        obj = self._ingest_real_image('2.jpg')
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash', '--dry-run')

        obj.refresh_from_db()
        self.assertIsNone(obj.phash)

    def test_does_not_touch_pixel_hash_or_thumbnails(self):
        obj = self._ingest_real_image('1.JPG')
        original_pixel_hash = obj.pixel_hash
        original_thumb = obj.thumbnail_small.name
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash')

        obj.refresh_from_db()
        self.assertEqual(obj.pixel_hash, original_pixel_hash)
        self.assertEqual(obj.thumbnail_small.name, original_thumb)

    def test_multiprocess_backfill_matches_single_process_result(self):
        obj_a = self._ingest_real_image('1.JPG')
        obj_b = self._ingest_real_image('2.jpg')
        ImageFile.objects.filter(id__in=[obj_a.id, obj_b.id]).update(phash=None)

        call_command('backfill_phash', '--processes', '2')

        obj_a.refresh_from_db()
        obj_b.refresh_from_db()
        self.assertIsNotNone(obj_a.phash)
        self.assertIsNotNone(obj_b.phash)


class GuessDateFromFilenameTests(unittest.TestCase):
    """Pure-function tests for guess_date_from_filename() -- no DB needed.
    Patterns and the "no match" cases are all drawn from a real 2026-09
    survey of this library's own images with no valid EXIF date."""

    def test_android_style_compact(self):
        got = guess_date_from_filename('/photos/x/IMG_20240719_211850.jpg')
        self.assertEqual(got.isoformat(), '2024-07-19T21:18:50+00:00')

    def test_iso_space_dot_style(self):
        got = guess_date_from_filename('/photos/x/2019-04-08 21.08.22.jpg')
        self.assertEqual(got.isoformat(), '2019-04-08T21:08:22+00:00')

    def test_iso_dash_time_style(self):
        got = guess_date_from_filename('/photos/x/2016-02-06_16-18-35_000.jpeg')
        self.assertEqual(got.isoformat(), '2016-02-06T16:18:35+00:00')

    def test_epoch_ms_style(self):
        got = guess_date_from_filename('/photos/x/1697447941437.jpg')
        self.assertEqual(got.isoformat(), '2023-10-16T09:19:01.437000+00:00')

    def test_mixed_format_double_date_picks_earlier_capture_not_later_export(self):
        """Regression case for the real bug found 2026-09-01: a filename
        embedding two timestamps in DIFFERENT formats (hyphenated capture
        date, compact export date) used to silently return whichever
        pattern's priority happened to match first -- which could be the
        LATER (export) date instead of the real, earlier capture date."""
        name = 'Resized_2022-05-29_09-45-29_923_20220531_164559.jpg'
        got = guess_date_from_filename(name)
        self.assertEqual(got.isoformat(), '2022-05-29T09:45:29+00:00')

    def test_same_format_double_date_picks_earlier_one(self):
        name = 'Resized_20221217_162217_20221220_013629.jpg'
        got = guess_date_from_filename(name)
        self.assertEqual(got.isoformat(), '2022-12-17T16:22:17+00:00')

    def test_no_date_in_whatsapp_received_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/received_3804555136348326.jpeg'))

    def test_no_date_in_uuid_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename(
            '/photos/x/744A2F1E-1265-4FA2-93CE-B9EBA91E2188.jpeg'))

    def test_no_date_in_plain_sequential_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_6403.JPG'))

    def test_no_date_in_scan_album_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/album_1 0161.jpg'))

    def test_invalid_calendar_date_is_rejected_not_crashed(self):
        # Digits that happen to line up like the pattern but aren't a
        # real date (month 99) -- must not raise, just skip this match.
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_20249919_211850.jpg'))

    def test_date_before_1990_is_rejected(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_19850101_120000.jpg'))

    def test_date_in_the_future_is_rejected(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_29990101_120000.jpg'))

    def test_epoch_ms_style_2014_to_2017_range_is_recognized(self):
        # Real bug, found 2026-10-28: the pattern was originally bounded
        # to a "1[5-8]" leading digit (~2017-2035), silently missing the
        # entire 2001-2017 range of real 13-digit ms-epoch timestamps --
        # confirmed against 1,815 real "Mom Phone" photos with a leading
        # "14" (2014-2017) that used to return None here despite having a
        # perfectly real, recoverable date embedded in the filename.
        got = guess_date_from_filename('/photos/x/1482109271865.jpg')
        self.assertEqual(got.isoformat(), '2016-12-19T01:01:11.865000+00:00')


class GetDateTakenFilenameFallbackTests(unittest.TestCase):
    """Integration-style coverage for ImageFile._get_date_taken()'s use of
    guess_date_from_filename() as a fallback -- constructs a bare,
    unsaved ImageFile with exifDict set directly, so no real image file
    or DB row is needed to exercise the branching logic itself."""

    def test_no_exif_falls_back_to_filename_guess(self):
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = None
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_exif_dict_present_but_no_valid_date_key_falls_back_to_filename(self):
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = {'Make': 'Canon'}  # no DateTime*/DateTimeOriginal/DateTimeDigitized
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_valid_exif_date_wins_even_if_filename_also_has_a_date(self):
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = {'DateTimeOriginal': '2018:06:15 08:00:00'}
        obj._get_date_taken()
        self.assertTrue(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2018-06-15T08:00:00+00:00')

    def test_no_exif_and_unparseable_filename_falls_back_to_now_unchanged(self):
        obj = ImageFile(filename='/photos/x/received_3804555136348326.jpeg')
        obj.exifDict = None
        before = timezone.now()
        obj._get_date_taken()
        after = timezone.now()
        self.assertFalse(obj.dateTakenValid)
        self.assertTrue(before <= obj.dateTaken <= after)

    def test_device_clock_reset_exif_date_is_rejected_falls_back_to_filename(self):
        # Real bug, found 2026-10-28: a genuine EXIF DateTimeOriginal
        # field containing "1900:01:01 01:01:01" -- a well-known device-
        # clock-reset default (confirmed against 2,376 real images, 1,815
        # from one phone) -- used to be accepted as a real, valid date
        # just because it parsed successfully, even though it's obviously
        # not a real digital capture date. Must be rejected the same as
        # the other garbage-string EXIF cases, falling through to the
        # filename guess instead.
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = {'DateTimeOriginal': '1900:01:01 01:01:01'}
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_filename_guess_also_sets_dateTakenUTC_not_just_dateTaken(self):
        # Real bug, found 2026-10-28: the filename-fallback branch only
        # ever set self.dateTaken (a DateTimeField) -- self.dateTakenUTC
        # (the plain float field Face.dateTakenUTC and other downstream
        # numeric-date logic actually reads) was silently left at its
        # stale/default value even when a real date was recovered.
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = None
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')
        self.assertEqual(obj.dateTakenUTC, obj.dateTaken.timestamp())


@override_settings(MEDIA_ROOT="/tmp/filepopulator_test_media")
class BackfillDatesFromFilenameTests(TestCase):
    """backfill_dates_from_filename replaces the now()-placeholder
    dateTaken with a filename-derived guess for existing rows that have
    no valid EXIF date -- mirrors PhashBackfillTests' pattern of ingesting
    a real fixture image, then forcing it into the "needs backfill" state
    via .update() (bypassing ImageFile.save(), which would otherwise redo
    its own EXIF-based date detection and undo the setup)."""

    def _ingest_real_image_at(self, dest_filename, fixture_name='1.JPG'):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', fixture_name)
        os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
        shutil.copy(src, dest_filename)
        create_image_file(dest_filename)
        return ImageFile.objects.get(filename=dest_filename)

    def test_backfill_updates_dateTaken_for_row_with_filename_date(self):
        dest = '/tmp/date_backfill_test/IMG_20200101_120000.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False)

        call_command('backfill_dates_from_filename')

        obj.refresh_from_db()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_row_with_no_filename_date_is_left_alone(self):
        dest = '/tmp/date_backfill_test/received_3804555136348326.jpeg'
        obj = self._ingest_real_image_at(dest, fixture_name='2.jpg')
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_dates_from_filename')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_row_with_valid_exif_is_never_touched(self):
        dest = '/tmp/date_backfill_test/IMG_20200101_120000_realexif.jpg'
        obj = self._ingest_real_image_at(dest)
        # Whatever real EXIF this fixture has (or its own now()-fallback)
        # -- the point is dateTakenValid=True rows must be skipped
        # entirely regardless of what their filename says.
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=True)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_dates_from_filename')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_dry_run_writes_nothing(self):
        dest = '/tmp/date_backfill_test/IMG_20200101_120000_dryrun.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_dates_from_filename', '--dry-run')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)


class BackfillImplausibleExifDatesTests(TestCase):
    """backfill_implausible_exif_dates -- corrects rows already written
    with a wrongly-trusted implausible (pre-1990) EXIF date, the real
    bug fixed 2026-10-28 (see _get_date_taken()'s own history). Mirrors
    BackfillDatesFromFilenameTests' pattern: ingest a real fixture, then
    force it into the "already has the bad value" state via .update()."""

    def _ingest_real_image_at(self, dest_filename, fixture_name='1.JPG'):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', fixture_name)
        os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
        shutil.copy(src, dest_filename)
        create_image_file(dest_filename)
        return ImageFile.objects.get(filename=dest_filename)

    def test_recovers_real_date_from_filename_when_available(self):
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(
            dateTakenValid=True, dateTakenUTC=-2208985139.0,
        )

        call_command('backfill_implausible_exif_dates')

        obj.refresh_from_db()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')
        self.assertEqual(obj.dateTakenUTC, obj.dateTaken.timestamp())

    def test_resets_to_unknown_placeholder_when_no_filename_date_available(self):
        dest = '/tmp/implausible_date_backfill_test/received_3804555136348326.jpeg'
        obj = self._ingest_real_image_at(dest, fixture_name='2.jpg')
        ImageFile.objects.filter(pk=obj.pk).update(
            dateTakenValid=True, dateTakenUTC=-2208985139.0,
        )

        before = timezone.now()
        call_command('backfill_implausible_exif_dates')
        after = timezone.now()

        obj.refresh_from_db()
        self.assertFalse(obj.dateTakenValid)
        self.assertTrue(before <= obj.dateTaken <= after)
        self.assertEqual(obj.dateTakenUTC, obj.dateTaken.timestamp())

    def test_row_with_a_plausible_date_is_never_touched(self):
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000_plausible.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=True, dateTakenUTC=1600000000.0)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_implausible_exif_dates')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_row_already_marked_invalid_is_never_touched(self):
        # Out of scope for this command -- backfill_dates_from_filename
        # already covers the dateTakenValid=False population.
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000_alreadyfalse.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False, dateTakenUTC=-2208985139.0)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_implausible_exif_dates')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_dry_run_writes_nothing(self):
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000_dryrun.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(
            dateTakenValid=True, dateTakenUTC=-2208985139.0,
        )

        call_command('backfill_implausible_exif_dates', '--dry-run')

        obj.refresh_from_db()
        self.assertTrue(obj.dateTakenValid)
        self.assertEqual(obj.dateTakenUTC, -2208985139.0)


@override_settings(MEDIA_ROOT="/tmp/filepopulator_test_media")
class VideoIngestionTests(TestCase):
    """VIDEO_DIR mirrors real fixture data locally: 8 real video samples
    (one per real format found in the actual video library -- mp4, mov,
    mpg, avi, m2ts, mts, wmv, 3gp) at
    /mnt/fast_storage/appdata/django_picasa/test_suite/sample_photos/
    video_samples/, mounted read-only under /photos/video_samples. CI
    only has the single synthetic no-metadata stub from
    ci_fixtures/video_stub/ (see generate_fixtures.py) -- tests here work
    with "whatever's present" rather than assuming a specific count or
    specific metadata, so they hold in both environments.

    Metadata handling verified empirically against all 8 real samples
    (see CLAUDE.md): ffprobe alone misses creation date/camera info that
    exiftool recovers for some real camcorder-sourced files; a global
    exiftool -n flag was tried and rejected (it also breaks Make/Model
    decoding for at least one real file); dateutil.parser was tried and
    rejected as a date-parsing fallback (silently defaults the date
    portion to today() for exiftool's colon-separated date format,
    confirmed against a real file, no exception raised)."""

    VIDEO_DIR = '/photos/video_samples'

    def tearDown(self):
        VideoFile.objects.all().delete()
        FailedVideoFile.objects.all().delete()

    def _video_files(self):
        return sorted(os.listdir(self.VIDEO_DIR))

    def _duration_seconds(self, path):
        from .video_scripts import _run_ffprobe
        probe = _run_ffprobe(path)
        return float(probe.get('format', {}).get('duration', 0) or 0)

    def _long_enough_files(self):
        # Real fixtures include at least one real clip under
        # settings.MIN_VIDEO_DURATION_SECONDS (a real ~2s digitized home
        # movie snippet), and the CI synthetic stub is 1s -- tests here
        # partition by actual measured duration rather than assuming
        # every fixture clears the bar.
        from django.conf import settings
        return [
            f for f in self._video_files()
            if self._duration_seconds(os.path.join(self.VIDEO_DIR, f)) >= settings.MIN_VIDEO_DURATION_SECONDS
        ]

    def test_ingests_fixtures_at_or_above_minimum_duration(self):
        video_files = self._video_files()
        self.assertGreaterEqual(len(video_files), 1)
        long_enough = set(self._long_enough_files())
        for filename in video_files:
            path = os.path.join(self.VIDEO_DIR, filename)
            create_video_file(path)
            if filename in long_enough:
                v = VideoFile.objects.filter(filename=path).first()
                self.assertIsNotNone(v, f"{filename} was not ingested")
                self.assertGreater(v.width, 0)
                self.assertGreater(v.height, 0)
                self.assertEqual(v.file_size_bytes, os.path.getsize(path))
                self.assertFalse(FailedVideoFile.objects.filter(filename=path).exists())
            else:
                self.assertFalse(VideoFile.objects.filter(filename=path).exists())
                failed = FailedVideoFile.objects.filter(filename=path).first()
                self.assertIsNotNone(failed, f"{filename} should have been excluded as too short")
                self.assertIn("Excluded", failed.error_message)

    def test_videos_shorter_than_minimum_are_excluded_not_ingested(self):
        # Explicit regression test for the exclusion itself, not just
        # incidental coverage from the fixture loop above.
        long_enough = set(self._long_enough_files())
        short_files = [f for f in self._video_files() if f not in long_enough]
        if not short_files:
            self.skipTest("no fixture in VIDEO_DIR is shorter than MIN_VIDEO_DURATION_SECONDS")
        path = os.path.join(self.VIDEO_DIR, short_files[0])
        create_video_file(path)
        self.assertFalse(VideoFile.objects.filter(filename=path).exists())
        self.assertTrue(FailedVideoFile.objects.filter(filename=path, error_message__icontains="Excluded").exists())

    def test_add_videos_from_root_dir_discovers_video_files(self):
        add_videos_from_root_dir([self.VIDEO_DIR])
        self.assertEqual(VideoFile.objects.count(), len(self._long_enough_files()))

    def test_rerunning_finds_nothing_new(self):
        # Regression-shaped test: a second run over the same root
        # shouldn't create duplicate rows or re-process already-known
        # files.
        add_videos_from_root_dir([self.VIDEO_DIR])
        first_count = VideoFile.objects.count()
        add_videos_from_root_dir([self.VIDEO_DIR])
        self.assertEqual(VideoFile.objects.count(), first_count)

    def test_tracked_video_root_going_empty_is_recorded_as_an_application_error(self):
        # Mirrors add_from_root_dir()'s own check (image side) -- a real,
        # exceptional signal (dropped mount, bad permissions, a renamed
        # root), distinct from a brand-new, genuinely-empty root (next
        # test), which must NOT be flagged. VIDEO_DIR itself is a
        # read-only real-fixture mount, so this copies one fixture into a
        # controlled tmp root instead of touching VIDEO_DIR directly.
        from common.models import ApplicationError
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no real fixture long enough to ingest")
        tmp_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_root, ignore_errors=True)
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[0]), os.path.join(tmp_root, long_enough[0]))

        add_videos_from_root_dir([tmp_root])
        self.assertGreater(VideoFile.objects.count(), 0)

        os.remove(os.path.join(tmp_root, long_enough[0]))
        add_videos_from_root_dir([tmp_root])

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'filepopulator.video_scripts.add_videos_from_root_dir')
        self.assertIn(tmp_root, error.message)

    def test_brand_new_empty_video_root_is_not_recorded(self):
        from common.models import ApplicationError
        empty_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty_root, ignore_errors=True)
        add_videos_from_root_dir([empty_root])
        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_multiple_roots_are_all_walked(self):
        # add_videos_from_root_dir() takes a LIST of roots (built by the
        # caller from every PhotoCollection's root_directory plus its
        # extra_video_roots -- see load_videos_into_db()) -- confirm
        # passing two roots actually walks both, not just the first.
        #
        # The second root's copy is BYTE-IDENTICAL to the first (same
        # fixture, copied verbatim) -- since content-hash-based duplicate
        # detection now runs at ingestion, that means it's correctly
        # recognized as a duplicate rather than given its own VideoFile
        # row (see VideoContentHashDuplicateTests). The real thing this
        # test needs to confirm -- that the second root was genuinely
        # walked and its file genuinely discovered/processed, not just
        # silently skipped -- still holds: a DuplicateVideoFile record
        # is exactly that evidence.
        other_dir = '/tmp/video_other_root'
        os.makedirs(other_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, other_dir, ignore_errors=True)
        long_enough = self._long_enough_files()
        self.assertGreaterEqual(len(long_enough), 1)
        first_file = os.path.join(self.VIDEO_DIR, long_enough[0])
        other_copy = os.path.join(other_dir, long_enough[0])
        shutil.copy(first_file, other_copy)

        add_videos_from_root_dir([self.VIDEO_DIR, other_dir])

        self.assertFalse(VideoFile.objects.filter(filename=other_copy).exists())
        self.assertTrue(DuplicateVideoFile.objects.filter(filename=other_copy).exists())
        self.assertEqual(VideoFile.objects.count(), len(long_enough))

    def test_nonexistent_file_is_recorded_as_failed(self):
        create_video_file('/photos/video_samples/does_not_exist.mp4')
        self.assertFalse(VideoFile.objects.filter(filename='/photos/video_samples/does_not_exist.mp4').exists())
        self.assertTrue(FailedVideoFile.objects.filter(filename='/photos/video_samples/does_not_exist.mp4').exists())

    def test_gps_and_camera_info_recovered_when_present(self):
        # Not every fixture (or CI's synthetic stub) has GPS/camera
        # metadata -- only assert against ones that actually do.
        found_gps = False
        found_camera = False
        for filename in self._long_enough_files():
            path = os.path.join(self.VIDEO_DIR, filename)
            create_video_file(path)
            v = VideoFile.objects.get(filename=path)
            if v.gps_lat_decimal != -999 or v.gps_lon_decimal != -999:
                found_gps = True
                self.assertGreaterEqual(v.gps_lat_decimal, -90)
                self.assertLessEqual(v.gps_lat_decimal, 90)
                self.assertGreaterEqual(v.gps_lon_decimal, -180)
                self.assertLessEqual(v.gps_lon_decimal, 180)
            if v.camera_make:
                found_camera = True
        if not found_gps and not found_camera:
            self.skipTest("no fixture in VIDEO_DIR has GPS or camera metadata (expected for CI's synthetic stub)")

    def test_backfill_command_populates_missing_file_size(self):
        from django.core.management import call_command

        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(path)
        VideoFile.objects.filter(filename=path).update(file_size_bytes=None)

        call_command('backfill_video_file_size')

        v = VideoFile.objects.get(filename=path)
        self.assertEqual(v.file_size_bytes, os.path.getsize(path))

    def test_backfill_command_dry_run_makes_no_changes(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        from django.core.management import call_command

        path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(path)
        VideoFile.objects.filter(filename=path).update(file_size_bytes=None)

        call_command('backfill_video_file_size', '--dry-run')

        v = VideoFile.objects.get(filename=path)
        self.assertIsNone(v.file_size_bytes)

    def test_content_hash_is_populated_on_ingest(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(path)
        v = VideoFile.objects.get(filename=path)
        self.assertIsNotNone(v.content_hash)
        self.assertEqual(v.content_hash, compute_video_content_hash(path))

    def test_duplicate_content_at_new_path_is_recorded_not_reingested(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        original_path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(original_path)
        original = VideoFile.objects.get(filename=original_path)

        dup_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, dup_dir, ignore_errors=True)
        dup_path = os.path.join(dup_dir, 'copy_' + long_enough[0])
        shutil.copy(original_path, dup_path)

        create_video_file(dup_path)

        self.assertFalse(VideoFile.objects.filter(filename=dup_path).exists())
        dup_record = DuplicateVideoFile.objects.filter(filename=dup_path).first()
        self.assertIsNotNone(dup_record)
        self.assertEqual(dup_record.original_id, original.pk)
        # No second row was created for the duplicate content.
        self.assertEqual(VideoFile.objects.filter(content_hash=original.content_hash).count(), 1)

    def test_moved_file_same_content_reuses_existing_row(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        move_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, move_dir, ignore_errors=True)
        original_copy = os.path.join(move_dir, 'orig_' + long_enough[0])
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[0]), original_copy)

        create_video_file(original_copy)
        original_pk = VideoFile.objects.get(filename=original_copy).pk

        new_path = os.path.join(move_dir, 'moved_' + long_enough[0])
        os.rename(original_copy, new_path)

        create_video_file(new_path)

        self.assertFalse(VideoFile.objects.filter(filename=original_copy).exists())
        moved = VideoFile.objects.filter(filename=new_path).first()
        self.assertIsNotNone(moved)
        self.assertEqual(moved.pk, original_pk)
        self.assertEqual(VideoFile.objects.count(), 1)

    def test_stale_duplicate_record_is_wiped_when_content_at_path_changes(self):
        # Real gap found 2026-10-14 (video move-robustness review): unlike
        # create_image_file(), create_video_file() had no guard for "a
        # stale DuplicateVideoFile record already sits at this exact
        # path" -- if that path's real content later genuinely changes,
        # the old record would just sit there stale, pointing at
        # whatever used to be the "original". Not a data-loss risk (no
        # VideoFile row ever coexists with a live DuplicateVideoFile at
        # the same path), just confusing bookkeeping left unresolved.
        long_enough = self._long_enough_files()
        if len(long_enough) < 2:
            self.skipTest("need at least 2 distinct real fixtures in VIDEO_DIR")

        work_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work_dir, ignore_errors=True)
        original_path = os.path.join(work_dir, 'orig_' + long_enough[0])
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[0]), original_path)
        create_video_file(original_path)
        original = VideoFile.objects.get(filename=original_path)

        # Record it as a duplicate of `original`, matching the real
        # shape of the stale-bookkeeping scenario.
        changing_path = os.path.join(work_dir, 'changing_' + long_enough[0])
        shutil.copy(original_path, changing_path)
        create_video_file(changing_path)
        self.assertTrue(DuplicateVideoFile.objects.filter(filename=changing_path).exists())

        # The real content at that path now genuinely changes.
        os.remove(changing_path)
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[1]), changing_path)

        create_video_file(changing_path)

        # The stale duplicate record is gone (wiped, not left pointing at
        # the wrong "original" forever) -- and a later rescan of this
        # same path can genuinely ingest it fresh.
        self.assertFalse(DuplicateVideoFile.objects.filter(filename=changing_path).exists())
        self.assertFalse(VideoFile.objects.filter(filename=changing_path).exists())
        create_video_file(changing_path)
        self.assertTrue(VideoFile.objects.filter(filename=changing_path).exists())


class VideoDirectoryCollectionTests(TestCase):
    """create_video_file() resolves+sets Directory.collection on a newly
    created video directory (2026-10-21 fix) -- previously a plain
    get_or_create with no collection resolution at all, leaving every
    video Directory permanently NULL. Mirrors PhotoCollectionTests'
    image-side setup/assertions, since VideoFile.directory is the SAME
    Directory model ImageFile uses -- no new field was needed, just the
    missing resolve_for_path() call."""

    def setUp(self):
        self.default_collection = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.second_collection = PhotoCollection.objects.create(
            name='Second Video Collection', root_directory=self.second_root,
        )

    def tearDown(self):
        VideoFile.objects.all().delete()
        FailedVideoFile.objects.all().delete()
        Directory.objects.exclude(collection=self.default_collection).delete()

    def _a_long_enough_fixture(self):
        from django.conf import settings
        video_dir = '/photos/video_samples'
        for filename in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, filename)
            from .video_scripts import _run_ffprobe
            probe = _run_ffprobe(path)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                return path
        self.skipTest("No fixture video clears MIN_VIDEO_DURATION_SECONDS in this environment.")

    def test_new_video_under_second_root_gets_that_collection(self):
        fixture = self._a_long_enough_fixture()
        dest = os.path.join(self.second_root, os.path.basename(fixture))
        shutil.copy(fixture, dest)

        create_video_file(dest)

        video = VideoFile.objects.get(filename=dest)
        self.assertEqual(video.directory.collection_id, self.second_collection.pk)

    def test_video_under_unregistered_path_falls_back_to_default(self):
        fixture = self._a_long_enough_fixture()
        unregistered_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, unregistered_dir, ignore_errors=True)
        dest = os.path.join(unregistered_dir, os.path.basename(fixture))
        shutil.copy(fixture, dest)

        create_video_file(dest)

        video = VideoFile.objects.get(filename=dest)
        self.assertEqual(video.directory.collection_id, self.default_collection.pk)

    def test_backfill_resolves_preexisting_null_video_directory(self):
        # Simulates a Directory created before the fix existed -- plain
        # get_or_create, no collection ever resolved.
        pre_existing_dir = Directory.objects.create(dir_path=self.second_root)
        self.assertIsNone(pre_existing_dir.collection)

        fixture = self._a_long_enough_fixture()
        dest = os.path.join(self.second_root, os.path.basename(fixture))
        shutil.copy(fixture, dest)
        video = VideoFile.objects.create(
            filename=dest, directory=pre_existing_dir, width=100, height=100, duration_seconds=10,
        )

        call_command('backfill_video_directory_collection')

        pre_existing_dir.refresh_from_db()
        self.assertEqual(pre_existing_dir.collection_id, self.second_collection.pk)

    def test_backfill_dry_run_does_not_write(self):
        pre_existing_dir = Directory.objects.create(dir_path=self.second_root)
        VideoFile.objects.create(
            filename=os.path.join(self.second_root, 'a.mp4'), directory=pre_existing_dir,
            width=100, height=100, duration_seconds=10,
        )

        call_command('backfill_video_directory_collection', '--dry-run')

        pre_existing_dir.refresh_from_db()
        self.assertIsNone(pre_existing_dir.collection)

    def test_backfill_does_not_touch_image_only_directories(self):
        # A Directory with no VideoFile at all (NULL collection for some
        # other, unrelated reason) must not be touched by this video-
        # specific backfill.
        image_only_dir = Directory.objects.create(dir_path=tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, image_only_dir.dir_path, ignore_errors=True)
        self.assertIsNone(image_only_dir.collection)

        call_command('backfill_video_directory_collection')

        image_only_dir.refresh_from_db()
        self.assertIsNone(image_only_dir.collection)


class VideoCrossCollectionDuplicateTests(TestCase):
    """create_video_file()'s content_hash-based duplicate detection
    already prevented a second VideoFile row for the same real content
    landing in a second collection -- these confirm the real gap that
    was missing until 2026-10-21: that the primary becomes queryable as
    "visible from" the second collection too (VideoFile.collections),
    not just left attributed to its own original home. Exact mirror of
    PhotoCollectionTests' equivalent image-side test."""

    def setUp(self):
        self.default_collection = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.second_collection = PhotoCollection.objects.create(
            name='Second Video Collection', root_directory=self.second_root,
        )

    def tearDown(self):
        VideoFile.objects.all().delete()
        FailedVideoFile.objects.all().delete()
        DuplicateVideoFile.objects.all().delete()
        Directory.objects.exclude(collection=self.default_collection).delete()

    def _a_long_enough_fixture(self):
        from django.conf import settings
        video_dir = '/photos/video_samples'
        for filename in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, filename)
            from .video_scripts import _run_ffprobe
            probe = _run_ffprobe(path)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                return path
        self.skipTest("No fixture video clears MIN_VIDEO_DURATION_SECONDS in this environment.")

    def test_duplicate_found_under_second_collection_cross_links_the_original(self):
        fixture = self._a_long_enough_fixture()

        original_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, original_dir, ignore_errors=True)
        original_path = os.path.join(original_dir, os.path.basename(fixture))
        shutil.copy(fixture, original_path)
        create_video_file(original_path)
        original = VideoFile.objects.get(filename=original_path)

        # Confirm its home really did resolve to the default collection
        # (original_dir was never registered as any collection's root).
        self.assertEqual(original.directory.collection_id, self.default_collection.pk)
        self.assertIn(self.default_collection, original.collections.all())
        self.assertNotIn(self.second_collection, original.collections.all())

        dup_path = os.path.join(self.second_root, 'duplicate_' + os.path.basename(fixture))
        shutil.copy(fixture, dup_path)
        create_video_file(dup_path)

        # No second VideoFile row -- same content, recorded as a
        # DuplicateVideoFile instead.
        self.assertFalse(VideoFile.objects.filter(filename=dup_path).exists())
        dup = DuplicateVideoFile.objects.get(filename=dup_path)
        self.assertEqual(dup.original_id, original.pk)
        self.assertEqual(dup.collection_id, self.second_collection.pk)

        # The ORIGINAL (whose home is still the default collection) is
        # now also visible from the second collection.
        original.refresh_from_db()
        self.assertIn(self.default_collection, original.collections.all())
        self.assertIn(self.second_collection, original.collections.all())


@override_settings(VIDEO_TRANSCODE_ROOT='/tmp/video_transcode_test')
class TranscodeVideoTests(TestCase):
    """Phase 5 (transcode pipeline), added 2026-10-15. Mocks subprocess.run
    entirely (real ffmpeg encoding is slow and this is about the
    control-flow/bookkeeping around it, not ffmpeg itself) -- a fake
    success writes a dummy file at the expected .tmp path (since the
    real code checks os.path.isfile() on it) before returning
    returncode=0."""

    def setUp(self):
        from .video_scripts import _sharded_transcode_path
        self.transcode_root = settings.VIDEO_TRANSCODE_ROOT
        if os.path.exists(self.transcode_root):
            shutil.rmtree(self.transcode_root)
        os.makedirs(self.transcode_root)
        self._sharded_transcode_path = _sharded_transcode_path

    def tearDown(self):
        shutil.rmtree(self.transcode_root, ignore_errors=True)
        VideoFile.objects.all().delete()

    _UNSET = object()

    def _make_video(self, codec='hevc', content_hash=_UNSET, **overrides):
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/transcode_test_videos')
        if content_hash is self._UNSET:
            content_hash = binascii.hexlify(os.urandom(16)).decode()
        defaults = dict(
            filename=f'/tmp/transcode_test_videos/{overrides.pop("name", "a")}.mp4',
            directory=directory, width=320, height=240, duration_seconds=10.0,
            codec=codec, content_hash=content_hash,
        )
        defaults.update(overrides)
        return VideoFile.objects.create(**defaults)

    def _fake_success(self, tmp_path_arg_index=-1):
        def side_effect(cmd, **kwargs):
            tmp_path = cmd[tmp_path_arg_index]
            with open(tmp_path, 'wb') as f:
                f.write(b'fake mp4 bytes')
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        return side_effect

    def test_successful_transcode_sets_path_and_clears_failure_state(self):
        from .video_scripts import transcode_video
        video = self._make_video()

        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            transcode_video(video)

        video.refresh_from_db()
        self.assertFalse(video.transcode_failed)
        self.assertIsNone(video.transcode_error)
        self.assertIsNotNone(video.transcoded_path)
        expected_rel = self._sharded_transcode_path(video.content_hash)
        self.assertEqual(video.transcoded_path, expected_rel)
        out_path = os.path.join(self.transcode_root, expected_rel)
        self.assertTrue(os.path.isfile(out_path))
        # The .tmp sibling was renamed away, not left behind.
        self.assertFalse(os.path.isfile(out_path + '.tmp'))

    def test_ffmpeg_nonzero_exit_marks_failed(self):
        from .video_scripts import transcode_video
        video = self._make_video()

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 1, stdout='', stderr='real ffmpeg error output')
            transcode_video(video)

        video.refresh_from_db()
        self.assertTrue(video.transcode_failed)
        self.assertIn('real ffmpeg error output', video.transcode_error)
        self.assertIsNone(video.transcoded_path)

    def test_ffmpeg_timeout_marks_failed(self):
        from .video_scripts import transcode_video
        video = self._make_video()

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd='ffmpeg', timeout=3600)
            transcode_video(video)

        video.refresh_from_db()
        self.assertTrue(video.transcode_failed)
        self.assertIn('timed out', video.transcode_error)

    def test_missing_content_hash_fails_without_calling_ffmpeg(self):
        from .video_scripts import transcode_video
        video = self._make_video(content_hash=None)

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            transcode_video(video)
            mock_run.assert_not_called()

        video.refresh_from_db()
        self.assertTrue(video.transcode_failed)

    def test_pending_videos_skips_native_codecs(self):
        from .video_scripts import transcode_pending_videos
        native = self._make_video(codec='h264', name='native')
        needs_it = self._make_video(codec='hevc', name='needs_it')

        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            done, failed = transcode_pending_videos()

        native.refresh_from_db()
        needs_it.refresh_from_db()
        self.assertIsNone(native.transcoded_path)
        self.assertIsNotNone(needs_it.transcoded_path)
        self.assertEqual((done, failed), (1, 0))

    def test_pending_videos_skips_already_done_or_failed(self):
        from .video_scripts import transcode_pending_videos
        already_done = self._make_video(codec='hevc', name='done', transcoded_path='xx/y/already.mp4')
        already_failed = self._make_video(codec='hevc', name='failed', transcode_failed=True)

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            done, failed = transcode_pending_videos()
            mock_run.assert_not_called()

        self.assertEqual((done, failed), (0, 0))

    def test_max_runtime_seconds_stops_between_videos(self):
        from .video_scripts import transcode_pending_videos
        self._make_video(codec='hevc', name='first')
        self._make_video(codec='hevc', name='second')

        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            done, failed = transcode_pending_videos(max_runtime_seconds=0)

        self.assertEqual((done, failed), (0, 0))

    def test_sharded_path_matches_thumbnail_style_convention(self):
        from .video_scripts import _sharded_transcode_path
        content_hash = 'abcdef0123456789'
        rel = _sharded_transcode_path(content_hash)
        self.assertEqual(rel, os.path.join('ab', 'c', 'abcdef0123456789.mp4'))

    def test_deleting_video_removes_transcoded_file(self):
        from .video_scripts import transcode_video
        video = self._make_video()
        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            transcode_video(video)
        video.refresh_from_db()
        out_path = os.path.join(self.transcode_root, video.transcoded_path)
        self.assertTrue(os.path.isfile(out_path))

        video.delete()

        self.assertFalse(os.path.isfile(out_path))


class DeleteRemovedVideosTests(TestCase):
    """delete_removed_videos() -- the video-side mirror of scripts.py's
    delete_removed_photos(), built 2026-09-15 (previously a deliberate
    Phase 1 scope cut -- see CLAUDE.md) once a real case came up (a
    VTS_*-named DVD-rip .avi moved out of the library, leaving its
    VideoFile row and Face row stranded)."""

    VIDEO_DIR = '/photos/video_samples'

    def tearDown(self):
        VideoFile.objects.all().delete()

    def _long_enough_candidates(self):
        from django.conf import settings
        from .video_scripts import _run_ffprobe

        out = []
        for filename in sorted(os.listdir(self.VIDEO_DIR)):
            src = os.path.join(self.VIDEO_DIR, filename)
            probe = _run_ffprobe(src)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                out.append(filename)
        if not out:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        return out

    def _a_real_fixture_copy(self):
        # Cycles through distinct real fixtures on repeated calls (the
        # real dev/test fixture dir has 8) so two calls in the same test
        # produce genuinely different file content -- needed since
        # content-hash-based duplicate detection (2026-09-22) now
        # recognizes two byte-identical copies as the same video and
        # only ever creates one VideoFile row for them. Falls back to
        # appending random trailing bytes if fewer distinct fixtures
        # exist than calls made (e.g. CI's single synthetic stub) --
        # verified safe against a real synthetic fixture: appending
        # bytes after a container's own real data doesn't disturb
        # ffprobe's ability to read it.
        candidates = self._long_enough_candidates()
        count = getattr(self, '_fixture_copy_count', 0)
        filename = candidates[count % len(candidates)]
        self._fixture_copy_count = count + 1

        src = os.path.join(self.VIDEO_DIR, filename)
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        dest = os.path.join(tmp_dir, filename)
        shutil.copy(src, dest)
        if count >= len(candidates):
            with open(dest, 'ab') as f:
                f.write(os.urandom(64))
        return dest

    def test_delete_removed_videos_removes_vanished_rows_only(self):
        # Two-phase behavior (see VideoFile.missing_since's own comment):
        # a vanished file is flagged, not deleted, on the first pass that
        # notices it, and only actually deleted once it's been missing
        # longer than MISSING_FILE_GRACE_PERIOD.
        gone_path = self._a_real_fixture_copy()
        create_video_file(gone_path)
        self.assertTrue(VideoFile.objects.filter(filename=gone_path).exists())

        staying_path = self._a_real_fixture_copy()
        create_video_file(staying_path)

        os.remove(gone_path)
        delete_removed_videos()
        self.assertTrue(VideoFile.objects.filter(filename=gone_path).exists())
        self.assertIsNotNone(VideoFile.objects.get(filename=gone_path).missing_since)

        VideoFile.objects.filter(filename=gone_path).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_videos()

        self.assertFalse(VideoFile.objects.filter(filename=gone_path).exists())
        self.assertTrue(VideoFile.objects.filter(filename=staying_path).exists())

    def test_delete_removed_videos_within_grace_period_is_not_yet_deleted(self):
        gone_path = self._a_real_fixture_copy()
        create_video_file(gone_path)
        os.remove(gone_path)

        VideoFile.objects.filter(filename=gone_path).update(
            missing_since=timezone.now() - timedelta(hours=1)
        )
        delete_removed_videos()

        self.assertTrue(VideoFile.objects.filter(filename=gone_path).exists())

    def test_delete_removed_videos_reappearing_clears_missing_since(self):
        path = self._a_real_fixture_copy()
        create_video_file(path)
        VideoFile.objects.filter(filename=path).update(missing_since=timezone.now())

        delete_removed_videos()

        self.assertIsNone(VideoFile.objects.get(filename=path).missing_since)

    def test_delete_removed_videos_cleans_up_attached_face_and_thumbnail(self):
        from face_manager.models import Person

        gone_path = self._a_real_fixture_copy()
        create_video_file(gone_path)
        video = VideoFile.objects.get(filename=gone_path)

        person, _ = Person.objects.get_or_create(person_name="DeleteRemovedVideosTests Person")
        face = Face(
            source_video_file=video, declared_name=person,
            box_left=1, box_top=1, box_right=min(40, video.width - 1), box_bottom=min(40, video.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        thumb_path = face.face_thumbnail.path
        self.assertTrue(os.path.isfile(thumb_path))

        os.remove(gone_path)
        VideoFile.objects.filter(pk=video.pk).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_videos()

        self.assertFalse(VideoFile.objects.filter(pk=video.pk).exists())
        self.assertFalse(Face.objects.filter(pk=face.pk).exists())
        self.assertFalse(os.path.isfile(thumb_path))


class VideoContentHashBackfillAndMergeTests(TestCase):
    """backfill_video_content_hash + merge_duplicate_videofiles -- the
    two management commands that resolve VideoFile rows ingested BEFORE
    content-hash-based duplicate detection existed (see
    VideoContentHashDuplicateTests above, which covers the same
    protection at ingestion time going forward). Real motivating case
    (2026-09-22): 5 Face rows all turned out to be the same real frame
    because 5 differently-named files were confirmed byte-for-byte
    identical, each independently face-extracted.

    Builds rows directly (not via create_video_file(), which now
    prevents this scenario from recurring) to simulate the pre-existing
    contamination these commands are meant to clean up."""

    VIDEO_DIR = '/photos/video_samples'

    def tearDown(self):
        VideoFile.objects.all().delete()

    def _a_real_fixture_copy(self):
        from django.conf import settings
        from .video_scripts import _run_ffprobe

        for filename in sorted(os.listdir(self.VIDEO_DIR)):
            src = os.path.join(self.VIDEO_DIR, filename)
            probe = _run_ffprobe(src)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                tmp_dir = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
                dest = os.path.join(tmp_dir, filename)
                shutil.copy(src, dest)
                return dest
        self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")

    def _pre_content_hash_video(self, path):
        # Ingest normally (populates content_hash going forward), then
        # strip it back off -- simulates a row that predates the field,
        # without duplicating all of create_video_file()'s own field
        # population logic here.
        create_video_file(path)
        video = VideoFile.objects.get(filename=path)
        VideoFile.objects.filter(pk=video.pk).update(content_hash=None)
        video.refresh_from_db()
        return video

    def _ingest(self, path):
        create_video_file(path)
        return VideoFile.objects.get(filename=path)

    def _make_duplicate_row(self, path, primary, isProcessed=False):
        # Bypasses create_video_file()'s own (now-live) duplicate
        # detection deliberately -- constructs a second real row
        # directly, the same shape a row ingested BEFORE this feature
        # existed would have: identical content_hash/dimensions, a
        # genuinely different path that's still really on disk. This is
        # exactly the pre-existing contamination merge_duplicate_
        # videofiles exists to clean up.
        return VideoFile.objects.create(
            filename=path, directory=primary.directory,
            width=primary.width, height=primary.height,
            duration_seconds=primary.duration_seconds,
            content_hash=primary.content_hash, isProcessed=isProcessed,
        )

    def _attach_face(self, video, person=None):
        from face_manager.models import Person
        if person is None:
            person, _ = Person.objects.get_or_create(person_name="VideoContentHashMergeTests Person")
        face = Face(
            source_video_file=video, declared_name=person,
            box_left=1, box_top=1,
            box_right=min(40, video.width - 1), box_bottom=min(40, video.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_backfill_populates_content_hash(self):
        path = self._a_real_fixture_copy()
        video = self._pre_content_hash_video(path)
        self.assertIsNone(video.content_hash)

        from django.core.management import call_command
        call_command('backfill_video_content_hash')

        video.refresh_from_db()
        self.assertEqual(video.content_hash, compute_video_content_hash(path))

    def test_backfill_dry_run_makes_no_changes(self):
        path = self._a_real_fixture_copy()
        video = self._pre_content_hash_video(path)

        from django.core.management import call_command
        call_command('backfill_video_content_hash', '--dry-run')

        video.refresh_from_db()
        self.assertIsNone(video.content_hash)

    def test_backfill_skips_file_vanished_from_disk(self):
        path = self._a_real_fixture_copy()
        video = self._pre_content_hash_video(path)
        os.remove(path)

        from django.core.management import call_command
        call_command('backfill_video_content_hash')

        video.refresh_from_db()
        self.assertIsNone(video.content_hash)

    def test_merge_transfers_faces_and_deletes_unprocessed_duplicate(self):
        # Two rows, same real content, only the primary has been
        # face-processed -- the safe case: merge should keep the
        # processed one, transfer nothing (the duplicate has no faces of
        # its own), and delete the duplicate row.
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        duplicate = self._make_duplicate_row(path_b, primary, isProcessed=False)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        face = self._attach_face(primary)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertFalse(VideoFile.objects.filter(pk=duplicate.pk).exists())
        self.assertTrue(VideoFile.objects.filter(pk=primary.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face.pk, source_video_file=primary).exists())

    def test_merge_transfers_faces_from_unprocessed_duplicate_onto_primary(self):
        # The duplicate itself has a stray face (e.g. hand-added, or from
        # some other path never marking isProcessed) -- confirm it's
        # really transferred, not silently dropped.
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        duplicate = self._make_duplicate_row(path_b, primary, isProcessed=False)
        stray_face = self._attach_face(duplicate)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertFalse(VideoFile.objects.filter(pk=duplicate.pk).exists())
        stray_face.refresh_from_db()
        self.assertEqual(stray_face.source_video_file_id, primary.pk)

    def _attach_face_at(self, video, first, last, person=None, validated=False,
                         embedding=None, kps=None, det_score=None):
        from face_manager.models import Person
        if person is None:
            person, _ = Person.objects.get_or_create(person_name=settings.SOFT_IGNORE_NAME)
        face = Face(
            source_video_file=video, declared_name=person, validated=validated,
            box_left=1, box_top=1,
            box_right=min(40, video.width - 1), box_bottom=min(40, video.height - 1),
            video_first_timestamp_seconds=first, video_last_timestamp_seconds=last,
            face_encoding_512=embedding, kps=kps, det_score=det_score,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_merge_resolves_both_processed_duplicates_keeping_unmatched_faces(self):
        # Both rows already independently face-extracted, and the two
        # faces have no timestamps at all (can't be matched) -- this
        # used to be left alone entirely ("unresolved"); now the videos
        # still merge, and since the faces can't be confidently matched
        # to each other, BOTH survive (transferred onto the primary),
        # never silently discarded.
        path_a = self._a_real_fixture_copy()
        video_a = self._ingest(path_a)
        VideoFile.objects.filter(pk=video_a.pk).update(isProcessed=True)
        video_a.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        video_b = self._make_duplicate_row(path_b, video_a, isProcessed=True)
        face_a = self._attach_face(video_a)
        face_b = self._attach_face(video_b)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertTrue(VideoFile.objects.filter(pk=video_a.pk).exists())
        self.assertFalse(VideoFile.objects.filter(pk=video_b.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face_a.pk, source_video_file=video_a).exists())
        face_b.refresh_from_db()
        self.assertEqual(face_b.source_video_file_id, video_a.pk)

    def test_merge_dedupes_matched_faces_keeping_the_labeled_survivor(self):
        # Same real span on both copies (the deterministic-pipeline
        # case this feature was actually built for) -- one copy's face
        # is a real, declared identification; the other is just
        # .ignore. The declared one must survive, not be discarded
        # because the un-labeled copy happened to live on the primary.
        from face_manager.models import Person
        path_a = self._a_real_fixture_copy()
        video_a = self._ingest(path_a)
        VideoFile.objects.filter(pk=video_a.pk).update(isProcessed=True)
        video_a.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        video_b = self._make_duplicate_row(path_b, video_a, isProcessed=True)

        real_person, _ = Person.objects.get_or_create(person_name="Test Real Person")
        ignore_face = self._attach_face_at(video_a, 10.0, 12.0)
        labeled_face = self._attach_face_at(video_b, 10.0, 12.0, person=real_person)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertFalse(Face.objects.filter(pk=ignore_face.pk).exists())
        labeled_face.refresh_from_db()
        self.assertEqual(labeled_face.source_video_file_id, video_a.pk)
        self.assertEqual(labeled_face.declared_name.person_name, "Test Real Person")
        self.assertEqual(Face.objects.filter(source_video_file=video_a).count(), 1)

    def test_merge_keeps_faces_whose_timestamps_dont_match_as_distinct(self):
        # Two faces at clearly different points in the clip -- these are
        # two different real detections (or the same person appearing
        # twice), not a duplicate pair, and must both survive.
        path_a = self._a_real_fixture_copy()
        video_a = self._ingest(path_a)
        VideoFile.objects.filter(pk=video_a.pk).update(isProcessed=True)
        video_a.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        video_b = self._make_duplicate_row(path_b, video_a, isProcessed=True)

        face_early = self._attach_face_at(video_a, 1.0, 2.0)
        face_late = self._attach_face_at(video_b, 40.0, 41.0)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertTrue(Face.objects.filter(pk=face_early.pk, source_video_file=video_a).exists())
        face_late.refresh_from_db()
        self.assertEqual(face_late.source_video_file_id, video_a.pk)
        self.assertEqual(Face.objects.filter(source_video_file=video_a).count(), 2)

    def test_merge_dry_run_makes_no_changes(self):
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        duplicate = self._make_duplicate_row(path_b, primary, isProcessed=False)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--dry-run')

        self.assertTrue(VideoFile.objects.filter(pk=primary.pk).exists())
        self.assertTrue(VideoFile.objects.filter(pk=duplicate.pk).exists())

    def test_merge_rerun_finds_nothing(self):
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        self._make_duplicate_row(path_b, primary, isProcessed=False)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')
        self.assertEqual(VideoFile.objects.count(), 1)

        call_command('merge_duplicate_videofiles', '--yes')
        self.assertEqual(VideoFile.objects.count(), 1)


class VideoPlaybackPathTests(TestCase):
    """compute_video_playback_path() (the single source of truth for
    VideoFile.playback_path, shared between create_video_file() and
    backfill_video_playback_path) and the backfill command itself --
    see Phase 4's "slideshow serving" write-up in CLAUDE.md."""

    def tearDown(self):
        VideoFile.objects.all().delete()

    @override_settings(PHOTO_ROOT='/photos', VIDEO_ROOT='/videos')
    def test_photo_root_path_uses_full_res_mount(self):
        from .video_scripts import compute_video_playback_path
        url = compute_video_playback_path('/photos/some dir/clip.mp4')
        self.assertIn('/full_res/some%20dir/clip.mp4', url)
        self.assertNotIn('video_original', url)

    @override_settings(PHOTO_ROOT='/photos', VIDEO_ROOT='/videos')
    def test_video_root_path_uses_video_original_mount(self):
        from .video_scripts import compute_video_playback_path
        url = compute_video_playback_path('/videos/Our_Home_Videos/clip.mp4')
        self.assertIn('/video_original/Our_Home_Videos/clip.mp4', url)

    def _bare_video(self, filename):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        return VideoFile.objects.create(
            filename=filename, directory=directory, width=100, height=100,
            playback_path=None,
        )

    def test_backfill_populates_missing_playback_path(self):
        video = self._bare_video('/videos/Our_Home_Videos/clip1.mp4')
        self.assertIsNone(video.playback_path)

        call_command('backfill_video_playback_path')

        video.refresh_from_db()
        self.assertIsNotNone(video.playback_path)
        self.assertIn('video_original', video.playback_path)

    def test_backfill_dry_run_does_not_write(self):
        video = self._bare_video('/videos/Our_Home_Videos/clip2.mp4')
        call_command('backfill_video_playback_path', '--dry-run')
        video.refresh_from_db()
        self.assertIsNone(video.playback_path)

    def test_backfill_skips_already_populated_rows(self):
        video = self._bare_video('/videos/Our_Home_Videos/clip3.mp4')
        VideoFile.objects.filter(pk=video.pk).update(playback_path='https://example.com/already/set.mp4')
        call_command('backfill_video_playback_path')
        video.refresh_from_db()
        self.assertEqual(video.playback_path, 'https://example.com/already/set.mp4')


class MatchDuplicateVideoFacesTests(unittest.TestCase):
    """Unit tests for merge_duplicate_videofiles.match_duplicate_video_faces
    and _survivor_key -- pure functions, no DB access needed. See that
    command's module docstring for the real motivating case (5 Face rows
    from 5 byte-identical video copies, matched by near-exact timestamp
    since the extraction pipeline is deterministic over identical input
    bytes)."""

    def _face(self, first, last, embedding=None):
        return {'first': first, 'last': last, 'embedding': embedding}

    def test_close_timestamps_match(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0)]
        other = [self._face(10.001, 12.002)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [(0, 0)])

    def test_timestamps_too_far_apart_do_not_match(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0)]
        other = [self._face(40.0, 41.0)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [])

    def test_missing_timestamps_never_match(self):
        # A face with no timestamps at all (e.g. a hand-added test/stub
        # row) can't be confidently matched -- must stay unmatched
        # rather than guessed at.
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(None, None)]
        other = [self._face(None, None)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [])

    def test_close_timestamps_but_dissimilar_embeddings_do_not_match(self):
        # Same instant, but the embeddings disagree -- guards against
        # two different real people who happen to overlap in time
        # within tolerance. Near-orthogonal 3-d vectors for clarity.
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0, embedding=[1.0, 0.0, 0.0])]
        other = [self._face(10.0, 12.0, embedding=[0.0, 1.0, 0.0])]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [])

    def test_close_timestamps_and_similar_embeddings_do_match(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0, embedding=[1.0, 0.0, 0.0])]
        other = [self._face(10.0, 12.0, embedding=[0.99, 0.01, 0.0])]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [(0, 0)])

    def test_optimal_assignment_picks_the_correct_pairing_not_first_match(self):
        # Two primary faces, two other faces, each clearly closer in
        # time to a DIFFERENT one of the two -- confirms the matcher
        # solves the whole matrix at once rather than greedily grabbing
        # whichever pair it sees first.
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0), self._face(50.0, 52.0)]
        other = [self._face(50.05, 52.05), self._face(10.05, 12.05)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_survivor_key_prefers_validated(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import _survivor_key
        validated = {'id': 2, 'validated': True, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        unvalidated = {'id': 1, 'validated': False, 'person_name': 'Real Name', 'has_kps': True, 'has_det_score': True}
        self.assertLess(_survivor_key(validated), _survivor_key(unvalidated))

    def test_survivor_key_prefers_real_label_over_ignore(self):
        # The real motivating case: a genuine person name must beat
        # .ignore even though .ignore isn't the blank sentinel either.
        from filepopulator.management.commands.merge_duplicate_videofiles import _survivor_key
        labeled = {'id': 2, 'validated': False, 'person_name': 'Emily Wilson', 'has_kps': False, 'has_det_score': False}
        ignored = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': True, 'has_det_score': True}
        self.assertLess(_survivor_key(labeled), _survivor_key(ignored))

    def test_survivor_key_prefers_kps_then_det_score_then_lowest_id(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import _survivor_key
        has_kps = {'id': 5, 'validated': False, 'person_name': '.ignore', 'has_kps': True, 'has_det_score': False}
        no_kps = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': True}
        self.assertLess(_survivor_key(has_kps), _survivor_key(no_kps))

        has_det_score = {'id': 5, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': True}
        no_det_score = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        self.assertLess(_survivor_key(has_det_score), _survivor_key(no_det_score))

        lower_id = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        higher_id = {'id': 2, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        self.assertLess(_survivor_key(lower_id), _survivor_key(higher_id))


class ParseExifVideoDateTests(unittest.TestCase):
    """Unit tests for video_scripts._parse_exif_date() -- no DB/file
    access needed. Covers the real parsing bug found against a real Sony
    camcorder file: dateutil.parser silently defaulted the date portion
    to today() for exiftool's 'YYYY:MM:DD ...' format instead of raising,
    which explicit strptime formats don't do."""

    def test_no_timezone_offset(self):
        date = _parse_exif_date('2025:11:27 01:31:30')
        self.assertEqual((date.year, date.month, date.day), (2025, 11, 27))
        self.assertEqual((date.hour, date.minute, date.second), (1, 31, 30))

    def test_with_timezone_offset(self):
        date = _parse_exif_date('2018:12:14 20:15:14-04:00')
        self.assertEqual((date.year, date.month, date.day), (2018, 12, 14))
        self.assertEqual(date.hour, 20)

    def test_with_timezone_offset_and_zone_name(self):
        # The exact real-world shape that broke dateutil.parser: a
        # numeric offset followed by a zone-name word ("DST").
        date = _parse_exif_date('2018:12:14 20:15:14-04:00 DST')
        self.assertEqual((date.year, date.month, date.day), (2018, 12, 14))
        self.assertEqual(date.hour, 20)

    def test_empty_or_none_returns_none(self):
        self.assertIsNone(_parse_exif_date(None))
        self.assertIsNone(_parse_exif_date(''))

    def test_garbage_string_returns_none_not_todays_date(self):
        # The actual regression: this used to silently return today's
        # date via dateutil.parser instead of None.
        result = _parse_exif_date('not a real date at all')
        self.assertIsNone(result)


@override_settings(OWNTRACKS_LOCATION_API_URL='https://owntracks.example.com',
                    OWNTRACKS_LOCATION_API_KEY='test-key')
class BackfillS23GpsTests(TestCase):
    """backfill_owntracks_gps / filepopulator.owntracks_gps -- see that
    module's own docstring for the real detail this was built around:
    ImageFile.dateTaken/_get_date_taken() throws away the real UTC
    offset, so this re-reads EXIF directly rather than trusting the DB,
    trying each known per-camera offset source in turn (standard EXIF
    OffsetTimeOriginal, Nikon's MakerNotes TimeZone+DaylightSavings).
    exiftool and the HTTP call are both mocked; no real subprocess or
    network calls."""

    FIXTURE_IMG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'ci_fixtures', 'test_imgs_filepopulate', 'gps.jpg',
    )

    def tearDown(self):
        ImageFile.objects.all().delete()

    def _make_face(self, image, person, validated):
        face = Face(
            source_image_file=image, declared_name=person,
            box_left=1, box_top=1, box_right=5, box_bottom=5, validated=validated,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def _bare_image(self, camera_model, gps_lat_decimal=-999, filename=None, dateTaken=None):
        # ImageFile.save() -> _init_image() really opens the file (PIL),
        # so this needs a real tiny image on disk, not just a DB row --
        # same reason other tests in this file copy a real fixture
        # rather than fabricate a path.
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        dest = filename or os.path.join(tmp_dir, f'{camera_model}_{ImageFile.objects.count()}.jpg')
        if filename is None:
            shutil.copy(self.FIXTURE_IMG, dest)
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(dest))
        img = ImageFile.objects.create(
            filename=dest,
            directory=directory,
            pixel_hash=f'hash{ImageFile.objects.count()}',
            file_hash=f'fhash{ImageFile.objects.count()}',
            width=100, height=100, orientation=1,
            dateAdded=timezone.now(), dateModified=timezone.now(),
            camera_model=camera_model,
            gps_lat_decimal=gps_lat_decimal,
        )
        if dateTaken is not None:
            ImageFile.objects.filter(pk=img.pk).update(dateTaken=dateTaken)
            img.refresh_from_db()
        return img

    def _exiftool_result(self, date_str=None, offset_str=None,
                          nikon_timezone=None, nikon_dst=None):
        entry = {}
        if date_str is not None:
            entry['DateTimeOriginal'] = date_str
        if offset_str is not None:
            entry['OffsetTimeOriginal'] = offset_str
        if nikon_timezone is not None:
            entry['TimeZone'] = nikon_timezone
        if nikon_dst is not None:
            entry['DaylightSavings'] = nikon_dst
        return mock.Mock(stdout=json.dumps([entry]))

    def test_nikon_timezone_and_dst_sets_gps(self):
        # Real case this was built from: a Nikon D3200 trip to Italy,
        # clock left on home World Time (-05:00) with DST on -- the
        # true effective offset is -04:00, confirmed against a real
        # OwnTracks match that resolved to Pisa, Italy.
        img = self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', nikon_timezone='-05:00', nikon_dst='Yes')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 43.718, 'longitude': 10.402},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        called_params = mock_get.call_args.kwargs['params']
        self.assertEqual(called_params['datetime'], '2026-06-14T14:28:52-04:00')
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 43.718)

    def test_nikon_without_dst_uses_base_timezone_only(self):
        img = self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:01:14 14:28:52', nikon_timezone='-05:00', nikon_dst='No')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 1.0, 'longitude': 1.0},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        called_params = mock_get.call_args.kwargs['params']
        self.assertEqual(called_params['datetime'], '2026-01-14T14:28:52-05:00')

    def test_exif_offset_preferred_over_nikon_tags_if_both_present(self):
        # Not a real-world case (a phone wouldn't have Nikon MakerNotes
        # tags), but confirms the precedence is deliberate, not
        # order-of-dict-insertion luck.
        from filepopulator.owntracks_gps import _read_offset_datetime
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', offset_str='+01:00',
                             nikon_timezone='-05:00', nikon_dst='Yes')):
            date_str, offset_str = _read_offset_datetime('/fake/path.jpg')
        self.assertEqual(offset_str, '+01:00')

    def test_unparseable_nikon_timezone_is_skipped_not_guessed(self):
        img = self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', nikon_timezone='garbage', nikon_dst='Yes')):
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)
        self.assertEqual(counts['no_offset_tag'], 1)

    def test_command_rejects_camera_model_not_in_allowlist(self):
        with self.assertRaises(CommandError):
            call_command('backfill_owntracks_gps', '--camera-model', 'Some Random Camera')

    def test_command_scopes_to_requested_camera_model_only(self):
        self._bare_image('Galaxy S23+')
        self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            call_command('backfill_owntracks_gps', '--camera-model', 'Galaxy S23+')

        nikon = ImageFile.objects.get(camera_model='NIKON D3200')
        self.assertEqual(nikon.gps_lat_decimal, -999)
        s23 = ImageFile.objects.get(camera_model='Galaxy S23+')
        self.assertEqual(s23.gps_lat_decimal, 40.1)

    def test_start_end_date_excludes_photos_outside_the_window(self):
        # Real motivating case: a camera with years of history predating
        # OwnTracks entirely would otherwise burn a real exiftool+HTTP
        # round trip per photo just to learn "no nearby point," every
        # single run.
        import datetime
        old = self._bare_image('NIKON D3200', dateTaken=datetime.datetime(2014, 10, 25, tzinfo=datetime.timezone.utc))
        in_range = self._bare_image('NIKON D3200', dateTaken=datetime.datetime(2026, 6, 14, tzinfo=datetime.timezone.utc))

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', nikon_timezone='-05:00', nikon_dst='Yes')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 43.718, 'longitude': 10.402},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps(
                camera_models=['NIKON D3200'],
                start_date=datetime.date(2026, 6, 10), end_date=datetime.date(2026, 6, 21),
            )

        self.assertEqual(counts['matched'], 1)
        old.refresh_from_db()
        self.assertEqual(old.gps_lat_decimal, -999)
        in_range.refresh_from_db()
        self.assertEqual(in_range.gps_lat_decimal, 43.718)

    def test_only_scopes_configured_s23_camera_models(self):
        self._bare_image('Galaxy S23+')
        self._bare_image('SM-S916U')
        self._bare_image('iPhone 12 mini')  # must never be touched

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        self.assertEqual(counts['matched'], 2)
        iphone = ImageFile.objects.get(camera_model='iPhone 12 mini')
        self.assertEqual(iphone.gps_lat_decimal, -999)

    def test_builds_offset_aware_datetime_from_fresh_exif_not_db(self):
        # The real detail this whole module exists to get right -- the
        # DB's own dateTaken is unreliable (tagged UTC regardless of the
        # real offset), so the request must reflect the file's own real
        # OffsetTimeOriginal, re-read fresh.
        self._bare_image('Galaxy S23+')

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:06:14 15:18:10', '+01:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 43.7, 'longitude': 10.4},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        called_params = mock_get.call_args.kwargs['params']
        self.assertEqual(called_params['datetime'], '2026-06-14T15:18:10+01:00')

    def test_200_sets_gps_and_source(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 40.1)
        self.assertEqual(img.gps_lon_decimal, -111.9)
        self.assertEqual(img.gps_source, 'owntracks')

    def test_404_leaves_gps_untouched(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404, text='nothing within 15 minutes')
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)
        self.assertEqual(counts['no_nearby_point'], 1)

    def test_401_aborts_the_whole_run(self):
        self._bare_image('Galaxy S23+')
        self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=401, text='bad key')
            from filepopulator.owntracks_gps import backfill_camera_gps, OwnTracksConfigError
            with self.assertRaises(OwnTracksConfigError):
                backfill_camera_gps()

    def test_missing_offset_tag_is_skipped_not_guessed(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', None)):
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)
        self.assertEqual(counts['no_offset_tag'], 1)

    def test_dry_run_does_not_write(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps(dry_run=True)

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_already_populated_rows_are_excluded(self):
        self._bare_image('Galaxy S23+', gps_lat_decimal=40.5)
        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_run:
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()
        mock_run.assert_not_called()

    def test_failed_attempt_is_recorded_and_skipped_on_next_run(self):
        from filepopulator.models import GpsEstimationAttempt
        from filepopulator.owntracks_gps import backfill_camera_gps

        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404, text='nothing nearby')
            counts = backfill_camera_gps()
        self.assertEqual(counts['no_nearby_point'], 1)
        attempt = GpsEstimationAttempt.objects.get(image=img)
        self.assertEqual(attempt.reason, GpsEstimationAttempt.REASON_NO_NEARBY_POINT)

        # Second run, no --retry-failed -- must not even attempt exiftool.
        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_run:
            counts2 = backfill_camera_gps()
        mock_run.assert_not_called()
        self.assertEqual(counts2['no_nearby_point'], 0)

    def test_no_offset_tag_attempt_is_never_retried_even_with_retry_failed(self):
        from filepopulator.models import GpsEstimationAttempt
        from filepopulator.owntracks_gps import backfill_camera_gps

        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', None)):
            backfill_camera_gps()
        self.assertEqual(
            GpsEstimationAttempt.objects.get(image=img).reason,
            GpsEstimationAttempt.REASON_NO_OFFSET_TAG,
        )

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_run:
            backfill_camera_gps(retry_failed=True)
        mock_run.assert_not_called()

    def test_retry_failed_reattempts_no_nearby_point_and_can_succeed(self):
        from filepopulator.owntracks_gps import backfill_camera_gps

        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404, text='nothing nearby')
            backfill_camera_gps()

        # Simulates the OwnTracks service's own history growing -- the
        # exact real case this session hit.
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            counts = backfill_camera_gps(retry_failed=True)

        self.assertEqual(counts['matched'], 1)
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 40.1)
        from filepopulator.models import GpsEstimationAttempt
        self.assertFalse(GpsEstimationAttempt.objects.filter(image=img).exists())

    def test_missing_config_raises_clear_error(self):
        self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             override_settings(OWNTRACKS_LOCATION_API_URL=None, OWNTRACKS_LOCATION_API_KEY=None):
            from filepopulator.owntracks_gps import backfill_camera_gps, OwnTracksConfigError
            with self.assertRaises(OwnTracksConfigError):
                backfill_camera_gps()

    def test_management_command_runs_end_to_end(self):
        self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            call_command('backfill_owntracks_gps')

    def test_validated_face_for_gps_backfill_source_person_is_eligible(self):
        # A camera NOT in CameraGpsBackfillConfig at all -- eligibility
        # here must come entirely from the verified-person path.
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        self.assertEqual(counts['matched'], 1)
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 40.1)

    def test_unvalidated_face_for_gps_backfill_source_person_is_not_eligible(self):
        # Only a CONFIRMED (validated) appearance is a real "this person
        # was physically here" signal -- a mere proposed/unconfirmed
        # candidate match isn't.
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, False)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        mock_exiftool.assert_not_called()
        self.assertEqual(counts['matched'], 0)
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_validated_face_for_non_backfill_source_person_is_not_eligible(self):
        # The person must have gps_backfill_source=True specifically --
        # a validated face for anyone else doesn't make a photo eligible.
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Someone Else', gps_backfill_source=False)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        mock_exiftool.assert_not_called()
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_include_verified_person_false_excludes_it(self):
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps(include_verified_person=False)

        mock_exiftool.assert_not_called()
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_no_verified_person_management_command_flag(self):
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            call_command('backfill_owntracks_gps', '--no-verified-person')

        mock_exiftool.assert_not_called()
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_eligible_via_both_camera_and_person_is_not_double_processed(self):
        # ImageFile.objects.filter(eligibility, ...) must .distinct() --
        # an image matching BOTH the camera allowlist AND the
        # verified-person clause via more than one Face row shouldn't be
        # joined into duplicate result rows.
        from face_manager.models import Person
        img = self._bare_image('Galaxy S23+')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)
        person2 = Person.objects.create(person_name='Also Tracked', gps_backfill_source=True)
        self._make_face(img, person2, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        self.assertEqual(counts['matched'], 1)

    def test_corrupted_ancient_date_is_excluded_from_the_queue(self):
        # Real production bug (2026-10-21): confirmed ImageFile.dateTaken
        # corruption (e.g. year 0102) sorted first under order_by(
        # 'dateTaken'), burning the whole run's budget on guaranteed-
        # failure rows before ever reaching a real candidate.
        # dateTakenValid=True on every one of these (it only tracks "a
        # date was parsed from somewhere," not "the date is plausible"),
        # so MIN_SANE_DATE_TAKEN_YEAR is a real, separate floor.
        import datetime
        old = self._bare_image(
            'Galaxy S23+', dateTaken=datetime.datetime(102, 10, 9, tzinfo=datetime.timezone.utc))

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        mock_exiftool.assert_not_called()
        from filepopulator.models import GpsEstimationAttempt
        self.assertFalse(GpsEstimationAttempt.objects.filter(image=old).exists())
        old.refresh_from_db()
        self.assertEqual(old.gps_lat_decimal, -999)

    def test_camera_and_person_populations_each_get_a_fair_share_of_limit(self):
        # Real production bug (2026-10-21): adding the verified-person
        # path grew the eligible queue substantially (camera-allowlist-
        # only backlog ~11k -> ~22k combined), and a single combined
        # date-ordered queue let the larger population completely starve
        # the other out of ever being reached. Each population must now
        # get its own fair share of `limit` instead.
        from face_manager.models import Person
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)

        camera_imgs = [self._bare_image('Galaxy S23+') for _ in range(3)]
        person_imgs = []
        for _ in range(3):
            img = self._bare_image('Totally Unenrolled Camera')
            self._make_face(img, person, True)
            person_imgs.append(img)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404)  # "no nearby point"
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps(limit=2)

        from filepopulator.models import GpsEstimationAttempt
        touched_camera = GpsEstimationAttempt.objects.filter(
            image__in=camera_imgs).count()
        touched_person = GpsEstimationAttempt.objects.filter(
            image__in=person_imgs).count()
        self.assertEqual(touched_camera, 1)
        self.assertEqual(touched_person, 1)


class PhotoCollectionTests(TestCase):
    """PhotoCollection: resolution, per-collection Directory assignment
    at ingestion, and the cross-collection duplicate-marking mechanism
    (ImageFile.collections gaining an extra collection when a LATER
    scan of a different collection discovers the same photo content --
    this is what makes "find all faces for a person visible from
    collection X, even if the source file's home is collection Y"
    actually answerable). See filepopulator/models.py's PhotoCollection
    for the full design discussion."""

    FIXTURE_IMG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'ci_fixtures', 'test_imgs_filepopulate', 'gps.jpg',
    )

    def setUp(self):
        self.default_collection = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.second_collection = PhotoCollection.objects.create(
            name='Second Collection', root_directory=self.second_root,
        )

    def tearDown(self):
        ImageFile.objects.all().delete()
        Directory.objects.exclude(collection=self.default_collection).delete()

    def test_resolve_for_path_matches_registered_root(self):
        nested = os.path.join(self.second_root, 'sub', 'dir')
        resolved = PhotoCollection.resolve_for_path(nested)
        self.assertEqual(resolved.pk, self.second_collection.pk)

    def test_resolve_for_path_falls_back_to_default(self):
        resolved = PhotoCollection.resolve_for_path('/some/totally/unregistered/path')
        self.assertEqual(resolved.pk, self.default_collection.pk)

    def test_new_photo_under_second_root_gets_that_collection(self):
        dest = os.path.join(self.second_root, 'new_photo.jpg')
        shutil.copy(self.FIXTURE_IMG, dest)

        create_image_file(dest)

        img = ImageFile.objects.get(filename=dest)
        self.assertEqual(img.directory.collection_id, self.second_collection.pk)
        self.assertIn(self.second_collection, img.collections.all())
        self.assertNotIn(self.default_collection, img.collections.all())

    def test_duplicate_found_under_second_collection_cross_links_the_original(self):
        # The real motivating case: a photo you already have (home =
        # default collection) gets re-discovered byte-for-byte while
        # scanning a newly-added collection -- it should become
        # queryable as "visible from" BOTH, not just its own home.
        original_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, original_dir, ignore_errors=True)
        original_path = os.path.join(original_dir, 'original.jpg')
        shutil.copy(self.FIXTURE_IMG, original_path)
        create_image_file(original_path)
        original = ImageFile.objects.get(filename=original_path)

        # Confirm its home really did resolve to the default collection
        # (original_dir was never registered as any collection's root).
        self.assertEqual(original.directory.collection_id, self.default_collection.pk)
        self.assertIn(self.default_collection, original.collections.all())
        self.assertNotIn(self.second_collection, original.collections.all())

        dup_path = os.path.join(self.second_root, 'duplicate_of_original.jpg')
        shutil.copy(self.FIXTURE_IMG, dup_path)
        create_image_file(dup_path)

        # No second ImageFile row -- same content, recorded as a
        # DuplicateFile instead.
        self.assertFalse(ImageFile.objects.filter(filename=dup_path).exists())
        dup = DuplicateFile.objects.get(filename=dup_path)
        self.assertEqual(dup.original_id, original.pk)
        self.assertEqual(dup.collection_id, self.second_collection.pk)

        # The ORIGINAL (whose home is still the default collection) is
        # now also visible from the second collection.
        original.refresh_from_db()
        self.assertIn(self.default_collection, original.collections.all())
        self.assertIn(self.second_collection, original.collections.all())

    def test_detection_floors_default_to_none(self):
        # The default collection (and any newly-created one with no
        # explicit floors) must impose no detection-time filtering --
        # None, not 0 or some other falsy-but-truthy-adjacent value,
        # since face_extract_encode.py's gate checks `is not None`.
        self.assertIsNone(self.default_collection.min_face_det_score)
        self.assertIsNone(self.default_collection.min_face_relative_size)
        fresh = PhotoCollection.objects.create(name='Fresh', root_directory='/tmp/fresh_collection_test')
        self.assertIsNone(fresh.min_face_det_score)
        self.assertIsNone(fresh.min_face_relative_size)

    def _attach_face(self, img, person):
        face = Face(
            source_image_file=img, declared_name=person, validated=True,
            box_left=1, box_top=1,
            box_right=min(40, img.width - 1), box_bottom=min(40, img.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_faces_for_a_person_are_queryable_by_collection_even_when_the_source_image_lives_elsewhere(self):
        # The exact behavior this whole feature was built around: a
        # photo whose HOME is the default collection, but that's also
        # visible from the second collection (via the duplicate
        # cross-link above), must show up when querying "this person's
        # faces, as seen from the second collection" -- via
        # source_image_file__collections, never directory__collection
        # alone (which would only ever see the home collection).
        from face_manager.models import Person

        person, _ = Person.objects.get_or_create(person_name='Collection Query Test Person')

        # A face whose source image's home is the default collection
        # AND is cross-linked into the second collection (same
        # mechanism as test_duplicate_found_under_second_collection_
        # cross_links_the_original above).
        home_path = os.path.join(tempfile.mkdtemp(), 'home.jpg')
        os.makedirs(os.path.dirname(home_path), exist_ok=True)
        shutil.copy(self.FIXTURE_IMG, home_path)
        create_image_file(home_path)
        cross_linked_img = ImageFile.objects.get(filename=home_path)
        cross_linked_img.collections.add(self.second_collection)
        cross_linked_face = self._attach_face(cross_linked_img, person)

        # A second face for the SAME person, genuinely only in the
        # default collection -- must NOT show up in the second
        # collection's view. A DIFFERENT fixture than home_path's --
        # otherwise it's pixel-identical and gets correctly recorded as
        # a duplicate of home_path instead of a genuinely new photo.
        other_fixture = os.path.join(
            os.path.dirname(self.FIXTURE_IMG), 'has_face_tags.jpg',
        )
        default_only_path = os.path.join(tempfile.mkdtemp(), 'default_only.jpg')
        os.makedirs(os.path.dirname(default_only_path), exist_ok=True)
        shutil.copy(other_fixture, default_only_path)
        create_image_file(default_only_path)
        default_only_img = ImageFile.objects.get(filename=default_only_path)
        default_only_face = self._attach_face(default_only_img, person)

        faces_in_second_collection = Face.objects.filter(
            declared_name=person, source_image_file__collections=self.second_collection,
        )
        self.assertIn(cross_linked_face, faces_in_second_collection)
        self.assertNotIn(default_only_face, faces_in_second_collection)

        # And person.face_declared still returns both, unfiltered --
        # Person/Face themselves are never partitioned by collection.
        self.assertEqual(set(person.face_declared.all()), {cross_linked_face, default_only_face})

    def test_load_images_into_db_walks_every_registered_collection_root(self):
        from unittest import mock
        from filepopulator.tasks import load_images_into_db

        with mock.patch('filepopulator.tasks.add_from_root_dir') as mock_add, \
             mock.patch('filepopulator.tasks.delete_removed_photos'):
            load_images_into_db()

        called_roots = {call.args[0] for call in mock_add.call_args_list}
        self.assertIn(self.default_collection.root_directory, called_roots)
        self.assertIn(self.second_collection.root_directory, called_roots)
        self.assertEqual(mock_add.call_count, PhotoCollection.objects.count())

    def test_load_videos_into_db_builds_roots_from_every_collection(self):
        from unittest import mock
        from filepopulator.tasks import load_videos_into_db

        self.second_collection.extra_video_roots = ['/videos/second_extra']
        self.second_collection.save()
        self.default_collection.extra_video_roots = ['/videos/default_extra_1', '/videos/default_extra_2']
        self.default_collection.save()

        with mock.patch('filepopulator.tasks.add_videos_from_root_dir') as mock_add, \
             mock.patch('filepopulator.tasks.delete_removed_videos'):
            load_videos_into_db()

        mock_add.assert_called_once()
        roots = set(mock_add.call_args[0][0])
        self.assertEqual(roots, {
            self.default_collection.root_directory, self.second_collection.root_directory,
            '/videos/second_extra', '/videos/default_extra_1', '/videos/default_extra_2',
        })
