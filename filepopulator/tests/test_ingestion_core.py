"""filepopulator tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test filepopulator.tests.test_ingestion_core
"""
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

from django.core.management import call_command
from django.core.management.base import CommandError

from filepopulator.models import ImageFile, Directory, DuplicateFile, FailedImageFile, PhotoCollection, guess_date_from_filename, VideoFile, FailedVideoFile, DuplicateVideoFile
from filepopulator.scripts import create_image_file, add_from_root_dir, delete_removed_photos, update_dirs_datetime, check_file_mods, delete_old_thumbnails
from filepopulator.video_scripts import create_video_file, add_videos_from_root_dir, delete_removed_videos, _parse_exif_date, compute_video_content_hash
from face_manager.models import Face, get_default_blank_person
from common.open_img_oriented import apply_exif_orientation
from filepopulator import video_scripts as _video_scripts_module
from filepopulator.test_video_cache import cached_run_ffprobe, cached_run_exiftool

from filepopulator.tests import _tiny_jpeg_bytes


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
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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
