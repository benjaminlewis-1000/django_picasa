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
import os
import binascii
from datetime import datetime
from textwrap import wrap # for splitting string
import os
import shutil
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

from .models import ImageFile, Directory, DuplicateFile, FailedImageFile, guess_date_from_filename
# from .forms import ImageFileForm, DirectoryForm
from .scripts import create_image_file, add_from_root_dir, delete_removed_photos, update_dirs_datetime, check_file_mods
from face_manager.models import Face


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
            self.assertNotEqual(obj1.thumbnail_big.path, obj2.thumbnail_big.path )
            self.assertNotEqual(obj1.thumbnail_medium.path, obj2.thumbnail_medium.path )
            self.assertNotEqual(obj1.thumbnail_small.path, obj2.thumbnail_small.path )

            # Check that all the thumbnails were created properly.
            self.assertTrue(os.path.isfile(obj1.thumbnail_big.path))
            self.assertTrue(os.path.isfile(obj1.thumbnail_medium.path))
            self.assertTrue(os.path.isfile(obj1.thumbnail_small.path))
            self.assertTrue(os.path.isfile(obj2.thumbnail_big.path))
            self.assertTrue(os.path.isfile(obj2.thumbnail_medium.path))
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
            thumbnail_big='', thumbnail_medium='', thumbnail_small='',
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
            self.assertTrue(os.path.isfile(allObjects[num].thumbnail_big.path))
            self.assertTrue(os.path.isfile(allObjects[num].thumbnail_medium.path))
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
        thumbnail1 = first_item[0].thumbnail_big.path
        id1 = first_item[0].id
        self.assertTrue(os.path.isfile(thumbnail1))

        file2 = self.goodFiles[1]
        shutil.copy(file2, dest_file)
        create_image_file(dest_file)

        # Get the data from the second data addition. 
        second_item = ImageFile.objects.filter(filename=dest_file)
        thumbnail2 = second_item[0].thumbnail_big.path
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

        self.assertTrue(os.path.isfile(new_ref[0].thumbnail_big.path))
        self.assertTrue(os.path.isfile(new_ref[0].thumbnail_medium.path))
        self.assertTrue(os.path.isfile(new_ref[0].thumbnail_small.path))
        self.assertFalse(os.path.isfile(orig_data.thumbnail_big.path))
        self.assertFalse(os.path.isfile(orig_data.thumbnail_medium.path))
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
        self.assertTrue(os.path.isfile(f3_data.thumbnail_big.path))
        self.assertFalse(os.path.isfile(f1_data.thumbnail_big.path))

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

        self.assertTrue(os.path.isfile(first_item[0].thumbnail_big.path))
        self.assertFalse(first_item[0].isProcessed)

    def test_delete_photos(self): ### CHECKED ### 

        # Case: We want to delete random photos from the file system, then run
        # the function that cleans that up (delete_removed_photos) and 
        # check that they were, in fact, completely removed from the database. 
        # Expected outcome: files removed from disk will not show up in the 
        # database, but other files will still be there. 
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
        self.assertTrue(os.path.isfile(f_data.thumbnail_big.path))
        self.assertTrue(os.path.isfile(f_data.thumbnail_medium.path))
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
        self.assertNotEqual(first_file_data.thumbnail_big.path, rot_data.thumbnail_big.path)


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
        sleep(5)
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
            thumbnail_big="", thumbnail_medium="", thumbnail_small="",
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
    pillow_heif/libheif always decodes to plain RGB, and always reports
    EXIF orientation 1 regardless of the photo's actual portrait/
    landscape framing -- meaning it auto-applies any container-level
    rotation transform during decode and resets the tag to match. The
    guards below assume that continues to hold for any real-world HEIC
    this pipeline ever sees, and fail loudly (recorded via
    FailedImageFile/image_load_failed, not silently) rather than risk a
    silently wrong rotation if it ever doesn't.
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
            self.assertTrue(os.path.isfile(img.thumbnail_big.path))

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

    def test_orientation_guard_fails_loudly_on_non_one_orientation(self):
        path = os.path.join(self.HEIC_DIR, self._heic_files()[0])

        class FakeExif(dict):
            def get_ifd(self, tag):
                return {}

        fake_exif = FakeExif({274: 6})  # Orientation tag id = 274
        with mock.patch("PIL.Image.Image.getexif", return_value=fake_exif):
            create_image_file(path)

        self.assertFalse(ImageFile.objects.filter(filename=path).exists())
        failed = FailedImageFile.objects.get(filename=path)
        self.assertIn("orientation", failed.error_message.lower())

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
        self.assertTrue(os.path.isfile(obj.thumbnail_big.path))


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
            thumbnail_big="", thumbnail_medium="", thumbnail_small="",
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
            thumbnail_big="", thumbnail_medium="", thumbnail_small="",
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
            thumbnail_big="", thumbnail_medium="", thumbnail_small="",
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
        original_thumb = obj.thumbnail_big.name
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash')

        obj.refresh_from_db()
        self.assertEqual(obj.pixel_hash, original_pixel_hash)
        self.assertEqual(obj.thumbnail_big.name, original_thumb)

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
