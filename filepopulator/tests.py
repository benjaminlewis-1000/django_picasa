from django.test import TestCase
from django.core.exceptions import ValidationError
from django import forms
from django.conf import settings
import os
import binascii
from textwrap import wrap # for splitting string
import os
import shutil
import numpy as np
import imageio
import time

# Create your tests here.

from .models import ImageFile, Directory
# from .forms import ImageFileForm, DirectoryForm
from .scripts import create_image_file, add_from_root_dir, delete_removed_photos
# from .views import create_or_get_directory# , create_image_file, add_from_root_dir

import logging

logging.basicConfig(level=settings.LOG_LEVEL)

class ImageFileTests(TestCase):

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
        def test_with_prestrings(str1_pre, str2_pre, is_same_hash):
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
            file1 = os.path.join(self.tmp_valid_dir, 'outfile1.jpg')
            file2 = os.path.join(self.tmp_valid_dir, 'outfile2.jpg')
            imageio.imsave('outfile1.png', array1[:, :, (2, 1, 0)])
            imageio.imsave('outfile2.png', array2[:, :, (2, 1, 0)])
            shutil.move('outfile1.png', file1)
            shutil.move('outfile2.png', file2)

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
        test_with_prestrings(str1_pre, str2_pre, True)
        rand_1 = str(binascii.b2a_hex(os.urandom(500)))[2:-1]
        rand_2 = str(binascii.b2a_hex(os.urandom(500)))[2:-1]
        test_with_prestrings(rand_1, rand_2, False)

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

        logging.debug("All files in test_file_names is: {}".format(allFiles))

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
        # Test to see that adding a file again does not create a duplicate file.
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

    # def test_image_path_changes(self):

    #     # Are we testing when the same file is encountered elsewhere? 

    #     file_orig = self.goodFiles[0]
    #     create_image_file(file_orig)
    #     new_path = os.path.join(self.tmp_valid_dir, 'tmpmv.jpg')
    #     shutil.move(file_orig, new_path )
    #     create_image_file(new_path)

    #     orig_ref = ImageFile.objects.filter(filename = file_orig)
    #     new_ref = ImageFile.objects.filter(filename = new_path)

    #     shutil.move(new_path, file_orig)

    #     self.assertEqual(orig_ref.count(), 0)
    #     self.assertEqual(new_ref.count(), 1)


    # def test_same_picture_two_paths(self):
    #     src_file = self.goodFiles[0]
    #     path1 = os.path.join(self.tmp_valid_dir, 'tmp1.jpg')
    #     shutil.copy(src_file, path1)
    #     path2 = os.path.join(self.tmp_valid_dir, 'tmp2.jpg')
    #     shutil.copy(src_file, path2)
    #     create_image_file(path1)
    #     create_image_file(path2)


    #     first_item = ImageFile.objects.filter(filename=path1)
    #     self.assertEqual(len(first_item), 1)
    #     pixel_hash = first_item[0].pixel_hash
    #     second_item = ImageFile.objects.filter(filename=path2)
    #     self.assertEqual(len(second_item), 1)
    #     pixel_hash2 = second_item[0].pixel_hash

    #     self.assertNotEqual(path1, path2)
    #     self.assertEqual(pixel_hash, pixel_hash2)
    #     self.assertTrue(os.path.isfile(first_item[0].thumbnail.path))
    #     self.assertTrue(os.path.isfile(second_item[0].thumbnail.path))


    # def test_move_id_stay_same(self):

    #     for good in self.goodFiles:
    #         create_image_file(good)

    #     items = ImageFile.objects.all()
    #     item_files_init = [x.filename for x in items]

    #     src_file = self.goodFiles[0]
    #     src_item = ImageFile.objects.filter(filename=src_file)
    #     first_item = ImageFile.objects.filter(filename=src_file)
    #     ident1 = first_item[0].id
    #     ph = first_item[0].pixel_hash
    #     date_add = first_item[0].dateAdded

    #     path1 = os.path.join(self.tmp_valid_dir, 'tmp1.jpg')

    #     shutil.move(src_file, path1)
    #     # This path will effectively be removed from the database,
    #     # so in the final comparison of the initial files and the 
    #     # files that are in the database at the end, it should be
    #     # removed from this list. 
    #     item_files_init.remove(src_file)
    #     create_image_file(path1)
    #     second_item = ImageFile.objects.filter(filename=path1)
    #     ident2 = second_item[0].id
    #     print(ident2)
    #     self.assertEqual(ident1, ident2)
    #     self.assertNotEqual(second_item[0].dateAdded, date_add)
    #     date_add = second_item[0].dateAdded

    #     # Have two of same input -- how can I figure out which moved?
    #     path2 = os.path.join(self.tmp_valid_dir, 'tmp2.jpg')
    #     path3 = os.path.join(self.tmp_valid_dir, 'tmp3.jpg')
    #     shutil.copy(path1, path2)
    #     create_image_file(path2)
    #     item2 = ImageFile.objects.filter(filename=path2)
    #     date_add2 = item2[0].dateAdded
    #     self.assertNotEqual(date_add2, date_add)
    #     # 1 and 2 in database
    #     shutil.move(path2, path3)
    #     create_image_file(path3)
    #     item3 = ImageFile.objects.filter(filename=path3)
    #     date_add3 = item3[0].dateAdded
    #     self.assertNotEqual(date_add2, date_add3)

    #     # 3 should replace 2
    #     items = ImageFile.objects.all()
    #     item_files = [x.filename for x in items]
    #     # The initial items should only have path1 and path3 added to them now. 
    #     self.assertEqual(set(item_files), set(item_files_init + [path1, path3] ))

    #     # Should gracefully handle moving existing files
    #     shutil.move(self.goodFiles[3], path3)
    #     path3_id = ImageFile.objects.filter(filename=path3)
    #     path3_id = path3_id[0].id
    #     create_image_file(path3)
    #     path3_id_aft = ImageFile.objects.filter(filename=path3)
    #     path3_id_aft = path3_id_aft[0].id
    #     self.assertNotEqual(path3_id, path3_id_aft)

    # # Tested adding bad file names? Not there?


    # def test_metadata_gps(self):
    #     # raise NotImplementedError('GPS Test')
    #     # Make sure at least one image has GPS data. 

    #     for root, dirs, files in os.walk(self.tmp_valid_dir):
    #         for fname in files:
    #             create_image_file( os.path.join(root, fname) )

    #     db_objects = ImageFile.objects.all()

    #     gps_lats = []
    #     gps_lons = []

    #     for obj in db_objects:
    #         gps_lats.append(obj.gps_lat_decimal)
    #         gps_lons.append(obj.gps_lon_decimal)

    #     lat_valid_set = set(gps_lats)
    #     lon_valid_set = set(gps_lons)

    #     # Assert that there is more than the default value for fields with 
    #     # latitude and longitude. 
    #     self.assertTrue(len(lat_valid_set) > 1)
    #     self.assertTrue(len(lon_valid_set) > 1)

    #     # Same test a different way: -999 is the default value.
    #     lat_valid_set -= set([-999])
    #     lon_valid_set -= set([-999])

    #     self.assertTrue(len(lat_valid_set) > 0)
    #     self.assertTrue(len(lon_valid_set) > 0)

    # def test_delete_photos(self):

    #     for good in self.goodFiles:
    #         create_image_file(good)

    #     all_files = ImageFile.objects.all()
    #     before_len = len(all_files)

    #     num_to_remove = 3
    #     # Remove a couple files
    #     for n in range(num_to_remove):
    #         os.remove(self.goodFiles[n])

    #     delete_removed_photos()

    #     # Make sure that adding again doesn't create more.
    #     for good in self.goodFiles:
    #         create_image_file(good)

    #     updated_files = ImageFile.objects.all()
    #     self.assertTrue(before_len - len(updated_files) == num_to_remove)



    # def test_rotated_image_update(self):
    #     for good in self.goodFiles:
    #         create_image_file(good)

    #     file1 = self.goodFiles[0]
    #     file2 = self.goodFiles[1]
    #     create_image_file(file1)

    #     first_file_data = ImageFile.objects.filter(filename=file1)
    #     initial_date = first_file_data[0].dateAdded
    #     image_id = first_file_data[0].id
    #     path1 = os.path.join(self.tmp_valid_dir, 'tmp1.jpg')

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

    # def test_is_processed_reset(self):
    #     raise NotImplementedError('Is processed')

    # def test_make_request_of_image(self):
    #     raise NotImplementedError('Request image')
    #     # We're not able to get to the thumbnail via the admin page.
    #     # Look into that. 
    #     # May also need to look into the media root. 

    # def test_exif_metadata(self):
    #     raise NotImplementedError('Metadata test')

    # def test_rotation_dir(self):
    #     raise NotImplementedError('Need to test out the directory called "orientation".')


    # def test_bulk_add(self):
    #     add_from_root_dir(self.tmp_valid_dir)

    #     valid_files = []

    #     for root, dirs, files in os.walk(self.tmp_valid_dir):
    #         for f in files:
    #             cur_file = os.path.join(root, f)
    #             if cur_file.lower().endswith( ('.jpg', '.jpeg', ) ):
    #                 valid_files.append(cur_file)

    #     files_in_db = ImageFile.objects.all()
        
    #     raise NotImplementedError('Not finished with this test -- need to validate.')