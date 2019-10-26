from django.test import TestCase

# Create your tests here.

from .models import ImageModel, Photo

from django.conf import settings
import logging
import shutil
import os

logging.basicConfig(level=settings.LOG_LEVEL)

class PhotoTests(TestCase):

    def setUp(self):

        self.validation_dir = settings.FILEPOPULATOR_VAL_DIRECTORY # '/validation_imgs'
        # self.val_train = os.path.join(self.val_img_prefix, 'train')
        # self.val_test = os.path.join(self.val_img_prefix, 'test')

        assert os.path.isdir(self.validation_dir), 'Validation directory in ImageFileTests does not exist.'
        # assert os.path.isdir(self.val_train), 'val_train directory in ImageFileTests does not exist.'
        # assert os.path.isdir(self.val_test), 'val_test directory in ImageFileTests does not exist.'

        self.tmp_valid_dir = '/home/benjamin/git_repos/local_picasa/media/tmp'

        if os.path.exists(self.tmp_valid_dir):
            shutil.rmtree(self.tmp_valid_dir)

        shutil.copytree(self.validation_dir, self.tmp_valid_dir)

        self.test_dir = os.path.join(self.tmp_valid_dir, 'naming')
        self.good_dir = os.path.join(self.test_dir, 'good')
        self.bad_dir = os.path.join(self.test_dir, 'bad')

        self.goodFiles = []
        self.badFiles = []

        for root, dirs, files in os.walk(self.good_dir):
            for fname in files:
                self.goodFiles.append(os.path.join(root, fname) )

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
        # Clean up the thumbnails
        allObjects = Photo.objects.all()
        for obj in allObjects:
            obj.thumbnail.delete()
            obj.delete()

        shutil.rmtree(self.tmp_valid_dir)


    def test_add(self):
    	print(self.goodFiles[0])
    	p = Photo(photo_name=self.goodFiles[0])
    	p.save() # make_thumbnail()