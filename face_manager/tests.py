from django.test import TestCase
from django.core.exceptions import ValidationError
from django.conf import settings

import os
import shutil

from filepopulator import scripts
from filepopulator.models import ImageFile
from .scripts import populateFromImage
# Create your tests here.

class FaceManageTests(TestCase):

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

        scripts.create_image_file(cls.face_file)
        scripts.add_from_root_dir(cls.tmp_valid_dir)

    def tearDown(self):
        pass

    def test_add_file(self):
        first_file = ImageFile.objects.filter(filename=self.face_file)[0]
        print("First file: ", first_file.filename)
        populateFromImage(first_file.filename)
        pass

    def test_that(self):
        pass