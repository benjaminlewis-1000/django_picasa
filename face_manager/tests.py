from django.test import TestCase
from django.core.exceptions import ValidationError
from django.conf import settings

import os
import cv2
import shutil
import numpy as np

from filepopulator import scripts
from filepopulator.models import ImageFile
from .scripts import populateFromImage
from .models import Face, Person
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
        cls.same_faces_file = os.path.join(cls.tmp_valid_dir, 'has_same_faces.jpg')

        scripts.create_image_file(cls.face_file)
        scripts.add_from_root_dir(cls.tmp_valid_dir)

    @classmethod
    def tearDownClass(cls):

        print("Teardown class")
        allObjects = ImageFile.objects.all()
        for obj in allObjects:
            obj.delete()

        shutil.rmtree(cls.tmp_valid_dir)

    def tearDown(self):
        allFaces = Face.objects.all()
        for face in allFaces:
            face.delete()

        allPersons = Person.objects.all()
        for per in allPersons:
            per.delete()

    def test_add_file(self):
        first_file = ImageFile.objects.get(filename=self.face_file)
        face_data = populateFromImage(first_file.filename)

        faces = Face.objects.all()
        self.assertEqual(len(faces), len(face_data))
        for f in range(len(faces)):
            save_face = faces[f]
            ext_face = face_data[f]
            save_name = save_face.declared_name.person_name
            self.assertTrue(os.path.isfile(save_face.face_thumbnail.path))
            save_thumbnail = cv2.imread(save_face.face_thumbnail.path)
            self.assertEqual(save_thumbnail.shape, ext_face.square_face.shape)
            self.assertEqual(save_name, ext_face.name)
            save_enc = save_face.face_encoding
            ext_enc = np.array(ext_face.encoding)
            enc_diff = np.abs(save_enc - ext_enc)

            self.assertTrue(np.all(enc_diff < 1e-10))

        pass

    def test_same_faces(self):
        first_file = ImageFile.objects.get(filename=self.face_file)
        face_data1 = populateFromImage(first_file.filename)
        pers = Person.objects.all()
        pers_len_first = len(pers)
        print(pers)
        second_file = ImageFile.objects.get(filename=self.same_faces_file)
        face_data2 = populateFromImage(second_file.filename)

        faces = Face.objects.all()
        pers = Person.objects.all()

        self.assertEqual(len(pers), pers_len_first)
        print(pers)
         
        pass

    def test_faceadd_all(self):

        face_set = set()
        all_files = ImageFile.objects.all()
        for photo in all_files:
            print(photo.filename)
            face_data = populateFromImage(photo.filename)
            if face_data is not None:
                print(face_data)
                for f in face_data:
                    name = f.name
                    face_set = face_set.union(set([name]))

        persons = Person.objects.all()
        self.assertEqual(len(persons), len(face_set))
        for p in persons:
            self.assertTrue(p.person_name in face_set)

    def test_addtwice(self):
        pass

class MLTrainTests(TestCase):

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

        scripts.add_from_root_dir(cls.tmp_valid_dir)

        all_files = ImageFile.objects.all()

        populateFromImage('/tmp/img_validation/naming/good/challenge dírectôry_with_repeats/1.JPG')
        for photo in all_files:
            print(photo.filename)
            face_data = populateFromImage(photo.filename)

    @classmethod
    def tearDownClass(cls):

        allFaces = Face.objects.all()
        for face in allFaces:
            face.delete()

        allPersons = Person.objects.all()
        for per in allPersons:
            per.delete()

        allObjects = ImageFile.objects.all()
        for obj in allObjects:
            obj.delete()

        shutil.rmtree(cls.tmp_valid_dir)

    def test_setup(self):
        pass