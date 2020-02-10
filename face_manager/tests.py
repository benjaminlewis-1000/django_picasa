from django.test import TestCase
from django.core.exceptions import ValidationError
from django.conf import settings
from django.test import override_settings

import os
import cv2
import random
import shutil
import numpy as np

from filepopulator import scripts
from filepopulator.models import ImageFile
from .scripts import populateFromImage, placeInDatabase, establish_server_connection
from .models import Face, Person
import image_face_extractor
from image_face_extractor
# Create your tests here.

class FaceManageTests(TestCase):

    @override_settings(MEDIA_ROOT='/tmp')

    def setUp(self):
        settings.MEDIA_ROOT='/tmp'
        self.server_conn = establish_server_connection()

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
        face_data = populateFromImage(first_file.filename, self.server_conn)

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
        face_data1 = populateFromImage(first_file.filename, self.server_conn)
        pers = Person.objects.all()
        pers_len_first = len(pers)
        print(pers)
        second_file = ImageFile.objects.get(filename=self.same_faces_file)
        face_data2 = populateFromImage(second_file.filename, self.server_conn)

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
            face_data = populateFromImage(photo.filename, self.server_conn)
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

class FakeFacesTests(TestCase):

    @override_settings(MEDIA_ROOT='/tmp')
    def setUp(self):
        self.names = ['Alpha', 'Beta', 'Gamma', 'Charlie', 'Epsilon', 'Ragnar', 'Tiger', 'Wolf',\
                'Genni', 'Einstein', 'Bravo']
        settings.MEDIA_ROOT='/tmp'
        self.server_conn = establish_server_connection()

    def tearDown(self): 
        allFaces = Face.objects.all()
        for obj in allFaces:
            obj.delete()

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

        allObjects = ImageFile.objects.all()
        for obj in allObjects:
            obj.delete()

        shutil.rmtree(cls.tmp_valid_dir)

    def generate_fake_face_list(self):
        num_faces = random.randint(0, 5)
        name_idcs = random.sample(range(len(self.names)), num_faces)  

        face_list = []

        for i in range(num_faces):
            if random.randint(0, 1): 
                name = self.names[name_idcs[i]]
            else:
                name = None
            encoding = np.random.rand(128)
            face_w = random.randint(20, 40)
            face_h = random.randint(20, 40)
            face_thumbnail = np.random.rand(face_h, face_w, 3)
            square_face = np.random.rand(face_w, face_w, 3)
            box_center = (40, 40) 
            bounding_rect = face_extraction.Rectangle(face_h, face_w, centerX=box_center[0], centerY=box_center[1])
            face_rect = face_extraction.FaceRect(bounding_rect, face_thumbnail, 1, encoding=encoding, name=name, square_face=square_face)
            face_list.append(face_rect)

        return face_list
            
    def test_fake_faces(self):
        all_imgs = ImageFile.objects.all()
        settings.MEDIA_ROOT='/tmp'
    
        for img in all_imgs:
            face_list = self.generate_fake_face_list()
            self.assertFalse(img.isProcessed)
            placeInDatabase(img, face_list)
            
            # Get the faces attached to this image
            relevant_faces = Face.objects.filter(source_image_file = img)
            all_faces = Face.objects.all()

            for each_db in all_faces:
                self.assertTrue(os.path.exists(each_db.face_thumbnail.path))

            for each_input in face_list:
                name = each_input.name
                enc = each_input.encoding
                allclose = False
                for each_db in all_faces:
                    declared_name = each_db.declared_name
                    if declared_name is not None:
                        declared_name = declared_name.person_name 
                    if declared_name != name:
                        continue
                    # Else
                    # Allclose -- for names that are None, there may be multiple
                    # faces in an image with a "none" name. At least one of them
                    # must have an encoding that is close to this face's encoding.
                    allclose = allclose or  (np.allclose(enc, each_db.face_encoding))

                    self.assertEqual(each_db.poss_ident1, None)
                    self.assertEqual(each_db.poss_ident2, None)
                    self.assertEqual(each_db.poss_ident3, None)
                    self.assertEqual(each_db.poss_ident4, None)
                    self.assertEqual(each_db.poss_ident5, None)
            
                self.assertTrue(allclose)
            
        all_names = Person.objects.all()
        print(all_names)
        name_list = []
        for name in all_names:
            name_list.append(name)
        
        self.assertEqual(len(name_list), len(set(name_list)))
        self.assertTrue(len(name_list) <= len(self.names))


class MLTrainTests(TestCase):

    @override_settings(MEDIA_ROOT='/tmp')

    def setUp(self):
        settings.MEDIA_ROOT='/tmp'

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

        populateFromImage('/tmp/img_validation/naming/good/challenge dírectôry_with_repeats/1.JPG', self.server_conn)
        for photo in all_files:
            print(photo.filename)
            self.assertFalse(photo.isProcessed)
            face_data = populateFromImage(photo.filename, self.server_conn)
            this_face = ImageFile.objects.get(filename = photo.filename)
            self.assertTrue(this_face[0].isProcessed)


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
