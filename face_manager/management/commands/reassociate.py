#! /usr/bin/env python

import image_face_extractor.face_extraction as fe
from filepopulator import models as file_models
from face_manager import models as face_models
from django.core.management.base import BaseCommand
from face_manager.models import Person, Face
from django.conf import settings
import pickle
from django.db.models import Q
import numpy as np
import time

class Command(BaseCommand):
    
    def handle(self, *args, **options):
        print("ps")
        # n = Person.objects.filter(person_name="Nathaniel Lewis")[0]
        criterion0 = ~Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        # # criterion1 = Q(poss_ident1= n)
        # # criterion2 = Q(poss_ident2= n)
        # # criterion3 = Q(poss_ident3= n)
        # # criterion4 = Q(poss_ident4= n)
        # # criterion5 = Q(poss_ident5= n)

        possibilities = Face.objects.filter(criterion0) # & (criterion1 | criterion2 | criterion3 | criterion4 | criterion5))
        print(possibilities.count())

        for face in possibilities.iterator():
            face.poss_ident1 = face.poss_ident2 = face.poss_ident3 = face.poss_ident4 = face.poss_ident5 = None
            face.save()
            print(face)
            # exit()
        #     print(face.id, "-----------")
        #     try:
        #         # print(face, face.id)
        #         sf = face.source_image_file
        #         # print(box)

        #         face_img = face.source_image_file.filename

        #         # Make a numpy array the size of the image
        #         intersect_img = np.zeros((sf.height, sf.width)).astype(bool)
        #         intersect_img[face.box_top:face.box_bottom, face.box_left:face.box_right] = 1

        #         main_face_size = len(np.where(intersect_img)[0])
        #         # print(main_face_size)
        #         # print(face_img, dir(face_img))
                    
        #         data = fe.Get_XMP_Faces(face_img)
        #         valid, people = data

        #         for p in people: 
        #             # print(p, dir(p))
        #             print(p)
        #             s = time.time()
        #             bounds = p['bounding_rectangle']
        #             # print(bounds, dir(bounds))
        #             cmp_img = np.zeros((sf.height, sf.width)).astype(bool)
        #             cmp_img[bounds.top:bounds.bottom, bounds.left:bounds.right] = 1
        #             written_face_size = len(np.where(cmp_img)[0])
        #             # print(written_face_size)

        #             sum_img = cmp_img & intersect_img
        #             intersect_size = len(np.where(sum_img)[0])
        #             total_img = cmp_img | intersect_img
        #             union_size = len(np.where(total_img)[0])
        #             # print(intersect_size, union_size)
        #             if intersect_size / union_size > 0.3:
        #                 # It's a match! 
        #                 # Find the person matching the person
        #                 person_name = p['Name']
        #                 person_obj = Person.objects.filter(person_name=person_name)[0]
        #                 print(person_obj)
        #                 print(face.declared_name)
        #                 face.declared_name = person_obj
        #                 face.validated = False
        #                 face.poss_ident1 = face.poss_ident2 = face.poss_ident3 = face.poss_ident4 = face.poss_ident5 = None
        #                 face.save()
        #     except:
        #         pass
