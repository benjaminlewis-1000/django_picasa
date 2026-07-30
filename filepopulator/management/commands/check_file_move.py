#! /usr/bin/env python

from filepopulator import models
from filepopulator import tasks
from filepopulator.models import ImageFile, Directory
from filepopulator import scripts
from face_manager import models as face_models
from django.core.management.base import BaseCommand
import shutil
import os

class Command(BaseCommand):
    def handle(self, *args, **options):
        # aa = ImageFile.objects.first()
        # print(aa)
        # tasks.load_images_into_db()
        # file_path = '/photos/Pictures_In_Progress/2019/Baltimore Trip/DSC_1171.JPG'
        file_path = '/code/lincoln.jpeg' # Pictures_In_Progress/syncthing/aggregated/IMG_5563_20230910_200439.jpg'
        new_path = '/code/lincoln2.jpeg'

        partial = '20260412_084313.jpg'
        aa = ImageFile.objects.filter(filename__contains = partial)[0]
        print(aa)
        print(aa, aa.id, aa.pixel_hash, aa.file_hash, aa.isProcessed)
        existing_faces = face_models.Face.objects.filter(source_image_file=aa)
        print(existing_faces)
        

        if not os.path.exists(file_path):
            shutil.move(new_path, file_path)
        # file_path = '/photos/Pictures_In_Progress/2024/Family Texts/IMG_6658_20230714_131728.jpg'
        print("Ready to create")
        scripts.create_image_file(file_path)

        aa = ImageFile.objects.filter(filename=file_path)[0]
        print(aa, aa.id, aa.pixel_hash, aa.file_hash, aa.isProcessed)
        aa.isProcessed = True
        aa.save()

        shutil.move(file_path, new_path)
        scripts.create_image_file(new_path)
        aa = ImageFile.objects.filter(filename=new_path)[0]
        print(aa, aa.id, aa.pixel_hash, aa.file_hash, aa.isProcessed)

        print("Script done")


#        scripts.check_file_mods()
        # IMG_5563_20230910_200439.jpg
        # IMG_5563_20230910_200439.jpg
        # IMG_5563_20230910_200439.jpg
