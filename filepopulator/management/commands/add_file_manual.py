#! /usr/bin/env python

from filepopulator import models
from filepopulator import tasks
from filepopulator.models import ImageFile, Directory
from filepopulator import scripts
from face_manager import models as face_models
from django.core.management.base import BaseCommand



class Command(BaseCommand):
    def handle(self, *args, **options):
        print("Adding files")
        aa = ImageFile.objects.first()
        # print(aa)
        tasks.load_images_into_db()
        # file_path = '/photos/Pictures_In_Progress/2019/Baltimore Trip/DSC_1171.JPG'
        file_path = '/photos/Pictures_In_Progress/syncthing/aggregated/IMG_5563_20230910_200439.jpg'
        # file_path = '/photos/Pictures_In_Progress/2024/Family Texts/IMG_6658_20230714_131728.jpg'
#        scripts.create_image_file(file_path)


#        scripts.check_file_mods()
        # IMG_5563_20230910_200439.jpg
        # IMG_5563_20230910_200439.jpg
        # IMG_5563_20230910_200439.jpg
