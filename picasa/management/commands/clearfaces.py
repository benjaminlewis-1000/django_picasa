#! /usr/bin/env python

from filepopulator import models as file_models
from face_manager import models as face_models
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    def handle(self, *args, **options):
            
        proc_files = file_models.ImageFile.filter(isProcessed=True)
        print(len(proc_files))
        proc_files = file_models.ImageFile.filter(isProcessed=False)
        print(len(proc_files))

        yn = input("Warning! You are about to delete all faces. Would you like to continue? (y/N): ")
        if yn.lower() != 'n': 
            print("Not deleting.")
            return
            
        print("Deleting...")
        faces = face_models.Face.objects.all()
        people = face_models.Person.objects.all()

        for f in faces:
            f.source_image_file.isProcessed = False
            f.source_image_file.save()
            f.delete()

        for p in people:
            p.delete()