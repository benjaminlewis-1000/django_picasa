#! /usr/bin/env python

from filepopulator import models as file_models
from face_manager import models as face_models
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    
    def handle(self, *args, **options):
            
        yn = input("Warning! You are about to delete all faces. Would you like to continue? (y/N): ")
        if yn.lower() != 'y': 
            print("Not deleting.")
            return
            
        print("Deleting...")
        faces = face_models.Face.objects.all()
        people = face_models.Person.objects.all()

        for f in faces:
            print(f.source_image_file)
            if f.source_image_file.isProcessed:
                f.source_image_file.isProcessed = False
                super(file_models.ImageFile, f.source_image_file).save()
            f.delete()

        for p in people:
            p.delete()

        proc_files = file_models.ImageFile.objects.filter(isProcessed=True)
        for f in proc_files:
            f.isProcessed = False
            super(file_models.ImageFile, f).save()

        proc_files = file_models.ImageFile.objects.filter(isProcessed=True)
        assert len(proc_files) == 0