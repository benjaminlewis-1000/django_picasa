#! /usr/bin/env python

from filepopulator import models
from face_manager import models as face_models
from django.core.management.base import BaseCommand



class Command(BaseCommand):
    def handle(self, *args, **options):

        yn = input("Warning! You are about to delete all data in your database. Would you like to continue? (y/N): ")
        if yn.lower() != 'y': 
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
            
        models.ImageFile.objects.all().delete()
        models.Directory.objects.all().delete()