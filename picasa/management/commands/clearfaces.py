#! /usr/bin/env python

from filepopulator import models as file_models
from face_manager import models as face_models
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    def handle(self, *args, **options):
        faces = face_models.Face.objects.all()
        people = face_models.Person.objects.all()

        for f in faces:
            f.source_image_file.isProcessed = False
            f.source_image_file.save()
            f.delete()

        for p in people:
            p.delete()
            