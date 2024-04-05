#! /usr/bin/env python

from filepopulator import models as file_models
from face_manager import models as face_models
from face_manager.models import Person, Face
from django.core.management.base import BaseCommand

# Resets unknown face counts

class Command(BaseCommand):
    
    def handle(self, *args, **options):
            
        people = Person.objects.all()

        for p in people:
            print(p)
            p.num_faces = p.face_declared.count()
            p.num_possibilities = p.face_poss1.count() + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
            p.num_unverified_faces = p.face_declared.filter(validated=False).count()
            p.save()
