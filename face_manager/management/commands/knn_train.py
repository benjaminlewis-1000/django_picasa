#! /usr/bin/env python

from filepopulator import models as file_models
from face_manager import models as face_models
from django.core.management.base import BaseCommand
from face_manager import knn_train

class Command(BaseCommand):
    
    def handle(self, *args, **options):
        
        knn_train.classify_unlabeled_faces()