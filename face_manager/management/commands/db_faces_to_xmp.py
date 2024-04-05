#! /usr/bin/env python

# A command to try debugging images that are not currently being correctly processed by the face extraction.
# Uses the database to find unprocessed images, then sends them through. 

from django.conf import settings
from django.core.management.base import BaseCommand

from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from face_manager.scripts import populateFromImage, populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection

from face_manager import face_to_xmp

class Command(BaseCommand):
    
    def handle(self, *args, **options):
        
        face_to_xmp.main()
