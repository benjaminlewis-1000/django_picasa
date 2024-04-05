#! /usr/bin/env python

# A command to try debugging images that are not currently being correctly processed by the face extraction.
# Uses the database to find unprocessed images, then sends them through. 

from django.conf import settings
from django.core.management.base import BaseCommand

from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from face_manager.scripts import populateFromImage, populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection


class Command(BaseCommand):
    
    def handle(self, *args, **options):

        # Get the list of files that are not processed

        all_images = ImageFile.objects.filter(isProcessed=False)
        print(len(all_images))

        server_conn = establish_multi_server_connection()
        
        print(server_conn.server_ips, dir(server_conn))
        for img in all_images:
            populateFromImageMultiGPU(img.filename,  server_conn = server_conn, server_ip = server_conn.server_ips[1])