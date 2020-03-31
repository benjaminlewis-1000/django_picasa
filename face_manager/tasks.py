from __future__ import absolute_import, unicode_literals

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings

from celery import shared_task, task
import time
import os

from .scripts import populateFromImage, establish_server_connection
from .models import Person, Face
from filepopulator.models import ImageFile

if not settings.configured:
    settings.configure()

# If you don’t care about the results of a task, be sure to set the ignore_result option,
# as storing results wastes time and resources.
# Shared tasks are to make apps without any concrete app instance. 
# Tasks depend on the app I guess.
# Tasks can be autodiscovered by placing the app in the projec, then creating 
# celery.py in the <PROJECT> directory and putting the following:
# app.config_from_object('django.conf:settings', namespace='CELERY')
# app.autodiscover_tasks() 

@task(ignore_result=True, name='face_manager.face_extraction')
def process_faces():
    settings.LOGGER.debug("Starting face extraction...")

    face_lockfile = settings.FACE_LOCKFILE

    if os.path.exists(face_lockfile):
        print("Face adding locked!")
        return
    else:
        f = open(face_lockfile, 'w')
        f.close()

    try:
        server_conn = establish_server_connection()
        if server_conn.server_ip is None:
            settings.LOGGER.critical('No GPU server found')
            return

        all_images = ImageFile.objects.all()
        for img in all_images:
            if not img.isProcessed:
                # Then we need to schedule it to be processed.
                populateFromImage(img.filename, server_conn = server_conn)
    finally:
            try:
                os.remove(face_lockfile)
            except FileNotFoundError:
                pass