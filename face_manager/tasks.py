from __future__ import absolute_import, unicode_literals

# from .scripts import populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection
from .models import Person, Face
from assign_faces import faceAssigner
from celery import shared_task
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Q
from django.utils.crypto import get_random_string
from face_extract_encode import FaceExtractor
from filepopulator.models import ImageFile
from picasa import celery_app
import os
import queue
import random
import threading
import time
import traceback


# from image_face_extractor import reencoder, ip_finder 

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

@shared_task(ignore_result=True, name='face_manager.face_extraction')
def process_faces():
    starttime = time.time()

    try:
        settings.LOGGER.debug("Starting face extraction...")

        i = celery_app.control.inspect()
        active_tasks = i.active()
        task_running = False
        num_this_task_running = 0
        for k in active_tasks.keys():
            tasks = active_tasks[k]
            if len(tasks) != 0:
                for tt in tasks:
                    if tt['name'] == 'face_manager.face_extraction':
                        num_this_task_running += 1

        if num_this_task_running > 1:
            # This task will be one, so looking for other tasks.
            settings.LOGGER.debug("Face file is locked, exiting.")
            settings.LOGGER.warning("Face adding locked!")
            return

        unprocessed_imgs = ImageFile.objects.filter(isProcessed=False).all()
        unprocessed_count = ImageFile.objects.filter(isProcessed=False).count()

        if unprocessed_count == 0:
            settings.LOGGER.debug("No images to extract! Exiting." )
            return

        extractor = FaceExtractor()
        extractor.find_and_encode_faces()

    except:

        settings.LOGGER.debug("Ending face adding task")
        
@shared_task(ignore_result=True, name='face_manager.reencode')
def reencode_missing_faces():
    i = celery_app.control.inspect()
    active_tasks = i.active()
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.reencode':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        settings.LOGGER.debug("Reencode is locked, exiting.")
        settings.LOGGER.warning("Reencode locked!")
        return

    try:
        extractor = FaceExtractor()
        extractor.reencode_missing_faces()
    except:
        settings.LOGGER.debug("Ending reencode task")

@shared_task(ignore_result=True, name='face_manager.assign_faces')
def thistask(redo_all=False):

    i = celery_app.control.inspect()
    active_tasks = i.active()
    task_running = False
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.assign_faces':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        # This task will be one, so looking for other tasks.
        settings.LOGGER.debug("Classification is locked, exiting.")
        settings.LOGGER.warning("Classification locked!")
        return

    try:
        classer = faceAssigner()
        classer.execute(redo_all)
    except:
        print("Image classification failed!")
        
@shared_task(ignore_result=True, name='face_manager.reload_classifier_pkl')
def classifier_pkl_file_reload():
    classer = faceAssigner()
    classer.load_encodings(reload_pkl_file = True)

@shared_task(ignore_result=True, name='face_manager.api_bulk_operation')
def api_bulk_operation(input_dict: dict):
    print("Executing a bulk operation task", input_dict)

@shared_task(ignore_result=True, name='face_manager.set_face_counts')
def reset_task():
    people = Person.objects.all()
    for p in people:
        p.num_faces = p.face_declared.count()
        p.num_possibilities = p.face_poss1.count() # + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
        p.num_unverified_faces = p.face_declared.filter(validated=False).count()
        p.save()
