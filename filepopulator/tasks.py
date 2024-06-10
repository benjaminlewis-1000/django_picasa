from __future__ import absolute_import, unicode_literals

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings
from picasa import celery_app

from celery import shared_task
import time
import os
# import scripts
from .scripts import create_image_file, add_from_root_dir, delete_removed_photos, update_dirs_datetime

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

@shared_task(ignore_result=True, name='filepopulator.populate_files_from_root')
def load_images_into_db():

    
    i = celery_app.control.inspect()
    active_tasks = i.active()
    task_running = False
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.populate_files_from_root':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        # This task will be one, so looking for other tasks.
        settings.LOGGER.debug("Already running a task to load images, exiting.")
        return

        
    base_directory = settings.FILEPOPULATOR_SERVER_IMG_DIR
    # fname = os.path.join(os.environ['HOME'], 'filepopulate.txt')
    # logger.info("Write time")
    # with open(fname, 'a') as fh:
    #     fh.write('The time is : {}\n'.format( time.time() ) )
    print("Loading images into database...")
    add_from_root_dir(base_directory)
    print("Finished adding photos")
    delete_removed_photos()
    print("Done! Finished removing photos that aren't there.")
    update_dirs_datetime()


@shared_task(name='filepopulator.update_dir_dates')
def update_dir_dates():
    print("Updates")
    update_dirs_datetime()
