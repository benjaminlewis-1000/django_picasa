from __future__ import absolute_import, unicode_literals

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings

from celery import shared_task, task
import time
import os
import scripts
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

@task(ignore_result=True, name='filepopulator.populate_files_from_root')
def load_images_into_db():
    base_directory = settings.FILEPOPULATOR_SERVER_IMG_DIR
    fname = '/home/benjamin/filepopulate.txt'
    # logger.info("Write time")
    with open(fname, 'a') as fh:
        fh.write('The time is : {}\n'.format( time.time() ) )
    add_from_root_dir(base_directory)
    delete_removed_photos()


@task(name='filepopulator.update_dir_dates')
def update_dir_dates():
    print("Updates")
    update_dirs_datetime()

# # Some test tasks:
# @shared_task
# #(name='filepopulator.add')
# def add(x, y):
#     return x + y


# @shared_task#(name='filepopulator.mul')
# def mul(x, y):
#     return x * y


# @shared_task#(name='filepopulator.xsum')
# def xsum(numbers):
#     return sum(numbers)

# from datetime import datetime as dt
# @shared_task#(name='filepopulator.xsum')
# def log_time():
#     string = 'Current time: {}'.format(dt.now())
#     with open(os.path.join(settings.CODE_DIR, 'timelog.log'), 'a') as fh:
#         print(string, file=fh)

