from __future__ import absolute_import, unicode_literals

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings

from celery import shared_task, task
import time
import os
import random

import queue
import threading

from .scripts import populateFromImage, populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection
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
        settings.LOGGER.debug("Face adding locked!")
        return
    else:
        f = open(face_lockfile, 'w')
        f.close()

    try:
        settings.LOGGER.debug("Starting face extraction...")

        face_lockfile = settings.FACE_LOCKFILE

        server_conn = establish_multi_server_connection()

        num_servers = len(server_conn.server_ips)
        if num_servers == 0:
            settings.LOGGER.critical('No GPU servers found')
            return

        def worker(ip_num):
            print("Hi, I'm worker # ", ip_num)
            while True:
                print(f"Queue size is {img_q.qsize()}, worker is {ip_num}")
                if img_q.qsize() == 0:
                    print("all done!")
                    break
                else:
                    img = img_q.get()
                # if 
                is_ok = None
                rndDelay = random.randrange(0, 10) * 0.03
                time.sleep(rndDelay)
                qs = img_q.qsize()
                while is_ok is None:
                    try:
                        is_ok = server_conn.check_ip(ip_num)
                    except OSError:
                        timeDelay = random.randrange(0, 2)
                        print("Failed in worker ", ip_num)
                        time.sleep(timeDelay)
                        
                    if not is_ok[0]:
                        break
                    if not img.isProcessed:
                        try:
                            populateFromImageMultiGPU(img.filename, server_conn = server_conn, server_idx = ip_num, ip_checked=True)
                        except OSError:
                            break


        all_images = ImageFile.objects.all()
        img_q = queue.Queue()
        threads = []
        for img in all_images:
            if not img.isProcessed:
                img_q.put(img)

        for i in range(num_servers):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        # img_q.join()
        for t in threads:
            t.join()

        # for img in all_images:
        #     if not server_conn.check_ip():
        #         # Lost connection to server
        #         try:
        #             os.remove(face_lockfile)
        #         except FileNotFoundError:
        #             pass
        #         raise IOError("Lost connection to server.")
        #     if not img.isProcessed:
        #         # Then we need to schedule it to be processed.
        #         populateFromImage(img.filename, server_conn = server_conn)

        # num_servers = len(server_conn.server_ips)
        # if server_conn.server_ip is None:
        #     settings.LOGGER.critical('No GPU server found')
        #     return

        # all_images = ImageFile.objects.all()
        # for img in all_images:
        #     if not server_conn.check_ip():
        #         # Lost connection to server
        #         try:
        #             os.remove(face_lockfile)
        #         except FileNotFoundError:
        #             pass
        #         raise IOError("Lost connection to server.")
        #     if not img.isProcessed:
        #         # Then we need to schedule it to be processed.
        #         populateFromImage(img.filename, server_conn = server_conn)
    finally:
        try:
            os.remove(face_lockfile)
        except FileNotFoundError:
            pass