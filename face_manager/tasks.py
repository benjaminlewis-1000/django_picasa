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


    all_images = ImageFile.objects.all()
    img_q = queue.Queue()
    # threads = []
    for img in all_images:
        if not img.isProcessed:
            img_q.put(img)

    face_lockfile = settings.FACE_LOCKFILE
    if os.path.exists(face_lockfile):
        settings.LOGGER.warning("Face adding locked!")
        return
    else:
        f = open(face_lockfile, 'w')
        f.close()
        
    def worker(ip_addr):
        print(f"Hi, I'm worker # {ip_addr}")
        while True:
            print(f"Queue size is {img_q.qsize()}, worker is {ip_addr}")
            if img_q.qsize() == 0:
                print("all done!")
                break
            else:
                img = img_q.get()

            is_ok = None
            rndDelay = random.randrange(5, 10) * 0.03
            time.sleep(rndDelay)
            qs = img_q.qsize()
            for _ in range(3):
                try:
                    is_ok = server_conn.check_ip(ip_addr)
                    break
                except OSError:
                    timeDelay = random.randrange(0, 2)
                    print(f"IS_OK failed in worker {ip_addr}")
                    time.sleep(timeDelay)

            if not is_ok: 
                break
                    
            if not img.isProcessed:
                try:
                    populateFromImageMultiGPU(img.filename, server_conn = server_conn, server_ip = ip_addr, ip_checked=True)
                except OSError:
                    print(f"IP {ip_addr} issue with populateFromImage")
                    break

    # Put everything in a while loop that polls for new workers
    # or dead workers every n (30 now) seconds

    # idx = 0
    running_threads = {}

    while True: 

        try:

            server_conn = establish_multi_server_connection()
            # print(f"Servers are {server_conn.server_ips}")

            num_servers = len(server_conn.server_ips)
            if num_servers == 0:
                settings.LOGGER.critical('No GPU servers found')
                return

            for i, ip in enumerate(running_threads.keys()):
                if not running_threads[ip].is_alive():
                    print(f"Killing thread {ip}")
                    running_threads[ip].join()

            for i, serv in enumerate(server_conn.server_ips):
                if serv not in running_threads.keys() or not running_threads[serv].is_alive():
                    t = threading.Thread(target=worker, args=(serv,))
                    running_threads[serv] = t
                    running_threads[serv].start()

            time.sleep(30)
            print(running_threads.keys())
            # print("Time slept")

        except:
            try:
                os.remove(face_lockfile)
            except FileNotFoundError:
                pass
            break