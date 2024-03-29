from __future__ import absolute_import, unicode_literals

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings

from django.db.models import Q
from celery import shared_task
import time
import os
import random

import queue
import threading

from .scripts import populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection
from .models import Person, Face
from filepopulator.models import ImageFile

from image_face_extractor import reencoder, ip_finder 

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

@shared_task(ignore_result=False, name='face_manager.reencode')
def reencode_face():
    print("Reencoding!")
    settings.LOGGER.warning("Reencoding!")

    start_time = time.time()

    criterion_ign = ~Q(declared_name__person_name__in=['.ignore', '.realignore', '.jessicatodo', '.cutler_tbd'] )
    criterion_2 = Q(reencoded=False)

    all_faces = Face.objects.filter(criterion_ign&criterion_2).order_by('?')

    num_faces = all_faces.count()
    print(f"There are {num_faces} faces to reencode.")

    client_ip = ip_finder.server_finder()
    print("Found client IP")

    iter_num = 1
    for face in all_faces.iterator():
        print(f'{iter_num}/{num_faces}, {iter_num / num_faces * 100:.4f}%')
        iter_num += 1
        if face.reencoded:
            continue

        face_location = [(face.box_top, face.box_right, face.box_bottom, face.box_right)]
        source_image_file = face.source_image_file.filename

        encoding = reencoder.face_encoding_client(source_image_file, face_location, client_ip)
        # print(source_image_file, face_location, encoding)

        face.encoding = encoding
        face.reencoded = True
        face.save()

        # Kill after 5 minutes
        if time.time() - start_time > 15 * 60:
            return

@shared_task(ignore_result=True, name='face_manager.face_extraction')
def process_faces():
    starttime = time.time()

    print('Starting face extraction...')
    settings.LOGGER.debug("Starting face extraction...")


    # Set up a lock on this function so that it doesn't have more than one running at once. 
    face_lockfile = settings.FACE_LOCKFILE
    print(f"Face lockfile is {face_lockfile}")
    if os.path.exists(face_lockfile):
        print("Face file is locked, exiting.")
        settings.LOGGER.warning("Face adding locked!")
        return
    else:
        f = open(face_lockfile, 'w')
        f.close()
        
    # all_images = ImageFile.objects.all()
    unprocessed_imgs = ImageFile.objects.filter(isProcessed=False).all()
    unprocessed_count =  ImageFile.objects.filter(isProcessed=False).count()

    if unprocessed_count == 0:
        print("No images to extract! Exiting." )
        try:
            os.remove(face_lockfile)
        except FileNotFoundError:
            pass
        return

    img_q = queue.Queue()

    for img in unprocessed_imgs:
        if not img.isProcessed:
            if os.path.isfile(img.filename):
                img_q.put(img)



    print("Going to start workers...")
        
    # Develop a threaded function. 

    class process_thread(threading.Thread):

        def __init__(self, ip_addr, *args, **kwargs):
            super(process_thread, self).__init__(*args, **kwargs)
            self._stopper = threading.Event()

            self.ip_addr = ip_addr

        def stop(self):
            self._stopper.set()
            
        def stopped(self): 
            return self._stopper.isSet() 

        def run(self):
            print(f"Hi, I'm worker # {self.ip_addr}")
            while True:
                if self.stopped():
                    return 
                    
                print(f"Queue size is {img_q.qsize()}, worker is {self.ip_addr}")
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
                        is_ok = server_conn.check_ip(self.ip_addr)
                        break
                    except OSError:
                        timeDelay = random.randrange(0, 2)
                        print(f"IS_OK failed in worker {self.ip_addr}")
                        time.sleep(timeDelay)

                if not is_ok: 
                    break
                        
                if not img.isProcessed:
                    try:
                        populateFromImageMultiGPU(img, server_conn = server_conn, server_ip = self.ip_addr, ip_checked=True)
                    except OSError as ose:
                        print(f"IP {self.ip_addr} issue with populateFromImage, filename {img.filename}, img id {img.id}. Error: {ose}")
                        break

    # Put everything in a while loop that polls for new workers
    # or dead workers every n (30 now) seconds

    running_threads = {}

    # Kill the process once every 6 hours if it hasn't stopped already.

    start_time = time.time()
            
    print("Starting to try...")

    def end_threads(running_threads):
        print("End the threads")
        keys = list(running_threads.keys())
        for key in keys:
            # print(running_threads[key])
            running_threads[key].stop()
            # running_threads[key].join()
            running_threads.pop(key)


    while True: 
        print("True loop")
        cur_time = time.time()
        
        if cur_time - start_time > (45 * 60): # 45 minutes
            print("KILLING Current time")
            end_threads(running_threads)
            break

        if img_q.qsize() == 0:
            end_threads(running_threads)            
            break

        try:
            server_conn = establish_multi_server_connection()
            # print(f"Servers are {server_conn.server_ips}")

            num_servers = len(server_conn.server_ips)
            # print(f"Number of servers is {num_servers}")
            if num_servers == 0:
                settings.LOGGER.critical('No GPU servers found')
                return

            for i, ip in enumerate(running_threads.keys()):
                if not running_threads[ip].is_alive():
                    print(f"Killing thread {ip}")
                    running_threads[ip].stop()
                    running_threads[ip].join()

            for i, serv in enumerate(server_conn.server_ips):
                if serv not in running_threads.keys() or not running_threads[serv].is_alive():
                    t = process_thread(ip_addr = serv) # threading.Thread(target=worker, args=(serv,))
                    # t.start()
                    print(i, serv)
                    running_threads[serv] = t
                    running_threads[serv].start()

            time.sleep(30)
            print(running_threads.keys())
            # print("Time slept")

        except Exception as e:
            print(f"Exception in face extraction found : {e}")
            try:
                os.remove(face_lockfile)
                end_threads(running_threads)

            except FileNotFoundError:
                pass
            break
            


    print("Ending face adding task")
    try:
        os.remove(face_lockfile)
    except FileNotFoundError:
        pass

from face_classify import faceAssigner
@shared_task(ignore_result=True, name='face_manager.assign_faces')
def thistask(redo_all=False):
    classify_lockfile = settings.CLASSIFY_LOCKFILE
    print(f"Classify lockfile is {classify_lockfile}")
    if os.path.exists(classify_lockfile):
        print("Classification is locked, exiting.")
        settings.LOGGER.warning("Classification is locked!")
        return
    else:
        f = open(classify_lockfile, 'w')
        f.close()

    try:
        classer = faceAssigner()
        classer.execute(redo_all)
    except:
        print("Image classification failed!")
    
    try:
        os.remove(classify_lockfile)
    except FileNotFoundError:
        pass


@shared_task(ignore_result=True, name='face_manager.set_face_counts')
def reset_task():
    people = Person.objects.all()
    for p in people:
        p.num_faces = p.face_declared.count()
        p.num_possibilities = p.face_poss1.count() + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
        p.num_unverified_faces = p.face_declared.filter(validated=False).count()
        p.save()
