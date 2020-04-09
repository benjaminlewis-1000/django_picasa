
from __future__ import absolute_import, unicode_literals
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from face_manager import models as face_models
from filepopulator import models as file_models
import sys
import time
import torch
import torch.optim as optim
import torch.utils.data as data
import random
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from time import sleep
import io
from PIL import Image
import pickle

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings

from celery import shared_task, task
import time
import os

from face_manager.scripts import populateFromImage, populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection
from face_manager.models import Person, Face
from filepopulator.models import ImageFile

class Command(BaseCommand):
    
    def handle(self, *args, **options):

        print("Hi there!")
        settings.LOGGER.debug("Starting face extraction...")

        face_lockfile = settings.FACE_LOCKFILE

        server_conn = establish_multi_server_connection()
        print("servers: ", len(server_conn))

        num_servers = len(server_conn.server_ips)
        if num_servers == 0:
            settings.LOGGER.critical('No GPU servers found')
            return

        import queue
        import threading

        def worker(ip_num):
            print("Hi, I'm worker # ", ip_num)
            while True:
                if img_q.qsize() == 0:
                    print("all done!")
                    break
                else:
                    img = img_q.get()
                print(f"Queue size is {img_q.qsize()}, worker is {ip_num}, Image ID is {img.id}")
                # if 
                is_ok = None
                simeDelay = random.randrange(0, 10) * 0.03
                sleep(simeDelay)
                qs = img_q.qsize()
                while is_ok is None:
                    try:
                        is_ok = server_conn.check_ip(ip_num)
                    except OSError:
                        timeDelay = random.randrange(0, 2)
                        print("Failed in worker ", ip_num)
                        sleep(timeDelay)
                        
                    if not img.isProcessed:
                        try:
                            populateFromImageMultiGPU(img.filename, server_conn = server_conn, server_idx = ip_num, ip_checked=True)
                        except OSError:
                            break


        img_done = 0
        all_images = ImageFile.objects.filter(isProcessed=False)
        print(len(all_images))
        while img_done < len(all_images):
            img_q = queue.Queue()
            threads = []
            for img_idx in range(img_done, len(all_images)):
                img_done += 1
                if img_q.qsize() > 500:
                    break
                img = all_images[img_idx]
                if not img.isProcessed:
                    img_q.put(img)

            for i in range(num_servers):
                t = threading.Thread(target=worker, args=(i,))
                t.start()
                threads.append(t)

            # img_q.join()
            for t in threads:
                t.join()

            #     