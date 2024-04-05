#! /usr/bin/env python

from filepopulator import models
from django.core.management.base import BaseCommand
import PIL
from dateutil import parser
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import pytz
import re
from django.utils import timezone
from datetime import datetime

class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--image_id', type=int)

    def handle(self,  *args, **options):
        # print( options)
        image_id = options['image_id']

        try:
            image_id = int(image_id)
            file = models.ImageFile.objects.get(id=image_id)
        except:
            print("Image ID not working")
            return

        file.isProcessed = True
        super(type(file), file).save()
