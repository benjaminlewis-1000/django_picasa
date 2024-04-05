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

def _get_date_taken(filename):

    exifDict = {}
    try:
        image = PIL.Image.open(filename)
    except PIL.Image.DecompressionBombError:
        image = PIL.Image.fromarray(cv2.imread(filename))

    try:
        info = image._getexif()
    except AttributeError as ae:
        info = None
    if info is not None:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            exifDict[decoded] = value

    dateTakenValid = False
    dateTaken = timezone.now()

    dateTakenKeys = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
    for exifKey in dateTakenKeys:
        datetaken_tmp = exifDict[exifKey] if exifKey in exifDict.keys() else None
        # Remediations for occasional problems - I've seen \x00\x00... in the string 
        # and date lines that are just spaces.
        if datetaken_tmp is None or re.match('^\s+$', datetaken_tmp) or re.match('0000:00:00 00:00:00', datetaken_tmp) or re.match('[\s:-]+', datetaken_tmp):
            continue  # No value at this EXIF key
        else:
            datetaken_tmp = datetaken_tmp.replace('\x00', '')
            try:
                date = datetime.strptime(datetaken_tmp, '%Y:%m:%d %H:%M:%S')
            except ValueError as ve:
                # settings.LOGGER.warning(f"Date taken format is _tmp}")
                date = parser.parse(datetaken_tmp)
#                         date = datetime.strptime(datetaken_tmp, '%Y-%m-%d %H:%M:%S')
            
            if date.tzinfo == None:
                date = pytz.utc.localize(date)
            if date < dateTaken: 
                dateTaken = date
                dateTakenValid = True

    return dateTaken, dateTakenValid

class Command(BaseCommand):
    def handle(self, *args, **options):

        files = models.ImageFile.objects.all()
        # files =[ models.ImageFile.objects.get(id=104442)]
        print(f"There are {len(files)} files.")

        for f in files:
            filename = f.filename
            taken, valid = _get_date_taken(filename)
            print(taken, valid, filename)

            if valid:
                f.dateTaken = taken
                super(type(f), f).save()