from django.db import models
from django.conf import settings
from django.forms import ModelForm
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.validators import *
from datetime import datetime
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField

import os
import sys

path_to_script = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_to_script)

import logging
import faceid
import filepopulator

# Create your models here
# logging.basicConfig(level=settings.LOG_LEVEL)

settings.LOGGER.warning("TODO: Implement model for faces")

class Person(models.Model):
	person_name = models.CharField(max_length=256)

class Face(models.Model):
    # Primary key comes for free
    declared_name = models.ForeignKey('Person', on_delete=models.PROTECT,)
    source_image = models.ForeignKey('filepopulator.ImageFile', on_delete=models.CASCADE, blank=True, null=True)
    # ArrayField 
    face_encoding = ArrayField(
    						models.FloatField(),
    						size=128,
    					)

    top_idents = ArrayField(
    						models.CharField(max_length=256)
    					)

    written_in_photo = models.BooleanField(default=False)
    box_top = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_bottom = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_left = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_right = models.IntegerField(validators=[MinValueValidator(1)], default=-1)

    face_thumbnail = models.ImageField(upload_to='face_thumbnails', default=None)


    def __str__(self):
        return "{}".format(self.declared_name)

    def face_from_facerect(self):
    	pass