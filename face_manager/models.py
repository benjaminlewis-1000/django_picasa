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
import filepopulator
# import image_face_extractor


# Create your models here
# logging.basicConfig(level=settings.LOG_LEVEL)

settings.LOGGER.warning("TODO: Implement model for faces")

class Person(models.Model):
    person_name = models.CharField(max_length=256)

    # id field has a primary key
    def delete(self):
        super(Person, self).delete()


class Face(models.Model):
    # Primary key comes for free
    declared_name = models.ForeignKey('Person', on_delete=models.PROTECT, related_name='face_declared', blank=True, null=True)
    source_image_file = models.ForeignKey('filepopulator.ImageFile', on_delete=models.CASCADE, blank=True, null=True)
    # ArrayField supported in PostGres
    face_encoding = ArrayField(
                            models.FloatField(),
                            size=128,
                        )

    # This field will contain the top 5 possible identities as categorized
    # by the FC network.

    poss_ident1 = models.ForeignKey('Person', on_delete=models.PROTECT, related_name='face_poss1', blank=True, null=True)
    poss_ident2 = models.ForeignKey('Person', on_delete=models.PROTECT, related_name='face_poss2', blank=True, null=True)
    poss_ident3 = models.ForeignKey('Person', on_delete=models.PROTECT, related_name='face_poss3', blank=True, null=True)
    poss_ident4 = models.ForeignKey('Person', on_delete=models.PROTECT, related_name='face_poss4', blank=True, null=True)
    poss_ident5 = models.ForeignKey('Person', on_delete=models.PROTECT, related_name='face_poss5', blank=True, null=True)
         #                    size=5,
         #                    blank=True,
         #                    null=True
                        # )

    written_to_photo_metadata = models.BooleanField(default=False)

    box_top = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_bottom = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_left = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_right = models.IntegerField(validators=[MinValueValidator(1)], default=-1)

    face_thumbnail = models.ImageField(upload_to='face_thumbnails', default=None)


    def __str__(self):
        return "{}".format(self.declared_name)

    def delete(self):
        os.remove(self.face_thumbnail.path)
        super(Face, self).delete()