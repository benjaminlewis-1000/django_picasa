#! /usr/bin/env python

from filepopulator import models
from django.core.management.base import BaseCommand



class Command(BaseCommand):
    def handle(self, *args, **options):
        models.ImageFile.objects.all().delete()
        models.Directory.objects.all().delete()