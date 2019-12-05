# Create your models here.
from django.db import models
from django.conf import settings
from django.forms import ModelForm
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.validators import *
from datetime import datetime
from django.utils import timezone

from io import BytesIO
from django.core.files.base import ContentFile
import os
import time
import hashlib
import PIL
from PIL.ExifTags import TAGS 
from PIL import Image
from datetime import datetime
from django.core.files import File
import pytz
from django.core.files.storage import default_storage as storage
# Create your models here.

class ImageModel(models.Model):
    # Primary key comes for free
    image = models.ImageField(upload_to='thumbnails_test')



# Thumbnail size tuple defined in an app-specific settings module - e.g. (400, 400)
THUMB_SIZE = (500, 500)

class Photo(models.Model):
    """
    Photo model with automatically generated thumbnail.
    """
    photo_name = models.CharField(max_length=255)
    thumbnail = models.ImageField(upload_to='thumbs_uploadimg', editable=False)

    def save(self, *args, **kwargs):
        """
        Make and save the thumbnail for the photo here.
        """
        if not self.make_thumbnail():
            raise Exception('Could not create thumbnail - is the file type valid?')
        super(Photo, self).save(*args, **kwargs)

    def delete(self):
        # print("Delete")
        files = Photo.objects.filter(id=self.id)
        # if self.thumbnail.name:
        # print("Remove " + self.thumbnail.name)
        # print(dir(self.thumbnail))
        # print(self.thumbnail.url)
        # print(self.thumbnail.path)
        # print(self.thumbnail.file)
        os.remove(self.thumbnail.path)
        super(Photo, self).delete()

    def make_thumbnail(self):
        """
        Create and save the thumbnail for the photo (simple resize with PIL).
        """
        # fh = storage.open(self.photo_name, 'r')
        try:
            image = Image.open(self.photo_name)
        except:
            print("Couldn't open")	
            return False

        image.thumbnail(THUMB_SIZE, Image.ANTIALIAS)
        # fh.close()

        # Path to save to, name, and extension
        thumb_name, thumb_extension = os.path.splitext(self.photo_name)
        thumb_extension = thumb_extension.lower()

        thumb_filename = 'asdf' + '_thumb' + thumb_extension
        # print(thumb_filename)

        if thumb_extension in ['.jpg', '.jpeg']:
            FTYPE = 'JPEG'
        elif thumb_extension == '.gif':
            FTYPE = 'GIF'
        elif thumb_extension == '.png':
            FTYPE = 'PNG'
        else:
            print("Thumb extension bad")
            return False    # Unrecognized file type

        # Save thumbnail to in-memory file as StringIO
        temp_thumb = BytesIO()
        image.save(temp_thumb, FTYPE)
        temp_thumb.seek(0)

        # Load a ContentFile into the thumbnail field so it gets saved
        self.thumbnail.save(thumb_filename, ContentFile(temp_thumb.read()), save=False)
        # print("Thumb saved as : " + self.thumbnail.name)
        # print(dir(self.thumbnail))
        # print(self.thumbnail.path, self.thumbnail.file, self.thumbnail.url)
        # if os.path.isfile(self.thumbnail.path):
        #     print("Is path")
        # else:
        #     print("Not there")
        temp_thumb.close()

        return True