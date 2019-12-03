#! /usr/bin/env python

from django.contrib.auth.models import User, Group
from rest_framework import serializers
from filepopulator.models import ImageFile, Directory


class UserSerializer(serializers.HyperlinkedModelSerializer):
	"""docstring for UserSerializer"""
	class Meta:
		model = User
		fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']

class ImageFileSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:

        model = ImageFile
        fields = ['id', 'filename', 'pixel_hash', 'file_hash', 'camera_make', 'camera_model', \
                  'flash_info', 'exposure_num', 'exposure_denom', 'focal_num', 'focal_denom', \
                  'fnumber_num', 'fnumber_denom', 'iso_value', 'light_source', 'gps_lat_decimal', \
                  'gps_lon_decimal', 'thumbnail_big', 'thumbnail_medium', 'thumbnail_small']
        # id = serializers.IntegerField(read_only=True)
        # filename = serializers.CharField(max_length=255, required=True, allow_blank=False)
        # pixel_hash = serializers.CharField(max_length = 64, allow_blank = False, default = -1)
        # file_hash = serializers.CharField(max_length = 64, allow_blank = False, default = -1)
      
        # camera_make = serializers.CharField(max_length = 64, null=False, default='Unknown')
        # camera_model = serializers.CharField(max_length = 64, null=False, default='Unknown')
        # flash_info = serializers.IntegerField(default= -1)
        # exposure_num = serializers.IntegerField(default= -1)
        # exposure_denom = serializers.IntegerField(default= -1)
        # focal_num = serializers.IntegerField(default= -1)
        # focal_denom = serializers.IntegerField(default= -1)
        # fnumber_num = serializers.IntegerField(default= -1)
        # fnumber_denom = serializers.IntegerField(default= -1)
        # iso_value = serializers.IntegerField(default= -1)
        # light_source = serializers.IntegerField(default= -1)
        # gps_lat_decimal = serializers.FloatField(default=-999,validators=[validate_lat])
        # gps_lon_decimal = serializers.FloatField(default=-999,validators=[validate_lon])

'''

    # CASCADE is expected; if delete directory, delete images.
    directory = models.ForeignKey(Directory, on_delete=models.PROTECT)
   
    # Thumbnails 
    thumbnail_big = models.ImageField(upload_to='thumbnails_big', editable=False, default=None)
    thumbnail_medium = models.ImageField(upload_to='thumbnails_med', editable=False, default=None)
    thumbnail_small = models.ImageField(upload_to='thumbnails_small', editable=False, default=None)

    # Fields for metadata
 
    # Default for date added is now.
    dateAdded = models.DateTimeField( default=timezone.now )
    width = models.IntegerField(validators=[MinValueValidator(1)])
    height = models.IntegerField(validators=[MinValueValidator(1)])

    # Default for date take in January 1, 1899.
    dateTaken = models.DateTimeField( default=datetime(2018, 1, 1) )
    dateTakenValid = models.BooleanField(default=False)

    # isProcessed -- whether the photo has had faces detected.
    isProcessed = models.BooleanField(default=False)
    orientation = models.IntegerField(default=-8008)
'''