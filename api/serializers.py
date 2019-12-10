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
                  'gps_lon_decimal', 'thumbnail_big', 'thumbnail_medium', 'thumbnail_small', 'exposure']

      
class DirectorySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:

        model = Directory
        fields = ['dir_path', 'top_level_name', 'imgs_in_dir', 'mean_datetime', 'mean_datesec', \
        'first_datetime', 'first_datesec', ]