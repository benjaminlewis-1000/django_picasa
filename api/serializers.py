#! /usr/bin/env python

from django.contrib.auth.models import User, Group
from rest_framework import serializers
from filepopulator.models import ImageFile, Directory
from drf_queryfields import QueryFieldsMixin


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

      
class DirectorySerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
#    imgs_in_dir = serializers.HyperlinkedRelatedField(
#        many=True,
#        read_only=True,
#        view_name='ImageFile'
#    )

    # The ForeignKey in the ImageFile that points to directory needs to have
    # the related_name parameter set to 'image_set'. 
    image_set = ImageFileSerializer(read_only=True, many=True)

#    image_set = serializers.HyperlinkedRelatedField(many=True, view_name=image_set, read_only=True)
    class Meta:

    # Able to filter to get just the fields you want usin QueryFieldsMixin.
    # https://github.com/wimglenn/djangorestframework-queryfields
    # Suggested for list of all directories: 
    # <URL>/api/directories/?fields!=imgs_in_dir
        model = Directory
        fields = ['url', 'id', 'dir_path', 'top_level_name', 'image_set', 'mean_datetime', 'mean_datesec', \
        'first_datetime', 'first_datesec', ]
#        fields = ['dir_path', 'top_level_name', 'mean_datetime', 'mean_datesec', \
#        'first_datetime', 'first_datesec', ]
