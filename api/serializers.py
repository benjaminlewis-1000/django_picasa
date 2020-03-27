#! /usr/bin/env python

from django.contrib.auth.models import User, Group
from django.conf import settings
from rest_framework import serializers
from filepopulator.models import ImageFile, Directory
from face_manager.models import Person, Face
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
        fields = ['url', 'id', 'filename', 'pixel_hash', 'file_hash', 'camera_make', 'camera_model', \
                  'flash_info', 'exposure_num', 'exposure_denom', 'focal_num', 'focal_denom', \
                  'fnumber_num', 'fnumber_denom', 'iso_value', 'light_source', 'gps_lat_decimal', \
                  'gps_lon_decimal', 'thumbnail_big', 'thumbnail_medium', 'thumbnail_small', \
                  'full_res_path',  'exposure', 'directory', 'dateTaken', 'tags', 'description']

      
class DirectorySerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
#    imgs_in_dir = serializers.HyperlinkedRelatedField(
#        many=True,
#        read_only=True,
#        view_name='ImageFile'
#    )

    # The ForeignKey in the ImageFile that points to directory needs to have
    # the related_name parameter set to 'image_set'. 
    
    # The 'image_set' is set up with the related_name='image_set' argument in the 
    # Django model's ForeignKey.
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

class FaceSerializer(serializers.HyperlinkedModelSerializer):
    
    class Meta:

        model = Face
        fields = ['source_image_file', 'written_to_photo_metadata', 'face_encoding',  \
            'declared_name', 'poss_ident1', 'poss_ident2', 'poss_ident3', \
            'poss_ident4', 'poss_ident5', 'box_top', 'box_bottom', 'box_left', \
            'box_right', 'face_thumbnail']

class PersonSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    face_declared = FaceSerializer(read_only = True, many=True)
    class Meta:

        model = Person
        fields = ['person_name', 'highlight_img', 'face_declared']

# Source : https://medium.com/django-rest-framework/django-rest-framework-viewset-when-you-don-t-have-a-model-335a0490ba6f
class ParameterSerializer(serializers.Serializer):
    img_server_user = serializers.CharField(max_length=256)
    img_server_pw = serializers.CharField(max_length=256)

    
    class Parameters(object):
        def __init__(self):
            self.img_server_pw = settings.MEDIA_URL_PW   
            self.img_server_user = settings.MEDIA_URL_USER
            self.img_server_address = settings.MEDIA_URL


    def create(self, validated_data):
        return Parameters()
    
