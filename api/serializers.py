#! /usr/bin/env python

from django.contrib.auth.models import User, Group
from django.conf import settings
from rest_framework import serializers
from filepopulator.models import ImageFile, Directory
from face_manager.models import Person, Face
from drf_queryfields import QueryFieldsMixin
import datetime
import dateutil.parser
import json


class FaceSubsetSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
    face_name = serializers.SerializerMethodField()

    class Meta:

        model = Face
        fields = ['url', 'source_image_file', 'face_thumbnail', \
            'declared_name', 'face_name']

    def get_face_name(self, obj):
        # face_declared is the related field of the ForeignKey.
        return obj.declared_name.person_name


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

    parsed_date = serializers.SerializerMethodField()
    face_set = FaceSubsetSerializer(read_only = True, many=True)

    def get_parsed_date(self, obj):
        first_date = obj.dateTaken
        
        data = {'year': first_date.year, 
                'month': first_date.month,
                'day': first_date.day, 
                'hour': first_date.hour,
                'minute': first_date.minute, 
                'sec': first_date.second}

        return data

    class Meta:

        model = ImageFile
        fields = ['url', 'id', 'filename', 'pixel_hash', 'file_hash', 'camera_make', 'camera_model', \
                  'flash_info', 'exposure_num', 'exposure_denom', 'focal_num', 'focal_denom', \
                  'fnumber_num', 'fnumber_denom', 'iso_value', 'light_source', 'gps_lat_decimal', \
                  'gps_lon_decimal', 'thumbnail_big', 'thumbnail_medium', 'thumbnail_small', \
                  'full_res_path',  'exposure', 'directory', 'dateTaken', 'tags', 'description', 'isProcessed', \
                  'parsed_date', 'face_set']

class ImageSummarySerializer(serializers.HyperlinkedModelSerializer):

    class Meta:

        model = ImageFile
        fields = ['url', 'filename', 'thumbnail_big', 'thumbnail_medium', 'thumbnail_small', \
                  'directory']

      
class DirectorySerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
    # The ForeignKey in the ImageFile that points to directory needs to have
    # the related_name parameter set to 'image_set'. 
    
    # The 'image_set' is set up with the related_name='image_set' argument in the 
    # Django model's ForeignKey.
    image_set = ImageSummarySerializer(read_only=True, many=True)
    num_images = serializers.SerializerMethodField()
    year = serializers.SerializerMethodField()
    month = serializers.SerializerMethodField()

    def get_num_images(self, obj):
        # image_set is the related field of the ForeignKey.
        return obj.image_set.count()

    def get_year(self, obj):
        first_date = obj.first_datesec
        return datetime.datetime.fromtimestamp(first_date).year

    def get_month(self, obj):
        first_date = obj.first_datesec
        return datetime.datetime.fromtimestamp(first_date).month

    class Meta:

    # Able to filter to get just the fields you want usin QueryFieldsMixin.
    # https://github.com/wimglenn/djangorestframework-queryfields
    # Suggested for list of all directories: 
    # <URL>/api/directories/?fields!=imgs_in_dir
        model = Directory
        fields = ['url', 'id', 'dir_path', 'top_level_name', 'mean_datetime', 'mean_datesec', \
        'first_datetime', 'first_datesec', 'num_images', 'year', 'month', 'image_set']
#        fields = ['dir_path', 'top_level_name', 'mean_datetime', 'mean_datesec', \
#        'first_datetime', 'first_datesec', ]

class FaceSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):

    face_name = serializers.SerializerMethodField()

    def get_face_name(self, obj):
        # face_declared is the related field of the ForeignKey.
        return obj.declared_name.person_name

    class Meta:

        model = Face
        # No face_encoding in this api -- it's not necessary. 
        fields = ['url', 'source_image_file', 'written_to_photo_metadata',  \
            'declared_name', 'face_name', 'poss_ident1', 'poss_ident2', 'poss_ident3', \
            'poss_ident4', 'poss_ident5', 'box_top', 'box_bottom', 'box_left', \
            'box_right', 'face_thumbnail']


class PersonSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):

    num_faces = serializers.SerializerMethodField()
    face_declared = FaceSubsetSerializer(read_only = True, many=True)

    class Meta:

        model = Person
        fields = ['url', 'person_name', 'highlight_img', 'num_faces', 'face_declared']

    def get_num_faces(self, obj):
        # face_declared is the related field of the ForeignKey.
        return obj.face_declared.count()

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
    
class ServerStatsSerializer(serializers.Serializer):
    num_imgs = serializers.IntegerField()
    num_face_processed = serializers.IntegerField()
    percent_face_processed = serializers.CharField(max_length=256)
    num_people = serializers.IntegerField()
    num_faces = serializers.IntegerField()
    estimated_hours_facerec_left = serializers.FloatField()
    imgs_per_hour_est = serializers.IntegerField()

    class Stats(object):
        def __init__(self):
            self.num_imgs = ImageFile.objects.count() 
            self.num_face_processed = ImageFile.objects.filter(isProcessed=True).count()
            self.num_people = Person.objects.count()
            self.num_faces = Face.objects.count()
            self.imgs_per_hour_est = 800 # An estimate -- I'm clocking 177 images in 26 minutes
            imgs_left_to_process = self.num_imgs - self.num_face_processed
            self.estimated_hours_facerec_left = imgs_left_to_process / self.imgs_per_hour_est
            proc_percent = self.num_face_processed / self.num_imgs * 100
            self.percent_face_processed = f'{proc_percent:.2f}%'

    def create(self, validated_data):
        return Stats()