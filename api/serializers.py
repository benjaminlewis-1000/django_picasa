#! /usr/bin/env python

from django.conf import settings
from django.contrib.auth.models import User, Group
from django.core.paginator import Paginator
from drf_queryfields import QueryFieldsMixin
from face_manager.models import Person, Face
from filepopulator.models import ImageFile, Directory
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
import datetime
import dateutil.parser
import json


class TokenPairSerializer(TokenObtainPairSerializer):

    @classmethod
    def get_token(cls, user):
        token = super(TokenPairSerializer, cls).get_token(user)

        # Add custom claims
        token['username'] = user.username
        return token


class FaceSubsetSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
    # face_name = serializers.SerializerMethodField()

    class Meta:

        model = Face
        # fields = ['url', 'source_image_file', 'face_thumbnail', \
        #     'declared_name', 'face_name']
        fields = ['url']

    # def get_face_name(self, obj):
    #     # face_declared is the related field of the ForeignKey.
    #     return obj.declared_name.person_name


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

    face_name = serializers.SerializerMethodField(read_only=True)
    poss_face_name1 = serializers.SerializerMethodField(read_only=True)
    poss_face_name2 = serializers.SerializerMethodField(read_only=True)
    poss_face_name3 = serializers.SerializerMethodField(read_only=True)
    poss_face_name4 = serializers.SerializerMethodField(read_only=True)
    poss_face_name5 = serializers.SerializerMethodField(read_only=True)

    def get_face_name(self, obj):
        # num_calls += 1
        # face_declared is the related field of the ForeignKey.
        return obj.declared_name.person_name

    def get_poss_face_name1(self, obj):
        if obj.poss_ident1 is not None:
            return obj.poss_ident1.person_name
        else:
            return "N/A"

    def get_poss_face_name2(self, obj):
        if obj.poss_ident2 is not None:
            return obj.poss_ident2.person_name
        else:
            return "N/A"
            
    def get_poss_face_name3(self, obj):
        if obj.poss_ident3 is not None:
            return obj.poss_ident3.person_name
        else:
            return "N/A"

    def get_poss_face_name4(self, obj):
        if obj.poss_ident4 is not None:
            return obj.poss_ident4.person_name
        else:
            return "N/A"

    def get_poss_face_name5(self, obj):
        if obj.poss_ident5 is not None:
            return obj.poss_ident5.person_name
        else:
            return "N/A"


    class Meta:

        model = Face
        print("Getting")
        # No face_encoding in this api -- it's not necessary. 
        fields = ['url', 'source_image_file', 'written_to_photo_metadata',  \
            'declared_name', 'face_name', \
            'poss_ident1', 'poss_face_name1', 'weight_1', \
            'poss_ident2', 'poss_face_name2', 'weight_2', \
            'poss_ident3', 'poss_face_name3', 'weight_3', \
            'poss_ident4', 'poss_face_name4', 'weight_4', \
            'poss_ident5', 'poss_face_name5', 'weight_5', \
            'box_top', 'box_bottom', 'box_left', \
            'box_right', 'face_thumbnail', 'rejected_fields']
        # fields = ['url', 'source_image_file', 'written_to_photo_metadata',  \
        #     'declared_name', 'face_name', 'poss_ident1', 'weight_1', 'poss_face_name1', \
        #     'poss_ident2', 'poss_face_name2', 'poss_ident3',  'poss_ident4', \
        #     'poss_ident5', 'box_top', 'box_bottom', 'box_left', \
        #     'box_right', 'face_thumbnail']




class PersonSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):

    num_faces = serializers.SerializerMethodField()
    num_possibilities = serializers.SerializerMethodField()
    face_declared = FaceSubsetSerializer(read_only = True, many=True) 

    class Meta:

        model = Person
        fields = ['url', 'person_name', 'highlight_img', 'num_faces', 'num_possibilities', 'face_declared', 'id']
        # fields = ['url', 'person_name', 'highlight_img', 'num_faces', 'id']

    def get_num_faces(self, obj):
        # face_declared is the related field of the ForeignKey.
        return obj.face_declared.count()

    def get_num_possibilities(self, obj):
        # face_declared is the related field of the ForeignKey.
        f1 = obj.face_poss1.count()
        f2 = obj.face_poss2.count()
        f3 = obj.face_poss3.count()
        f4 = obj.face_poss4.count()
        f5 = obj.face_poss5.count()

        return  f1 + f2 + f3 + f4 + f5

    
    # def update(self, instance, validated_data):
    #     print("Seri update")

# Source : https://medium.com/django-rest-framework/django-rest-framework-viewset-when-you-don-t-have-a-model-335a0490ba6f
class ParameterSerializer(serializers.Serializer):
    img_server_user = serializers.CharField(max_length=256)
    img_server_pw = serializers.CharField(max_length=256)
    random_access_key = serializers.CharField(max_length=256)

    class Parameters(object):
        def __init__(self):
            self.img_server_pw = settings.MEDIA_URL_PW   
            self.img_server_user = settings.MEDIA_URL_USER
            self.img_server_address = settings.MEDIA_URL
            self.random_access_key = settings.RANDOM_ACCESS_KEY

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
    num_unlabeled_faces = serializers.IntegerField()

    class Stats(object):
        def __init__(self):
            self.num_imgs = ImageFile.objects.count() 
            self.num_face_processed = ImageFile.objects.filter(isProcessed=True).count()
            self.num_people = Person.objects.count()
            self.num_faces = Face.objects.count()
            self.num_unlabeled_faces = Face.objects.filter(declared_name__person_name=settings.BLANK_FACE_NAME).count( )
            self.imgs_per_hour_est = 800 # An estimate -- I'm clocking 177 images in 26 minutes
            imgs_left_to_process = self.num_imgs - self.num_face_processed
            self.estimated_hours_facerec_left = imgs_left_to_process / self.imgs_per_hour_est
            proc_percent = self.num_face_processed / self.num_imgs * 100
            self.percent_face_processed = f'{proc_percent:.2f}%'

    def create(self, validated_data):
        return Stats()
