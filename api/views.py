from django.shortcuts import render

from django.contrib.auth.models import User, Group
from filepopulator.models import ImageFile, Directory
from face_manager.models import Person, Face
from django.conf import settings
from rest_framework import viewsets, filters
#from api.serializers import UserSerializer, GroupSerializer, ImageFileSerializer, DirectorySerializer, ParameterSerializer
import api.serializers as api_ser
# Authentication: https://simpleisbetterthancomplex.com/tutorial/2018/11/22/how-to-implement-token-authentication-using-django-rest-framework.html
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
import base64
import cv2
import time
import PIL
from PIL import Image, ExifTags
from django.http import HttpResponse, Http404
from django.shortcuts import render
from django_filters.rest_framework import DjangoFilterBackend

import json
from django.db.models import Q
from filters.mixins import (
    FiltersMixin,
)

# Create your views here.

def render_404(request, message):
    
    response = render(request, '404.html', context = {'error': message})
    response.status_code = 404
    return response

class TokenPairWithUsername(TokenObtainPairView):
    permission_classes = (AllowAny,)
    serializer_class = api_ser.TokenPairSerializer

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    # The permission class means that the user must be
    # logged in to see this. 
    permission_classes = (IsAuthenticated,)
    queryset = User.objects.all()
    serializer_class = api_ser.UserSerializer

class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    permission_classes = (IsAuthenticated,)
    queryset = Group.objects.all()
    serializer_class = api_ser.GroupSerializer

class ImageViewSet(FiltersMixin, viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    permission_classes = (IsAuthenticated,)

    queryset = ImageFile.objects.all()
    # def get_queryset(self):
    #     return ImageFile.objects.all()
        
    serializer_class = api_ser.ImageFileSerializer

    filter_backends = (filters.OrderingFilter,)

    ordering_fields = ('id', 'filename')
    ordering = ('id')
    
    filter_mappings = {
        'id': 'id',
        'filename': 'filename__icontains',
    }

# Use URL filters https://github.com/manjitkumar/drf-url-filters
class DirectoryViewSet(FiltersMixin, viewsets.ModelViewSet):

    permission_classes = (IsAuthenticated,)

    queryset = Directory.objects.all()
    serializer_class = api_ser.DirectorySerializer
    filter_backends = (filters.OrderingFilter,)

    ordering_fields = ('id', 'dir_path', 'mean_datesec')
    ordering = ('id')

    filter_mappings = {
        'id': 'id',
        'dir_path': 'dir_path__icontains',
        'mean_datesec': 'mean_datesec',
        'mean_datesec__lte': 'mean_datesec__lte',
        'mean_datesec__gte': 'mean_datesec__gte',
        'time': 'mean_datesec',
    }
        
class PersonViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,) 

    queryset = Person.objects.all()
    
    serializer_class = api_ser.PersonSerializer

class FaceViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,) 

    queryset = Face.objects.all()
    serializer_class = api_ser.FaceSerializer

class ImageSlideshowView(APIView):
    # queryset = Session.objects.all()
    # serializer_class = SessionSerializer
    # permission_classes = (IsAuthenticated,)             # <-- And here

    def get(self, request, *args, **kwargs):
        # print(self.request.query_params)
        params = self.request.query_params
        if 'access_key' in params.keys():
            if params['access_key'] != settings.RANDOM_ACCESS_KEY:
                print(params['access_key'], settings.RANDOM_ACCESS_KEY)
                return render_404(request, "Incorrect access_key parameter passed")
        else:
            return render_404(request, "access_key parameter must be passed")
        if 'img_id' in params.keys():
            img_id = params['img_id']
        else:
            return render_404(request, "No img_id tag was passed")
        if 'height' in params.keys():
            height = int(params['height'])
        else:
            height = 1080

        width = 1920 / 1080 * height

        try:
            img_obj = ImageFile.objects.get(id=img_id)
        except Exception as e:
            return render_404(request, e)
        
        s = time.time()
        try:
            image = PIL.Image.open(img_obj.filename)
        except Exception as e:
            return render_404(request, e)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        try:
            exif=dict(image._getexif().items())
        except Exception as e:
            exif = {}

        if orientation in exif.keys():
            # print(exif[orientation])
            if exif[orientation] in [6, 8]:
                image.thumbnail( (height, width), Image.ANTIALIAS)
            else:
                image.thumbnail( (width,height), Image.ANTIALIAS)

            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image=image.rotate(90, expand=True)

        else:
            image.thumbnail( (width, height), Image.ANTIALIAS)
            
        # print(time.time()-s)
        # print(image.size)
        FTYPE = 'JPEG'
        from io import BytesIO
        temp_thumb = BytesIO()
        # print(time.time()-s)
        image.save(temp_thumb, FTYPE)
        # print("Here")
        temp_thumb.seek(0)
        
        return HttpResponse(temp_thumb.read(), content_type="image/jpeg")


class filteredImagesView(APIView):

    permission_classes = (IsAuthenticated,)
    
    def get(self, request, *args, **kwargs):
        # print(self.request.query_params)
        params = self.request.query_params
        if 'people' in params.keys():
            people = params['people'].split(',')
            if len(people) > 0:
                p_query = Q(face__declared_name__person_name=people[0])
            for p in people[1:]:
                p_query = p_query | Q(face__declared_name__person_name=p) 

        if 'year_start' in params.keys():
            # print(params['year_start'])
            # try:
            #     p_query = p_query | Q()
            # except NameError:
            #     p_query = Q()
            pass
        if 'year_end' in params.keys():
            # print(params['year_end'])
            # try:
            #     p_query = p_query | Q()
            # except NameError:
            #     p_query = Q()
            pass
        

        obj = ImageFile.objects.filter(p_query).distinct().order_by('dateTaken')
        ids = list(([o.id for o in obj]))
        dates = list(([o.dateTaken.isoformat() for o in obj]))
        
        zip_list = list(zip(ids, dates))
            
        js = {'url_keys': ids}
        if 'full_data' in params.keys():
            js['full_data'] = zip_list
        return HttpResponse(json.dumps(js), content_type='application/json')


# class HelloView(APIView):
#     permission_classes = (IsAuthenticated,)             # <-- And here

#     def get(self, request):
#         content = {'message': 'Hello, World!'}
#         return Response(content)

###############################################
### NOTE###  Don't Use ViewSet for MODELS!!!
class ParameterViewSet(viewsets.ViewSet):
### NOTE###  Don't Use ViewSet for MODELS!!!
###############################################
    serializer_class = api_ser.ParameterSerializer

    permission_classes = (IsAuthenticated,)

    def list(self, request):
        serializer = api_ser.ParameterSerializer(instance=api_ser.ParameterSerializer.Parameters(), many=False)
        return Response(serializer.data)

class StatsViewSet(viewsets.ViewSet):
### NOTE###  Don't Use ViewSet for MODELS!!!
###############################################
    serializer_class = api_ser.ServerStatsSerializer

    permission_classes = (IsAuthenticated,)

    def list(self, request):
        stat_serializer = api_ser.ServerStatsSerializer(instance=api_ser.ServerStatsSerializer.Stats(), many=False)
        return Response(stat_serializer.data)

