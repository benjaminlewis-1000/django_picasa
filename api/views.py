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
from django.core.paginator import Paginator
from rest_framework.reverse import reverse, reverse_lazy
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Q

import json
from io import BytesIO
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


    def get_serializer_context(self):
        context = super(PersonViewSet, self).get_serializer_context()
        print(context)
        context.update({"request": self.request})
        return context



class PersonParamView(APIView):
    permission_classes = (IsAuthenticated,) 
    def get(self, request, *args, **kwargs):

        params = self.request.query_params
        if 'page' in params.keys():
            try:
                page = int(params['page'])
            except:
                return render_404(request, 'Page requested is not an integer.')
        else:
            page = 1

        id_key = kwargs['id']
        field = kwargs['field']

        if field not in ['face_declared', 'face_poss', 'directory']:
            return render_404(request, f"Requested type '{field}' not in ['face_declared', 'face_poss', 'directory']")

        if field in ['face_declared', 'face_poss']:
            try:
                person_obj = Person.objects.get(id=id_key)
            except Exception as e:
                return render_404(request, f"Error: Requested ID {id_key} not available in database.")

            if field == 'face_declared':
                faces = Face.objects.filter(declared_name=person_obj).values_list('id', flat=True)
            else:
                p1 = Q(poss_ident1=person_obj)
                p2 = Q(poss_ident2=person_obj)
                p3 = Q(poss_ident3=person_obj)
                p4 = Q(poss_ident4=person_obj)
                p5 = Q(poss_ident5=person_obj)
                faces = Face.objects.filter(p1 | p2 | p3 | p4 | p5).values_list('id', flat=True)
                
            id_list = list(faces)
        else:
            try:
                dir_obj = Directory.objects.get(id=id_key)
            except Exception as e:
                return render_404(request, f"Error: Requested ID {id_key} not available in database.")


            image_set = ImageFile.objects.filter(directory=dir_obj).values_list('id', flat=True)
            id_list = list(image_set)

        js = {'num_results': len(id_list), 'type': field, 'id_list': id_list,}

        return HttpResponse(json.dumps(js), content_type='application/json')


class FaceViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,) 

    queryset = Face.objects.all()
    serializer_class = api_ser.FaceSerializer
    print(queryset[0].face_thumbnail)

class KeyedImageView(APIView):
    def get(self, request, *args, **kwargs):

        params = self.request.query_params
        valid_types = ['face_highlight', 'face_array', 'face_source', 'slideshow', 'full_big', 'full_medium', 'full_small']
        
        def err_404(message=""):
            msg_start = f'Invalid request. Url is of format (/keyed_image/<type>/?id=<int>&access_key=<string>). Valid types include {valid_types}. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404


        if 'type' not in kwargs.keys():
            return err_404('You did not provide a type variable.')

        image_type = kwargs['type']
        if not image_type in valid_types:
            return err_404('You did not provide a valid type.')

        if 'id' not in params.keys():
            return err_404('ID key not passed as parameter.')
        else:
            id_key = params['id']

        if 'access_key' not in params.keys():
            return err_404('Access key not provided.')
        else:
            if params['access_key'] != settings.RANDOM_ACCESS_KEY:
                # print(params['access_key'], settings.RANDOM_ACCESS_KEY)
                return err_404('Invalid access key.')

        def getAndCheckID(obj_type, id_key):
            print(obj_type)
            if obj_type == 'person':
                obj = Person
            elif obj_type == 'face':
                obj = Face
            elif obj_type == 'image':
                obj = ImageFile
            print(obj)
            try:
                return obj.objects.get(id=id_key)
            except:
                return None# err_404(f"Invalid object id of type {image_type}")

        try:
            if image_type == 'face_highlight':
                person = getAndCheckID('person', id_key)
                image = person.highlight_img
            elif image_type == 'face_array':
                face = getAndCheckID('face', id_key)
                image = face.face_thumbnail
            elif image_type == 'face_source':
                face = getAndCheckID('face', id_key)
                source = face.source_image_file
                image = source.filename
                height = 700
                width = 1920 / 1080 * height
            elif image_type == 'slideshow':
                img_obj = getAndCheckID('image', id_key)
                image = img_obj.filename
                if 'height' in params.keys():
                    height = int(params['height'])
                else:
                    height = 1080

                width = 1920 / 1080 * height
            elif image_type in ['full_big', 'full_medium', 'full_small']:
                img_obj = getAndCheckID('image', id_key)
                if image_type == 'full_big':
                    image = img_obj.thumbnail_big
                elif image_type == 'full_medium':
                    image = img_obj.thumbnail_medium
                elif image_type == 'full_small':
                    image = img_obj.thumbnail_small
                print(image)

        except:
            return err_404(f'Bad id for object of type {image_type}')

        try:
            image = PIL.Image.open(image)
        except Exception as e:
            return err_404('Unable to open image.')

        if image_type in ['slideshow', 'face_source']: 
            # Resize the image dynamically
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

            
        FTYPE = 'JPEG'
        temp_thumb = BytesIO()
        # print(time.time()-s)
        image.save(temp_thumb, FTYPE)
        # print("Here")
        temp_thumb.seek(0)
        
        return HttpResponse(temp_thumb.read(), content_type="image/jpeg")
        return HttpResponse('asdf')

class filteredImagesView(APIView):

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        print("Hey")
        # print(self.request.query_params)
        params = self.request.query_params
        if len(params.keys()) == 0 or 'people' not in params.keys():
            # Return all objects!
            obj = ImageFile.objects.all().order_by('dateTaken').values_list('id', 'dateTaken')
            # return render_404(request, "No parameters were passed. Must pass a combination of 'people', 'year_start', 'year_end'...")
        else:
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
            

            obj = ImageFile.objects.filter(p_query).distinct().order_by('dateTaken').values_list('id', 'dateTaken')
            
        ids = list(([o[0] for o in obj]))
        dates = list(([o[1].isoformat() for o in obj]))
        
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

