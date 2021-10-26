from django.shortcuts import render

from django.contrib.auth.models import User, Group
from filepopulator.models import ImageFile, Directory
from face_manager.models import Person, Face
from django.db.models import Count
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
from rest_framework.decorators import action
import datetime
import time
import os
import numpy as np
import threading
from queue import Queue
import PIL
from PIL import Image, ExifTags
from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from rest_framework.reverse import reverse, reverse_lazy
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Q
from time import sleep

import json
from io import BytesIO
from filters.mixins import (
    FiltersMixin,
)


def open_img_oriented(filename):

    try:
        image = PIL.Image.open(filename)
    except Exception as e:
        return None

    # if image_type in ['slideshow', 'face_source']: 
    #     # Resize the image dynamically
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break

    try:
        exif=dict(image._getexif().items())
    except Exception as e:
        exif = {}

    if orientation in exif.keys():
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

    return image

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
        print("Context: ", context)
        context.update({"request": self.request})
        return context


    @action(detail=True, methods=['put'])
    def toggle_further_unlikely(self, request, pk=None):
        # Accessible as <root>/api/people/<face_id>/toggle_further_unlikely/
        # Accept: HTML PUT
        # Given face_id in the URL, toggle the further_images_unlikely property. 
        person = self.get_object()

        def err_404(message=""):
            msg_start = f'Invalid toggle unlikely request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        unlikely = person.further_images_unlikely
        person.further_images_unlikely = not unlikely
        person.save()

        js = {'success': True, }
        return HttpResponse(json.dumps(js), content_type='application/json')

class PersonListView(APIView):
    permission_classes = (IsAuthenticated,) 
    def get(self, request, *args, **kwargs):
        params = self.request.query_params

        all_people = Person.objects.all()
        num_people = all_people.count()
        domain_name = 'https://' + os.environ['WEBAPP_DOMAIN'] + '/api/people'

        # ###############
        # s = time.time()
        # all_people = Person.objects.all().order_by('id')
        # p_facecount = Person.objects.annotate(n=Count('face_declared'))
        # num_faces = [p.n for p in p_facecount]
        # urls = [f'{domain_name}/{p.id}/' for p in all_people]
        # p_poss1 = Person.objects.annotate(n=Count('face_poss1'))
        # p_poss2 = Person.objects.annotate(n=Count('face_poss2'))
        # num_possibilities = [p_poss1[x].n + p_poss2[x].n for x in range(num_people)]
        # ids = [p.id for p in all_people]
        # names = [p.person_name for p in all_people]
        # further_images_unlikely = [p.further_images_unlikely for p in all_people]
        # unvalidated_q = all_people.filter(face_declared__validated=False).annotate(n=Count('face_declared'))
        # unvalidated = [p.n for p in unvalidated_q]
        # print(time.time() - s)


        result_list = []
        p_queue = Queue()

        for p in all_people:
            p_queue.put(p)

        def worker():
            while not p_queue.empty():
                p = p_queue.get()
                p_dict = {}
                p_dict['num_faces'] = p.num_faces # face_declared.count()
                p_dict['url'] = f'{domain_name}/{p.id}/'
                p_dict['num_possibilities'] = p.num_possibilities # face_poss1.count() + p.face_poss2.count()
                p_dict['id'] = p.id
                p_dict['person_name'] = p.person_name
                p_dict['further_images_unlikely'] = p.further_images_unlikely

                # p_dict['num_faces'] = p.num_faces
                # p_dict['num_possibilities'] = p.poss_1 + p.poss_2
                p_dict['num_unverified_faces'] = p.num_unverified_faces # face_declared.filter(validated=False).count()
                result_list.append(p_dict)
                p_queue.task_done()
                # return p_dict
        
        for i in range(4):
            threading.Thread(target=worker).start()
        

        p_queue.join()

        js = {'count': num_people, 'next': None, 'previous': None, 'results': result_list,}

        return HttpResponse(json.dumps(js), content_type='application/json')
    
class FolderListView(APIView):
    permission_classes = (IsAuthenticated,) 
    def get(self, request, *args, **kwargs):
        params = self.request.query_params

        all_folders = Directory.objects.all()
        f_queue = Queue()

        for f in all_folders:
            f_queue.put(f)

        result_list = []
        domain_name = 'https://' + os.environ['WEBAPP_DOMAIN'] + '/api/directories'

        def worker():
            while not f_queue.empty():
                fold = f_queue.get()
                f_dict = {}
                f_dict['id'] = fold.id
                f_dict['url'] = f'{domain_name}/{fold.id}/'
                f_dict['top_level_name'] = fold.dir_path.split('/')[-1]
                f_dict['first_datesec'] = fold.first_datesec
                f_dict['mean_datesec'] = fold.mean_datesec
                f_dict['num_images'] = fold.num_images
                f_dict['year'] = datetime.datetime.fromtimestamp(fold.first_datesec).year
                result_list.append(f_dict)
                f_queue.task_done()

        for i in range(4):
            threading.Thread(target=worker).start()
        f_queue.join()

        # /directories/?fields=id,url,top_level_name,first_datesec,mean_datesec,num_images,year&limit=2000

        js = {'count': len(all_folders), 'next': None, 'previous': None, 'results': result_list,}

        return HttpResponse(json.dumps(js), content_type='application/json')

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

        if 'only_unverified' in params.keys():
            value = params['only_unverified']
            # Invert it for the DB query.
            do_only_unverified = (value.lower() == 'true')
        else:
            do_only_unverified = False

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

                default_query = Q(declared_name=person_obj)
                unverified_query = Q(validated=False)
                if do_only_unverified:
                    faces = Face.objects.filter(default_query & unverified_query).values_list('id', flat=True)
                else:
                    faces = Face.objects.filter(default_query).values_list('id', flat=True)
            else:

                faces1 = list(Face.objects.filter(poss_ident1=person_obj).values_list('id', 'weight_1'))
                faces2 = list(Face.objects.filter(poss_ident2=person_obj).values_list('id', 'weight_2'))
                # faces3 = list(Face.objects.filter(poss_ident3=person_obj).values_list('id', 'weight_3'))
                # faces4 = list(Face.objects.filter(poss_ident4=person_obj).values_list('id', 'weight_4'))
                # faces5 = list(Face.objects.filter(poss_ident5=person_obj).values_list('id', 'weight_5'))

                f = faces1 + faces2  #+ faces3 + faces4 + faces5
                faces = sorted(f, key=lambda tup: tup[1], reverse=True)
                faces = [f[0] for f in faces]
                
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

    def get_serializer_context(self):
        context = super(FaceViewSet, self).get_serializer_context()
        # print("Context for face: ", context, dir(context['request']))
        context.update({"request": self.request})
        return context


    def highlight_from_face(self, face, person_name):
        # Helper method to extract a 500x500 pixel highlight
        # image from the face. Uses the source_image_file
        # of the face to get the original image, extracts the
        # highlight given the face borders, and resizes. Returns
        # the face as a ContentFile, which is used in the 
        # save method for the person.highlight_image

        # Get the borders
        r_left = face.box_left
        r_right = face.box_right
        r_bot = face.box_bottom
        r_top = face.box_top

        # Make the borders a square
        top_to_bot = r_bot - r_top
        left_to_right = r_right - r_left
        lr_cent = r_left + left_to_right // 2
        tb_cent = r_top + top_to_bot // 2

        extent = min(top_to_bot, left_to_right) // 2
        r_left = lr_cent - extent
        r_right = lr_cent + extent
        r_top = tb_cent - extent
        r_bot = tb_cent + extent

        # Extract the source file, oriented. 
        source_file = face.source_image_file.filename
        pixel_hash = face.source_image_file.pixel_hash
        image = open_img_oriented(source_file)
        # Crop out the image, resize, encode in ByteIO, etc.
        img_thmb = image.crop((r_left, r_top, r_right, r_bot))
        img_thmb = np.array(img_thmb)
        img_thmb = cv2.resize(img_thmb, (500, 500))
        # Convert color space
        img_thmb = cv2.cvtColor(img_thmb, cv2.COLOR_BGR2RGB)

        is_success, person_buff = cv2.imencode(".jpg", img_thmb)

        # Save thumbnail to in-memory file as BytesIO
        person_byte_thumbnail = BytesIO(person_buff)

        person_thumb_filename = f'{pixel_hash}_{person_name}.jpg'
        
        person_byte_thumbnail.seek(0)
        person_content_file = ContentFile(person_byte_thumbnail.read())

        # Return the filename to write and the content file.
        return person_thumb_filename, person_content_file

    @action(detail=True, methods=['patch'])
    def verify_face(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/verify_face/
        # Accept: HTML PATCH
        # Set the face's 'verified' status to True. 
        face = self.get_object()
        data = request.data

        # print('verify face', face, face.id)

        def err_404(message=""):
            msg_start = f'Invalid face update request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404
            
        face.verify_person_in_image()

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['patch'])
    def assign_face_to_person(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/assign_face_to_person/
        # Accept: HTML PATCH
        # Requires a declared_name_key parameter in the body which
        # has the pk for a desired person with which to associate
        # the face.
        # Given the face to set to a person, changes the face's 
        # declared_name to point to the person and zeros
        # out any possible identity fields and associated 
        # weights. 

        # Get the face object and the request data
        face = self.get_object()
        data = request.data

        def err_404(message=""):
            msg_start = f'Invalid face update request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'declared_name_key' not in request.data.keys():
            return err_404("declared_name_key not set in body")

        # Get the person to associate and ensure the value
        # passed is an integer
        name_key = request.data['declared_name_key']
        print(name_key)
        try:
            name_key = int(name_key)
        except:
            return err_404("declared_name_key passed was not an integer.")

        # Given the name key, try to find the appropriate person. 
        # Return a 404 if that person pk is not in the database. 
        try:
            identity = Person.objects.get(id=name_key)
        except:
            return err_404("Desired declared name key is not a key found in the database.")
            
        # Update the face object fields accordingly
        face.associate_person(name_key)

        # Return a json block with success=true, the new key (passed as
        # an argument), and the declared human-meaningful name. 
        js = {'success': True, 'new_key': name_key, 'new_name': face.declared_name.person_name}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['put'])
    def face_to_new_person(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/face_to_new_person/
        # Accept: HTML PUT
        # Given face_id in the URL and person_name in the request
        # body, create a new person of name person_name. Extract
        # the face highlight image from the original source image
        # of face.source_image and resize it to 500 px. Save the
        # highlight image to the new person, assign the name
        # to the new person, and assign this face to the new person
        # and zero out all proposed images. 
        face = self.get_object()
        data = request.data

        def err_404(message=""):
            msg_start = f'Invalid face update request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'person_name' not in data.keys():
            return err_404("person_name not set in body")
        else:
            person_name = data['person_name']


        new_person = Person(person_name = person_name)
        
        filename, person_content_file = self.highlight_from_face(face, person_name)
        # Load a ContentFile into the thumbnail field so it gets saved
        new_person.highlight_img.save(filename, person_content_file) 

        new_person.save()

        face.associate_person(new_person.id)

        js = {'success': True, 'new_id': new_person.id, 'new_name': person_name}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['put'])
    def set_as_person_thumbnail(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/set_as_person_thumbnail/
        # Accept: HTML PUT
        # No body parameters
        # Given the face, sets the associated person's thumbnail to be
        # that face. 
        print("Thumb")
        face = self.get_object()
        person = face.declared_name

        print(person.person_name)
        filename, person_content_file = self.highlight_from_face(face, person.person_name)
        person.highlight_img.save(filename, person_content_file) 

        person.save()
        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')


    @action(detail=True, methods=['put'])
    def ignore_face(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/ignore_face/
        # Accept: HTML PUT
        # Body parameter : ignore_type in ['soft', 'hard']
        # Given the face, set the assigned name to '.ignore'
        def err_404(message=""):
            msg_start = f'Invalid face ignore request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'ignore_type' not in request.data.keys():
            return err_404("ignore_type not set in body")
        else:
            ignore_type = request.data['ignore_type']

        if ignore_type.lower() not in ['soft', 'hard']:
            return err_404("ignore_type body parameter must be either 'soft' or 'hard'.")

        face = self.get_object()

        # print('ignore face', face, face.id)
        if ignore_type == 'soft':
            ignore_person = Person.objects.filter(person_name='.ignore')
        else:
            soft_person = Person.objects.filter(person_name='.ignore')[0]
            hard_person = Person.objects.filter(person_name='.realignore')[0]
            if face.declared_name == soft_person or face.declared_name == hard_person or face.poss_ident1 == soft_person:
                pass
            else:
                #print("Not good")
                js = {'success': False}
                #return HttpResponse(json.dumps(js), content_type='application/json')
                return err_404("Person currently assigned is not .ignore or .realignore.")
            
            ignore_person = Person.objects.filter(person_name='.realignore')
            

        face.associate_person(ignore_person[0].id)

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['put'])
    def unassign_face(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/unassign_face/
        # Accept: HTML PUT
        # Body parameter : None
        # Given the face, remove any assigned face

        face = self.get_object()
        unassigned_person = Person.objects.filter(person_name=settings.BLANK_FACE_NAME)[0]
        
        face.associate_person(unassigned_person.id)

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')


    @action(detail=True, methods=['put'])
    def reject_association(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/reject_association/
        # Accept: HTML PUT
        # Body parameter : unassociate_id (person_id)
        # Given the face, remove association to the given
        # person and add the person_id to the rejected_fields. 
        
        print(request.data, request.data.keys())
        def err_404(message=""):
            msg_start = f'Invalid reject association request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'unassociate_id' not in request.data.keys():
            return err_404("unassociate_id not set in body")
        else:
            unassociate_id = request.data['unassociate_id']

        try:
            unassociate_id = int(unassociate_id)
        except:
            return err_404("unassociate_id passed was not an integer.")

        face = self.get_object()
        face.reject_association(unassociate_id)

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')

class KeyedImageView(APIView):
    def get(self, request, *args, **kwargs):
        '''
        Call to get an image defined by a given key.
        Method: GET
        URL: /api/keyed_image/<key_value>/?=<request_type>&access_key=<key>
        Valid types:
        face_highlight: key type - person. Gets the highlight image of that person
        face_array: key type - face. Gets the cropped image of just that face
        face_source: key type: face. Gets the full image containing that face.
        slideshow: key type: image. Gets a reduced resolution image for the slideshow.
        full_big/medium/small: key type: image. Gets the corresponding thumbnail image. 
        '''

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
            # print(obj_type)
            if obj_type == 'person':
                obj = Person
            elif obj_type == 'face':
                obj = Face
            elif obj_type == 'image':
                obj = ImageFile
            # print(obj)
            try:
                return obj.objects.get(id=id_key)
            except:
                return None# err_404(f"Invalid object id of type {image_type}")

        try:
            if image_type == 'face_highlight':
                person = getAndCheckID('person', id_key)
                image = person.highlight_img
                height = 700
                width = 1920 / 1080 * height
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
                    height = settings.DEFAULT_RESOLUTION_HEIGHT

                width = 1920 / 1080 * height
            elif image_type in ['full_big', 'full_medium', 'full_small']:
                img_obj = getAndCheckID('image', id_key)
                if image_type == 'full_big':
                    image = img_obj.thumbnail_big
                elif image_type == 'full_medium':
                    image = img_obj.thumbnail_medium
                elif image_type == 'full_small':
                    image = img_obj.thumbnail_small
                    
        except:
            return err_404(f'Bad id for object of type {image_type}')


        # print(image)
        image = open_img_oriented(image)
        # print(image)
        try:
            image.thumbnail( (width, height), Image.ANTIALIAS)
        except:
            pass

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
        # print(self.request.query_params)
        params = self.request.query_params
        print(params)

        if len(params.keys()) == 0 : # or 'people' not in params.keys():
            # Return all objects!
            obj = ImageFile.objects.all().order_by('dateTaken').values_list('id', 'dateTaken')
            # return render_404(request, "No parameters were passed. Must pass a combination of 'people', 'year_start', 'year_end'...")
        else:
            p_query = None
            years_query = None
            if 'people' in params.keys():
                people = params['people'].split(',')
                if len(people) > 0:
                    people_query = Q(face__declared_name__person_name=people[0])
                for p in people[1:]:
                    people_query = people_query | Q(face__declared_name__person_name=p) 

                # people_query = (people_query) & ~Q(face__declared_name__person_name='Charlotte Lewis')
                if p_query is None:
                    p_query = people_query



            if 'year_start' in params.keys():
                year_start = int(params['year_start'])
                year_start_query = Q(face__source_image_file__dateTaken__year__gte = year_start)
                if years_query is None:
                    years_query = year_start_query
                else:
                    years_query = years_query & year_start_query

            if 'year_end' in params.keys():
                year_end = int(params['year_end'])
                year_end_query = Q(face__source_image_file__dateTaken__year__lte = year_end)
                if years_query is None:
                    years_query = year_end_query
                else:
                    years_query = years_query & year_end_query
            

            if p_query is None:
                p_query = years_query
            else:
                if years_query is not None:
                    p_query = p_query | years_query

            print(p_query)
            obj = ImageFile.objects.filter(p_query).exclude(face__declared_name__person_name='Charlotte Lewis').distinct().order_by('dateTaken').values_list('id', 'dateTaken')

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

