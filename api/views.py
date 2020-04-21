from django.shortcuts import render

from django.contrib.auth.models import User, Group
from filepopulator.models import ImageFile, Directory
from face_manager.models import Person, Face
from django.conf import settings
from rest_framework import viewsets
#from api.serializers import UserSerializer, GroupSerializer, ImageFileSerializer, DirectorySerializer, ParameterSerializer
import api.serializers as api_ser
# Authentication: https://simpleisbetterthancomplex.com/tutorial/2018/11/22/how-to-implement-token-authentication-using-django-rest-framework.html
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView


# Create your views here.

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

class ImageViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    permission_classes = (IsAuthenticated,)

    queryset = ImageFile.objects.all()
    serializer_class = api_ser.ImageFileSerializer

class DirectoryViewSet(viewsets.ModelViewSet):

	permission_classes = (IsAuthenticated,)

	queryset = Directory.objects.all()
	serializer_class = api_ser.DirectorySerializer
        
class PersonViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,) 

    queryset = Person.objects.all()
    
    serializer_class = api_ser.PersonSerializer

class FaceViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,) 

    queryset = Face.objects.all()
    serializer_class = api_ser.FaceSerializer


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

