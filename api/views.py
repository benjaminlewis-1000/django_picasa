from django.shortcuts import render

from django.contrib.auth.models import User, Group
from filepopulator.models import ImageFile
from rest_framework import viewsets
from api.serializers import UserSerializer, GroupSerializer, ImageFileSerializer

# Create your views here.


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer

class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

class ImageViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = ImageFile.objects.all()
    serializer_class = ImageFileSerializer