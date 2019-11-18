#! /usr/bin/env python

from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import viewsets

# Create your views here.

# from .forms import ImageFileForm, DirectoryForm



def index(request):
    return HttpResponse("Hello, world. You're at the file populator index.")
