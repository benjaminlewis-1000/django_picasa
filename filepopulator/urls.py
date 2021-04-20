from django.urls import path
from filepopulator import views

urlpatterns = [
    path('', views.index, name='index'),
]