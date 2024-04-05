from django.urls import path
from face_manager import views

urlpatterns = [
    path('', views.index, name='index'),
]