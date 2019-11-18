from django.conf.urls import url, include

from django.conf.urls import url, include
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'mygroups', views.GroupViewSet)
router.register(r'images', views.ImageViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
]