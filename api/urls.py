from django.conf.urls import url, include

from django.conf.urls import url, include
from rest_framework import routers
from api import views
from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token  # <-- Here

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'mygroups', views.GroupViewSet)
router.register(r'images', views.ImageViewSet)
# router.register(r'hello', views.HelloView)

urlpatterns = [
    url(r'^', include(router.urls)),
    path('request-token/', obtain_auth_token, name='api_token_auth'),  
    # path(r'hello/', views.HelloView.as_view(), name='hello'),
]