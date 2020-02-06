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
router.register(r'directories', views.DirectoryViewSet)
# router.register(r'hello', views.HelloView)
# router.register(r'request-token', obtain_auth_token)

# TODO: Add a 'Help' section or something. 


# Requesting a token: https://simpleisbetterthancomplex.com/tutorial/2018/11/22/how-to-implement-token-authentication-using-django-rest-framework.html#user-requesting-a-token
# http post https://picasa.exploretheworld.tech/api/request-token/ username=benjamin password=********
urlpatterns = [
    path('request-token/', obtain_auth_token, name='api_token_auth'),  
    url(r'^', include(router.urls)),
    # path(r'hello/', views.HelloView.as_view(), name='hello'),
]
