from django.conf.urls import url, include

from django.conf.urls import url, include
from rest_framework import routers
from api import views
from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token  # <-- Here
from rest_framework_simplejwt import views as jwt_views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'mygroups', views.GroupViewSet)
router.register(r'images', views.ImageViewSet)
router.register(r'directories', views.DirectoryViewSet)
router.register(r'faces', views.FaceViewSet)
router.register(r'people', views.PersonViewSet)
router.register(r'parameters', views.ParameterViewSet, basename='params')
router.register(r'server_stats', views.StatsViewSet, basename='stats')
# router.register(r'request-token', obtain_auth_token)

# TODO: Add a 'Help' section or something. 


# Requesting a token: https://simpleisbetterthancomplex.com/tutorial/2018/11/22/how-to-implement-token-authentication-using-django-rest-framework.html#user-requesting-a-token
# http post https://picasa.exploretheworld.tech/api/request-token/ username=benjamin password=********
urlpatterns = [
    path('request-token/', obtain_auth_token, name='api_token_auth'),  
    path('token/obtain/', views.TokenPairWithUsername.as_view(), name='token_create'),  # override sjwt stock token
    path('token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
    # url(r'^', include(router.urls)),
    path('', include(router.urls)),
    path(r'image_list/', views.filteredImagesView.as_view(), name='image_list'),
    path(r'paginate_obj_ids/<int:id>/<slug:field>', views.PersonParamView.as_view(), name='face_pages'),
    path(r'keyed_image/<slug:type>/', views.KeyedImageView.as_view(), name='keyed_image')
]
