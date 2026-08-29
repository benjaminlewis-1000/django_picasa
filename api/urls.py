
from django.conf.urls import include
from rest_framework import routers
from api import views
from api import mobile_views
from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token  # <-- Here
from rest_framework_simplejwt import views as jwt_views
from django.views.generic import RedirectView

# app_name = 'api'

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
    # path('request-token/', obtain_auth_token, name='api_token_auth'),  
    path(r'token/obtain/', views.TokenPairWithUsername.as_view(), name='token_create'),  # override sjwt stock token
    path(r'token/obtain', RedirectView.as_view(url = '/token/obtain/', permanent=True), name='token_create'),
    path(r'token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
    # url(r'^', include(router.urls)),
    path('', include(router.urls)),
    path(r'image_list/', views.filteredImagesView.as_view(), name='image_list'),
    path(r'paginate_obj_ids/<int:id>/<slug:field>', views.PersonParamView.as_view(), name='face_pages'),
    path(r'person_list/', views.PersonListView.as_view(), name='person_list'),
    path(r'folder_list/', views.FolderListView.as_view(), name='folder_list'),
    path(r'keyed_image/<slug:type>/', views.KeyedImageView.as_view(), name='keyed_image'),
    path(r'authelia_state/', views.AutheliaStateView.as_view(), name='authelia_state'),
    path(r'clean_logout/', views.CleanLogoutView.as_view(), name='clean_logout'),
    path(r'mobile/confident_unlabeled/', mobile_views.ConfidentUnlabeledView.as_view(), name='unlabeled'),
    path(r'mobile/labeling_groups/', mobile_views.LabelingGroupsView.as_view(), name='labeling_groups'),
    path(r'mobile/unlabeled_instance/<int:id>/', mobile_views.UnlabeledMobileInfo.as_view(), name='unlabeled_instances'),
    path(r'mobile/reset/<int:id>/', mobile_views.ResetFace.as_view(), name='reset'),
    path(r'mobile/name_list/', mobile_views.MobileNameList.as_view(), name='name_list'),
    path(r'mobile/ignore_candidates/', mobile_views.IgnoreCandidatesList.as_view(), name='ignore_candidates'),
    path(r'mobile/bulk_confirm_ignore/', mobile_views.BulkConfirmIgnore.as_view(), name='bulk_confirm_ignore'),
    path(r'mobile/verify_candidates/', mobile_views.VerifyCandidatesList.as_view(), name='verify_candidates'),
    path(r'mobile/verify_ignore_candidates/', mobile_views.VerifyIgnoreCandidatesList.as_view(), name='verify_ignore_candidates'),
    path(r'mobile/bulk_verify/', mobile_views.BulkVerify.as_view(), name='bulk_verify'),
]
