"""picasa URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from django.conf.urls.static import static
from picasa import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('periodic/', include('periodic.urls')),
    # path('filepopulator/', include('filepopulator.urls')),
    # path('face_manager/', include('face_manager.urls')),
    path(r'api/', include('api.urls')),
    path(r'', views.index),
    path(r'index.html', views.index),
]

# https://stackoverflow.com/questions/5517950/django-media-url-and-media-root
# urlpatterns += staticfiles_urlpatterns()
# print(settings.STATIC_URL)
# urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# urlpatterns += path(r'^', views.index),

# from django.conf.urls import (
#     handler400, handler403, handler404, handler500
# )

# handler400 = views.bad_request
# handler403 = views.permission_denied
# handler404 = views.page_not_found
# handler500 = views.server_error
