from django.urls import path
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path('', views.index ),
    # path('', TemplateView.as_view(template_name='templates/frontend/index.html'))
]
