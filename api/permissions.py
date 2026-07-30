# permissions.py
from rest_framework import permissions
from django.conf import settings

class HasSlideshowKeyOrAuthenticated(permissions.BasePermission):
    def has_permission(self, request, view):
        # 1. Allow standard Authelia-authenticated session users
        if request.user and request.user.is_authenticated:
            return True
        
        # 2. Check for custom header: X-Slideshow-Key -> HTTP_X_SLIDESHOW_KEY
        provided_key = request.META.get('HTTP_X_SLIDESHOW_KEY')
        
        # Fallback: check query parameter just in case (useful for direct raw <img> URLs)
        if not provided_key:
            provided_key = request.query_params.get('key')

        return provided_key == getattr(settings, 'SLIDESHOW_API_KEY', None)