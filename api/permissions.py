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

        expected_key = getattr(settings, 'SLIDESHOW_API_KEY', None)

        # # DIAGNOSTIC PRINTS - Check your docker logs!
        # print(f"--- DEBUG SLIDESHOW AUTH ---")
        # print(f"Header Key received: {provided_key}")
        # print(f"Expected Key in Settings: {expected_key}")
        
        if expected_key and provided_key == expected_key:
            return True

        return False #  provided_key == getattr(settings, 'SLIDESHOW_API_KEY', None)