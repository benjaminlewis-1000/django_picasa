from rest_framework.views import exception_handler
from rest_framework import exceptions
from django.shortcuts import redirect
from django.conf import settings

def api_redirect_to_login_handler(exc, context):
    # Let DRF handle the exception first to see what kind of error it is
    response = exception_handler(exc, context)

    # If it's a 401 Not Authenticated or 403 Permission Denied error
    if isinstance(exc, (exceptions.NotAuthenticated, exceptions.PermissionDenied)):
        request = context['request']
        
        # Check if the request is coming from a standard web browser browser look
        # (This ensures background scripts or your React code still get normal JSON errors)
        if 'text/html' in request.META.get('HTTP_ACCEPT', ''):
            login_url = getattr(settings, 'LOGIN_URL', '/accounts/oidc/authelia/login/')
            
            # Build the return path so Authelia drops them right back onto this specific API endpoint
            full_path = request.get_full_path()
            return redirect(f"{login_url}?next={full_path}")

    return response