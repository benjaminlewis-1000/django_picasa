"""Tests for the project-level (non-app) pieces of picasa: the DRF
exception handler, the allauth adapter, and a handful of settings-as-code
sanity checks that guard against an accidental security regression (e.g.
someone flips CORS_ALLOW_ALL_ORIGINS to True while debugging and forgets
to revert). Not an installed app -- run with:
    manage.py test picasa.tests
"""
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, TestCase
from rest_framework import exceptions
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory

from picasa.adapters import SubdomainRedirectAdapter
from picasa.exceptions import api_redirect_to_login_handler


class ApiRedirectToLoginHandlerTests(TestCase):
    def _drf_request(self, accept="application/json"):
        django_request = APIRequestFactory().get("/api/images/", HTTP_ACCEPT=accept)
        django_request.user = AnonymousUser()
        return Request(django_request)

    def test_json_client_gets_normal_401_response(self):
        request = self._drf_request(accept="application/json")
        response = api_redirect_to_login_handler(
            exceptions.NotAuthenticated(), {"request": request}
        )
        self.assertEqual(response.status_code, 401)

    def test_browser_client_gets_redirected_to_login(self):
        request = self._drf_request(accept="text/html,application/xhtml+xml")
        response = api_redirect_to_login_handler(
            exceptions.NotAuthenticated(), {"request": request}
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn(settings.LOGIN_URL, response.url)
        self.assertIn("next=", response.url)

    def test_permission_denied_also_redirects_for_browser_client(self):
        request = self._drf_request(accept="text/html")
        response = api_redirect_to_login_handler(
            exceptions.PermissionDenied(), {"request": request}
        )
        self.assertEqual(response.status_code, 302)

    def test_other_exception_types_are_not_redirected(self):
        # A validation error from a browser client should still come back
        # as a normal DRF error response, not a login redirect -- only
        # NotAuthenticated/PermissionDenied should trigger the redirect.
        request = self._drf_request(accept="text/html")
        response = api_redirect_to_login_handler(
            exceptions.ValidationError("bad input"), {"request": request}
        )
        self.assertEqual(response.status_code, 400)


class SubdomainRedirectAdapterTests(TestCase):
    def setUp(self):
        self.adapter = SubdomainRedirectAdapter()
        self.factory = RequestFactory()
        from django.contrib.auth.models import User

        self.user = User.objects.create_user(username="adaptertest", password="pw123456")

    def _authenticated_request(self, *args, **kwargs):
        request = self.factory.get(*args, **kwargs)
        request.user = self.user
        return request

    def test_next_param_to_trusted_frontend_domain_is_honored(self):
        request = self.factory.get(
            "/accounts/oidc/authelia/login/callback/",
            {"next": "https://facewire.exploretheworld.tech/some/path"},
        )
        result = self.adapter.get_login_redirect_url(request)
        self.assertEqual(result, "https://facewire.exploretheworld.tech/some/path")

    def test_next_param_to_untrusted_domain_is_rejected(self):
        # Open-redirect guard: a `next` pointing anywhere other than the
        # trusted frontend domain must NOT be honored verbatim. Falls
        # through to DefaultAccountAdapter, which needs an authenticated
        # request.user.
        request = self._authenticated_request(
            "/accounts/oidc/authelia/login/callback/",
            {"next": "https://evil.example.com/phish"},
        )
        result = self.adapter.get_login_redirect_url(request)
        self.assertNotEqual(result, "https://evil.example.com/phish")

    def test_missing_next_param_falls_back_to_default(self):
        request = self._authenticated_request("/accounts/oidc/authelia/login/callback/")
        # Should not raise, and should not be None -- falls through to
        # DefaultAccountAdapter's own default redirect logic.
        result = self.adapter.get_login_redirect_url(request)
        self.assertIsNotNone(result)


class SettingsSanityTests(TestCase):
    """Regression guards against accidentally loosened security settings."""

    def test_cors_does_not_allow_all_origins(self):
        self.assertFalse(settings.CORS_ALLOW_ALL_ORIGINS)

    def test_api_default_permission_requires_authentication(self):
        self.assertIn(
            "rest_framework.permissions.IsAuthenticated",
            settings.REST_FRAMEWORK["DEFAULT_PERMISSION_CLASSES"],
        )

    def test_api_uses_custom_exception_handler(self):
        self.assertEqual(
            settings.REST_FRAMEWORK["EXCEPTION_HANDLER"],
            "picasa.exceptions.api_redirect_to_login_handler",
        )

    def test_debug_is_off(self):
        # This app is internet-facing (Authelia/Tailscale-fronted); DEBUG=True
        # would leak stack traces/settings to anyone who can trigger a 500.
        self.assertFalse(settings.DEBUG)

    def test_jwt_signing_key_is_the_django_secret_key(self):
        self.assertEqual(settings.SIMPLE_JWT["SIGNING_KEY"], settings.SECRET_KEY)
