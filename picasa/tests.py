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

    def test_next_param_to_any_exploretheworld_subdomain_is_honored(self):
        # Not just the one hardcoded 'facewire' subdomain -- any real
        # exploretheworld.tech subdomain should be trusted.
        request = self.factory.get(
            "/accounts/oidc/authelia/login/callback/",
            {"next": "https://slideshow.exploretheworld.tech/some/path"},
        )
        result = self.adapter.get_login_redirect_url(request)
        self.assertEqual(result, "https://slideshow.exploretheworld.tech/some/path")

    def test_next_param_with_domain_name_only_in_query_string_is_rejected(self):
        # Regression test for a fixed open-redirect bug: the old check was
        # `'facewire.exploretheworld.tech' in next_param`, plain substring
        # containment -- so a malicious host with the trusted domain name
        # merely present somewhere in the URL (e.g. a query string) passed
        # the check and would have redirected a freshly-authenticated
        # user's browser to an attacker-controlled host. Now the actual
        # parsed hostname is validated, not the raw string.
        request = self._authenticated_request(
            "/accounts/oidc/authelia/login/callback/",
            {"next": "https://evil.example/?x=facewire.exploretheworld.tech"},
        )
        result = self.adapter.get_login_redirect_url(request)
        self.assertNotEqual(result, "https://evil.example/?x=facewire.exploretheworld.tech")

    def test_next_param_relative_path_is_honored(self):
        # A relative path has no host to spoof -- safe by construction,
        # and allauth commonly passes these (e.g. '/accounts/profile/').
        request = self.factory.get(
            "/accounts/oidc/authelia/login/callback/",
            {"next": "/some/relative/path"},
        )
        result = self.adapter.get_login_redirect_url(request)
        self.assertEqual(result, "/some/relative/path")

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

    def test_session_cookie_renews_on_activity(self):
        # Without this, SESSION_COOKIE_AGE is a fixed lifetime from login
        # (browser deletes the cookie 30 min later regardless of activity),
        # not a sliding inactivity timeout - see picasa/settings.py's
        # comment on SESSION_COOKIE_AGE for the full explanation.
        self.assertTrue(settings.SESSION_SAVE_EVERY_REQUEST)


class BlockScannerPathsMiddlewareTests(TestCase):
    """Real incident, 2026-10-05: see block_scanners.py's own comment.

    Exercises the middleware directly (RequestFactory + a sentinel
    get_response) rather than through the full URL dispatch -- this app
    requires auth on every real view by default (see
    SettingsSanityTests.test_api_default_permission_requires_authentication
    above), so going through self.client would conflate "my middleware
    blocked this" with "DRF's own permission check blocked this."
    """

    def setUp(self):
        from picasa.block_scanners import BlockScannerPathsMiddleware
        self.factory = RequestFactory()
        self.sentinel_called = False

        def sentinel_get_response(request):
            self.sentinel_called = True
            from django.http import HttpResponse
            return HttpResponse('ok')

        self.middleware = BlockScannerPathsMiddleware(sentinel_get_response)

    def test_php_style_paths_are_rejected_without_reaching_the_view(self):
        for path in ('/wp-signup.php', '/xmlrpc.php', '/adminfuns.PHP', '/foo.php7'):
            self.sentinel_called = False
            response = self.middleware(self.factory.get(path))
            self.assertEqual(response.status_code, 403, path)
            self.assertFalse(self.sentinel_called, path)

    def test_ordinary_paths_reach_the_real_view(self):
        response = self.middleware(self.factory.get('/api/'))
        self.assertTrue(self.sentinel_called)
        self.assertEqual(response.status_code, 200)


class MallocTrimMiddlewareTests(TestCase):
    """Real incident, 2026-10-05: see malloc_trim_middleware.py's own
    comment -- real mobile-app traffic against api/views.py's
    KeyedImageView (face_array/face_source) accumulates native-buffer
    memory the same way the batch face-extraction pipeline did."""

    def test_calls_get_response_and_returns_its_response_unchanged(self):
        from picasa.malloc_trim_middleware import MallocTrimMiddleware
        from django.http import HttpResponse

        sentinel_response = HttpResponse('ok', status=201)
        calls = []

        def get_response(request):
            calls.append(request)
            return sentinel_response

        middleware = MallocTrimMiddleware(get_response)
        request = RequestFactory().get('/anything/')
        response = middleware(request)

        self.assertEqual(len(calls), 1)
        self.assertIs(response, sentinel_response)
        self.assertEqual(response.status_code, 201)

    def test_survives_a_missing_libc_gracefully(self):
        from picasa import malloc_trim_middleware

        original_libc = malloc_trim_middleware._libc
        malloc_trim_middleware._libc = None
        try:
            middleware = malloc_trim_middleware.MallocTrimMiddleware(
                lambda request: __import__('django.http', fromlist=['HttpResponse']).HttpResponse('ok')
            )
            response = middleware(RequestFactory().get('/anything/'))
            self.assertEqual(response.status_code, 200)
        finally:
            malloc_trim_middleware._libc = original_libc
