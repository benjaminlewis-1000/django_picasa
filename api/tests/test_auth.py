"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_auth
"""
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import unittest
import zipfile
from datetime import datetime, time
from datetime import timezone as dt_timezone
from io import BytesIO
from unittest import mock

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils.functional import SimpleLazyObject
from rest_framework import status
from rest_framework.test import APIClient

from api.models import ScheduleRule, SlideshowConfig, resolve_active_slideshow_config
from api.throttles import AccessKeyMediaThrottle
from face_manager.models import Face, Person
from filepopulator.models import GeocodeCache, ImageFile, Directory, VideoFile
from filepopulator.scripts import create_image_file

from api.tests import FaceFixtureMixin, ApiTestCase, _tiny_jpeg_bytes, ensure_sentinel_people


class LazySentinelPersonTests(ApiTestCase):
    """Regression test for a fixed bug: api/views.py used to run
    Person.objects.filter(person_name='.ignore')[0] etc as plain
    module-level queries -- executed at import time (URL resolution,
    Django's system checks), before any test's setUp/sentinel seeding has
    run. That crashed the entire app on any DB without those rows
    already present, which is every fresh install and every CI run (only
    production avoided it, because someone seeded them by hand once).
    Now they're SimpleLazyObject-wrapped so the query is deferred to
    first actual use."""

    def test_sentinel_people_are_lazy_not_eagerly_queried(self):
        import api.views as api_views
        self.assertIsInstance(api_views.soft_ignore_person, SimpleLazyObject)
        self.assertIsInstance(api_views.hard_ignore_person, SimpleLazyObject)
        self.assertIsInstance(api_views.blank_person, SimpleLazyObject)

    def test_sentinel_people_resolve_correctly_on_first_access(self):
        import api.views as api_views
        ignore = Person.objects.get(person_name=".ignore")
        realignore = Person.objects.get(person_name=".realignore")
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.assertEqual(api_views.soft_ignore_person.id, ignore.id)
        self.assertEqual(api_views.hard_ignore_person.id, realignore.id)
        self.assertEqual(api_views.blank_person.id, blank.id)


class AuthenticationTests(ApiTestCase):
    def test_anonymous_request_to_protected_viewset_is_rejected(self):
        resp = self.anon_client.get("/api/images/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_authenticated_request_to_protected_viewset_succeeds(self):
        resp = self.client.get("/api/images/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


class SlideshowKeyPermissionTests(ApiTestCase):
    """Covers api/permissions.py HasSlideshowKeyOrAuthenticated, used by
    filteredImagesView and ParameterViewSet."""

    def test_no_key_and_unauthenticated_is_rejected(self):
        resp = self.anon_client.get("/api/image_list/")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_wrong_key_header_is_rejected(self):
        resp = self.anon_client.get("/api/image_list/", HTTP_X_SLIDESHOW_KEY="not-the-key")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_correct_key_header_is_accepted(self):
        resp = self.anon_client.get(
            "/api/image_list/", HTTP_X_SLIDESHOW_KEY=settings.SLIDESHOW_API_KEY
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_correct_key_query_param_is_accepted(self):
        # Regression test for a fixed bug: when query params were present
        # but none of them were 'people'/'year_start'/'year_end' (e.g. just
        # '?key=...'), `p_query` was left as None and
        # `ImageFile.objects.filter(p_query)` raised TypeError instead of
        # falling back to "all images" the way the no-params case does.
        # The header-based key auth path (tested above) doesn't hit this
        # because it sends no query params at all.
        resp = self.anon_client.get(f"/api/image_list/?key={settings.SLIDESHOW_API_KEY}")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_authenticated_session_bypasses_key_requirement(self):
        resp = self.client.get("/api/image_list/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


@override_settings(
    AUTHELIA_ISSUER="https://auth.test.example",
    AUTHELIA_MOBILE_CLIENT_ID="photoverify_mobile",
)
class AutheliaOIDCAuthenticationTests(TestCase):
    """api/authentication.py:AutheliaOIDCAuthentication -- the PhotoVerify
    mobile app sends an Authelia-issued OIDC ID token as a Bearer token;
    this class verifies its RS256 signature against Authelia's JWKS and
    maps the email claim to a local user. Here the JWKS lookup is stubbed
    with a locally generated RSA keypair so no network / real Authelia is
    involved."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ensure_sentinel_people()
        from cryptography.hazmat.primitives.asymmetric import rsa

        cls._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def setUp(self):
        import time
        from unittest.mock import patch

        self._time = time
        self.user = User.objects.create_user(
            username="mobileuser", email="Person@Example.com", password="pw123456"
        )
        # get_signing_key_from_jwt(token) -> object with a .key attribute
        # (the verifying key). Return the public half of our test keypair.
        self._signing_key = type("K", (), {"key": self._key.public_key()})()
        self._jwks_patcher = patch(
            "api.authentication._jwks_client.get_signing_key_from_jwt",
            return_value=self._signing_key,
        )
        self._jwks_mock = self._jwks_patcher.start()
        self.addCleanup(self._jwks_patcher.stop)

        # The last-known-good key cache is a module global; isolate tests.
        import api.authentication as _authmod
        _authmod._last_good_keys.clear()
        self.addCleanup(_authmod._last_good_keys.clear)

        self.client = APIClient()

    def _token(self, **overrides):
        import jwt

        now = int(self._time.time())
        claims = {
            "iss": "https://auth.test.example",
            "aud": "photoverify_mobile",
            "sub": "abc123",
            "iat": now,
            "exp": now + 3600,
            "email": "person@example.com",
        }
        claims.update(overrides)
        for k in [k for k, v in overrides.items() if v is None]:
            claims.pop(k, None)
        return jwt.encode(claims, self._key, algorithm="RS256")

    def _get(self, token):
        return self.client.get(
            "/api/mobile/name_list/", HTTP_AUTHORIZATION=f"Bearer {token}"
        )

    def test_valid_token_authenticates_and_maps_email_case_insensitively(self):
        resp = self._get(self._token())
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    REJECTED = (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

    def test_no_bearer_header_is_rejected_not_500(self):
        resp = self.client.get("/api/mobile/name_list/")
        self.assertIn(resp.status_code, self.REJECTED)

    def test_garbage_bearer_value_is_rejected_not_500(self):
        # A bearer value that isn't a JWT at all -- get_signing_key_from_jwt
        # raises jwt.DecodeError parsing the header; must not 500.
        for junk in ["not.a.jwt", "garbage", "a.b.c"]:
            resp = self._get(junk)
            self.assertIn(resp.status_code, self.REJECTED, junk)

    def test_expired_token_rejected(self):
        now = int(self._time.time())
        resp = self._get(self._token(iat=now - 7200, exp=now - 3600))
        self.assertIn(resp.status_code, self.REJECTED)
        self.assertIn("expired", resp.content.decode().lower())

    def test_wrong_audience_rejected(self):
        resp = self._get(self._token(aud="some_other_client"))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_wrong_issuer_rejected(self):
        resp = self._get(self._token(iss="https://evil.example"))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_token_signed_by_a_different_key_rejected(self):
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa

        attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = int(self._time.time())
        forged = jwt.encode(
            {
                "iss": "https://auth.test.example",
                "aud": "photoverify_mobile",
                "sub": "abc123",
                "iat": now,
                "exp": now + 3600,
                "email": "person@example.com",
            },
            attacker_key,
            algorithm="RS256",
        )
        self.assertIn(self._get(forged).status_code, self.REJECTED)

    def test_unknown_email_rejected(self):
        resp = self._get(self._token(email="nobody@example.com"))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_missing_email_claim_rejected(self):
        resp = self._get(self._token(email=None))
        self.assertIn(resp.status_code, self.REJECTED)

    def test_inactive_user_rejected(self):
        self.user.is_active = False
        self.user.save()
        resp = self._get(self._token())
        self.assertIn(resp.status_code, self.REJECTED)

    def test_jwks_fetch_failure_returns_503_not_401(self):
        # A transient JWKS outage must NOT look like a bad token -- 401
        # would make the app throw away a perfectly valid session.
        import jwt as _jwt

        self._jwks_mock.side_effect = _jwt.PyJWKClientError("boom")
        resp = self._get(self._token())
        self.assertEqual(resp.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)

    def test_jwks_fetch_failure_falls_back_to_last_known_good_key(self):
        import jwt as _jwt

        # 1st call succeeds -> caches the key.
        self.assertEqual(self._get(self._token()).status_code, status.HTTP_200_OK)
        # JWKS now unreachable, but we've seen this key before -> still 200.
        self._jwks_mock.side_effect = _jwt.PyJWKClientError("boom")
        self.assertEqual(self._get(self._token()).status_code, status.HTTP_200_OK)


class ParameterCredentialExposureTests(ApiTestCase):
    """Regression test for a real pre-fix vulnerability: ParameterViewSet
    used to hand img_server_user/img_server_pw (Apache's real media Basic
    Auth credentials) to ANY caller holding the slideshow key - and that
    key is unavoidably public (baked into net_slideshow's frontend
    bundle). Real credentials must only ever reach a genuine
    authenticated session."""

    def test_slideshow_key_alone_gets_null_media_credentials(self):
        resp = self.anon_client.get(
            '/api/parameters/', HTTP_X_SLIDESHOW_KEY=settings.SLIDESHOW_API_KEY
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIsNone(resp.data['img_server_user'])
        self.assertIsNone(resp.data['img_server_pw'])
        # The low-stakes nonce is still fine to hand out - that's what
        # KeyedImageView/KeyedVideoView actually gate on.
        self.assertTrue(resp.data['random_access_key'])

    def test_authenticated_session_still_gets_real_media_credentials(self):
        resp = self.client.get('/api/parameters/')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['img_server_user'], settings.MEDIA_URL_USER)
        self.assertEqual(resp.data['img_server_pw'], settings.MEDIA_URL_PW)


class CsrfCookieViewTests(ApiTestCase):
    """GET /api/csrf/ - exists solely to make Django issue a csrftoken
    cookie, since nothing else in this project ever does (see the
    view's own docstring). Without it, SlideshowConfigViewSet's
    POST/PUT 403s with a missing-CSRF-cookie error that's invisible
    until someone actually tries to save."""

    def test_anonymous_request_still_gets_a_csrf_cookie(self):
        resp = self.anon_client.get('/api/csrf/')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('csrftoken', resp.cookies)

    def test_authenticated_request_also_gets_a_csrf_cookie(self):
        resp = self.client.get('/api/csrf/')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('csrftoken', resp.cookies)


