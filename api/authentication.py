#! /usr/bin/env python

# Bearer-token authentication for the PhotoVerify mobile app.
#
# The web frontend authenticates through Authelia -> allauth OIDC ->
# Django session cookie (see picasa/adapters.py). A native mobile app
# can't ride a shared browser session cookie, so instead it runs its own
# OIDC Authorization Code + PKCE flow directly against Authelia (public
# client "photoverify_mobile"), and sends the resulting OIDC ID token on
# every API call as:
#
#     Authorization: Bearer <id_token>
#
# This class validates that token (RS256 signature against Authelia's
# JWKS, plus issuer/audience/expiry) and maps its email claim to a local
# Django user -- the same "trust the email from Authelia" rule allauth
# uses via SOCIALACCOUNT_EMAIL_AUTHENTICATION. It does NOT auto-create
# users: the account must already exist (provisioned by a prior web SSO
# login), so this stays a pure authentication path with no side effects.
#
# The legacy rest_framework_simplejwt path (/api/token/obtain/, header
# type "JWT ") is left in place and untouched as a fallback.

import logging

import jwt
from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions

logger = logging.getLogger("__main__")

# One JWKS client process-wide. PyJWKClient keeps an in-memory cache of
# fetched signing keys (cache_keys=True) so we're not hitting Authelia's
# /jwks.json on every request; lifespan bounds how long a cached key is
# trusted before a re-fetch (picks up key rotation).
# auth.exploretheworld.tech sits behind Cloudflare, which 403s requests
# carrying the default "Python-urllib/x.y" User-Agent that PyJWKClient's
# urllib fetch would otherwise send. Give it a browser-ish UA.
_jwks_client = jwt.PyJWKClient(
    settings.AUTHELIA_JWKS_URL,
    cache_keys=True,
    lifespan=settings.AUTHELIA_JWKS_CACHE_SECONDS,
    headers={"User-Agent": "django_picasa/AutheliaOIDCAuthentication"},
)


class AutheliaOIDCAuthentication(authentication.BaseAuthentication):
    keyword = "Bearer"

    def authenticate(self, request):
        auth = authentication.get_authorization_header(request).split()

        if not auth or auth[0].lower() != self.keyword.lower().encode():
            # No Bearer token -- let the other authenticators (session,
            # DRF token, simplejwt "JWT ") have their turn.
            return None

        if len(auth) == 1:
            raise exceptions.AuthenticationFailed(
                "Invalid bearer header: no credentials provided."
            )
        if len(auth) > 2:
            raise exceptions.AuthenticationFailed(
                "Invalid bearer header: token string should not contain spaces."
            )

        token = auth[1].decode()
        claims = self._decode(token)
        user = self._user_from_claims(claims)
        return (user, claims)

    def authenticate_header(self, request):
        # Makes DRF return 401 (not 403) for a missing/blown token, so the
        # app knows to refresh or re-login.
        return self.keyword

    def _decode(self, token):
        try:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError as exc:
            logger.warning("Authelia JWKS lookup failed: %s", exc)
            raise exceptions.AuthenticationFailed("Could not verify token signing key.")
        except jwt.InvalidTokenError as exc:
            # Bearer value isn't a well-formed JWT at all (get_signing_key_
            # from_jwt has to parse the header to find the kid).
            logger.warning("Malformed bearer token: %s", exc)
            raise exceptions.AuthenticationFailed("Malformed token.")

        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=settings.AUTHELIA_MOBILE_CLIENT_ID,
                issuer=settings.AUTHELIA_ISSUER,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Token has expired.")
        except jwt.InvalidAudienceError:
            raise exceptions.AuthenticationFailed("Token audience mismatch.")
        except jwt.InvalidIssuerError:
            raise exceptions.AuthenticationFailed("Token issuer mismatch.")
        except jwt.InvalidTokenError as exc:
            logger.warning("Rejected Authelia bearer token: %s", exc)
            raise exceptions.AuthenticationFailed("Invalid token.")

    def _user_from_claims(self, claims):
        email = (claims.get("email") or "").strip()
        if not email:
            raise exceptions.AuthenticationFailed(
                "Token has no email claim; cannot map to a local account."
            )

        User = get_user_model()
        user = User.objects.filter(email__iexact=email).first()
        if user is None:
            raise exceptions.AuthenticationFailed(
                f"No local account for {email}. Log in to the web app once first."
            )
        if not user.is_active:
            raise exceptions.AuthenticationFailed("Local account is disabled.")
        return user
