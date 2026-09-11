"""API views backing the frontend's Tools-tab "Google Photos" screen --
lets a user track a small set of named Google Photos albums and pull in
newly-picked items via the Picker API.

Kept in its own module rather than api/views.py - same reasoning as
mobile_views.py/geocode_views.py being separate.

There's no watch/webhook/scheduled-poll mechanism here on purpose: Google's
Picker API requires the user to manually reselect an album's contents every
time (see api/google_photos_client.py's module docstring and CLAUDE.md) -
every one of these endpoints is only ever called in direct response to a
frontend button click, not a periodic task.
"""

import os
import secrets
from urllib.parse import quote

from django.conf import settings
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from api.google_photos_client import GooglePhotosError, build_authorization_url, create_session, \
    delete_session, download_media_item, exchange_code_for_refresh_token, get_session, list_media_items
from api.models import GooglePhotosCredential, GooglePhotosSyncedItem, GooglePhotosWatchedAlbum
from api.upload_views import _chown_upload_path, _unique_destination


def _frontend_tools_url():
    return f'https://{settings.FRONTEND_DOMAIN}/faces'


class GooglePhotosCredentialView(APIView):
    """GET/POST /api/google_photos/credentials/ -- backs the Tools tab's
    "Connect Google Photos" panel. The client secret is write-only over
    this API (never echoed back in GET) - only whether one is configured."""

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        creds = GooglePhotosCredential.load()
        return Response({
            'client_id': creds.client_id,
            'configured': bool(creds.client_id and creds.client_secret),
            'connected': bool(creds.refresh_token),
        })

    def post(self, request):
        client_id = (request.data.get('client_id') or '').strip()
        client_secret = (request.data.get('client_secret') or '').strip()
        if not client_id or not client_secret:
            return Response({'error': 'Both "client_id" and "client_secret" are required.'}, status=400)

        creds = GooglePhotosCredential.load()
        creds.client_id = client_id
        creds.client_secret = client_secret
        # A changed client invalidates any existing refresh token - it was
        # minted against whatever the previous client_id/secret were.
        creds.refresh_token = ''
        creds.save(update_fields=['client_id', 'client_secret', 'refresh_token', 'updated_at'])
        return Response({'client_id': creds.client_id, 'configured': True, 'connected': False})


class GooglePhotosOAuthStartView(APIView):
    """GET /api/google_photos/oauth/start/ -- the frontend navigates the
    whole browser here directly (window.location, not axios/XHR) to begin
    the "Connect Google Photos" flow; there's nothing to return but a
    redirect to Google's consent screen. `state` is round-tripped through
    the Django session (already cookie-based in this app) and re-checked
    in the callback below as basic CSRF protection on the flow."""

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        state = secrets.token_urlsafe(32)
        request.session['google_photos_oauth_state'] = state
        try:
            url = build_authorization_url(state)
        except GooglePhotosError as e:
            return HttpResponseRedirect(f'{_frontend_tools_url()}?google_photos_error={quote(str(e))}')
        return HttpResponseRedirect(url)


class GooglePhotosOAuthCallbackView(APIView):
    """GET /api/google_photos/oauth/callback/ -- Google redirects here
    after the user approves (or denies) access. Exchanges the code for a
    refresh token, stores it, and bounces the browser back to the
    frontend's Tools tab with a query param the UI reads to show a
    success/error banner (see picasaScreen.jsx)."""

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        error = request.GET.get('error')
        if error:
            return HttpResponseRedirect(f'{_frontend_tools_url()}?google_photos_error={quote(error)}')

        state = request.GET.get('state')
        expected_state = request.session.pop('google_photos_oauth_state', None)
        if not state or state != expected_state:
            return HttpResponseRedirect(f'{_frontend_tools_url()}?google_photos_error=invalid_state')

        try:
            exchange_code_for_refresh_token(request.GET.get('code'))
        except GooglePhotosError as e:
            return HttpResponseRedirect(f'{_frontend_tools_url()}?google_photos_error={quote(str(e))}')

        return HttpResponseRedirect(f'{_frontend_tools_url()}?google_photos_connected=1')


class WatchedAlbumListView(APIView):
    """GET/POST /api/google_photos/watch/ -- list watched albums (with
    item counts) / add a new one. `title` is purely user-typed - Google's
    Picker API returns no album/sharer metadata to auto-fill it with."""

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        albums = GooglePhotosWatchedAlbum.objects.filter(is_active=True).order_by('title')
        return Response([_serialize_album(a) for a in albums])

    def post(self, request):
        title = (request.data.get('title') or '').strip()
        if not title:
            return Response({'error': 'A "title" field is required.'}, status=400)
        album = GooglePhotosWatchedAlbum.objects.create(title=title)
        return Response(_serialize_album(album), status=201)


class WatchedAlbumDetailView(APIView):
    """DELETE /api/google_photos/watch/<id>/ -- stop watching. Leaves
    already-downloaded files and their GooglePhotosSyncedItem rows alone;
    this only removes the album from the active list."""

    permission_classes = (IsAuthenticated,)

    def delete(self, request, album_id):
        album = get_object_or_404(GooglePhotosWatchedAlbum, id=album_id)
        album.is_active = False
        album.save(update_fields=['is_active'])
        return Response(status=204)


class WatchedAlbumSessionInitView(APIView):
    """POST /api/google_photos/watch/<id>/session/init/ -- starts a new
    Picker session for this album. Returns {session_id, picker_uri}; the
    frontend opens picker_uri in a new tab for the user to select items in."""

    permission_classes = (IsAuthenticated,)

    def post(self, request, album_id):
        get_object_or_404(GooglePhotosWatchedAlbum, id=album_id, is_active=True)
        try:
            session = create_session()
        except GooglePhotosError as e:
            return Response({'error': str(e)}, status=502)
        return Response(session, status=201)


class WatchedAlbumSessionPollView(APIView):
    """GET /api/google_photos/watch/<id>/session/<session_id>/poll/ --
    proxies Google's sessions.get so the frontend can poll for
    mediaItemsSet without holding a Google access token itself."""

    permission_classes = (IsAuthenticated,)

    def get(self, request, album_id, session_id):
        get_object_or_404(GooglePhotosWatchedAlbum, id=album_id, is_active=True)
        try:
            status = get_session(session_id)
        except GooglePhotosError as e:
            return Response({'error': str(e)}, status=502)
        return Response(status)


class WatchedAlbumSessionCompleteView(APIView):
    """POST /api/google_photos/watch/<id>/session/<session_id>/complete/ --
    once the frontend has seen mediaItemsSet=true, lists every picked item,
    downloads only the ones not already synced for this album (dedup on
    Google's persistent PickedMediaItem.id), and cleans up the session.

    Downloads happen synchronously in this request rather than a background
    task - baseUrls are only valid ~60 minutes post-session and a picker
    selection is bounded (2000 items max), so this stays well within a
    normal request timeout even for a large batch."""

    permission_classes = (IsAuthenticated,)

    def post(self, request, album_id, session_id):
        album = get_object_or_404(GooglePhotosWatchedAlbum, id=album_id, is_active=True)

        try:
            picked_items = list_media_items(session_id)
        except GooglePhotosError as e:
            return Response({'error': str(e)}, status=502)

        already_synced_ids = set(
            GooglePhotosSyncedItem.objects.filter(watched_album=album)
            .values_list('google_media_item_id', flat=True)
        )

        os.makedirs(settings.GOOGLE_PHOTOS_STAGING_DIR, exist_ok=True)
        _chown_upload_path(settings.GOOGLE_PHOTOS_STAGING_DIR)

        new_count = 0
        for item in picked_items:
            item_id = item['id']
            if item_id in already_synced_ids:
                continue

            media_file = item.get('mediaFile', {})
            filename = media_file.get('filename') or f'{item_id}.jpg'
            base_url = media_file.get('baseUrl')
            if not base_url:
                continue

            dest = _unique_destination(settings.GOOGLE_PHOTOS_STAGING_DIR, os.path.basename(filename))
            try:
                download_media_item(base_url, dest)
            except GooglePhotosError:
                # One bad item shouldn't abort the rest of a large batch --
                # it simply isn't recorded as synced, so it's retried
                # (still not deduped) on the next sync of this album.
                continue
            _chown_upload_path(dest)

            GooglePhotosSyncedItem.objects.create(
                watched_album=album, google_media_item_id=item_id, filename=os.path.basename(dest))
            new_count += 1

        delete_session(session_id)

        album.last_synced_at = timezone.now()
        album.save(update_fields=['last_synced_at'])

        return Response({
            'new_count': new_count,
            'already_had_count': len(picked_items) - new_count,
        })


def _serialize_album(album):
    return {
        'id': album.id,
        'title': album.title,
        'created_at': album.created_at.isoformat(),
        'last_synced_at': album.last_synced_at.isoformat() if album.last_synced_at else None,
        'item_count': album.synced_items.count(),
    }
