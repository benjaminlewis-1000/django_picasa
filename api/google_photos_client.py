"""Thin wrapper around Google's Photos Picker API (photospicker.googleapis.com)
-- backs api/photos_watch_views.py.

Google retired background/library-wide photo access in March 2025
(sharedAlbums.list, albums.share/join/etc, and the old photoslibrary* scopes
all now 403). The Picker API is what replaced it: a session-based flow where
the *user* has to browse Google Photos and manually select items every time
-- there is no "anything new since last sync?" call. See CLAUDE.md for the
full writeup and the one-time OAuth bootstrap (scripts/google_photos_authorize.py)
this depends on.

Deliberately plain `requests` (no google-auth/google-api-python-client
dependency) -- the only Google-side operations needed are a standard OAuth
refresh-token exchange and a handful of REST calls, both already well within
what `requests` (an existing dependency) handles directly.
"""

import time

import requests
from django.conf import settings

TOKEN_URL = 'https://oauth2.googleapis.com/token'
API_BASE = 'https://photospicker.googleapis.com/v1'

# Cached in-process (per worker) rather than persisted -- an access token is
# only valid ~1 hour, cheap to re-mint, and there's no multi-process
# coordination benefit to storing it anywhere shared (unlike UploadSession,
# which genuinely needs to be visible across worker processes/requests).
_access_token = None
_access_token_expiry = 0


class GooglePhotosError(Exception):
    pass


def _require_credentials():
    if not (settings.GOOGLE_PHOTOS_CLIENT_ID and settings.GOOGLE_PHOTOS_CLIENT_SECRET
            and settings.GOOGLE_PHOTOS_REFRESH_TOKEN):
        raise GooglePhotosError(
            'Google Photos isn\'t configured -- GOOGLE_PHOTOS_CLIENT_ID/'
            'CLIENT_SECRET/REFRESH_TOKEN must be set (see '
            'scripts/google_photos_authorize.py).')


def _get_access_token():
    global _access_token, _access_token_expiry
    _require_credentials()
    # 60s safety margin so a token doesn't expire mid-request.
    if _access_token and time.time() < _access_token_expiry - 60:
        return _access_token

    response = requests.post(TOKEN_URL, data={
        'client_id': settings.GOOGLE_PHOTOS_CLIENT_ID,
        'client_secret': settings.GOOGLE_PHOTOS_CLIENT_SECRET,
        'refresh_token': settings.GOOGLE_PHOTOS_REFRESH_TOKEN,
        'grant_type': 'refresh_token',
    }, timeout=15)
    if not response.ok:
        raise GooglePhotosError(f'Failed to refresh Google access token: {response.text}')

    body = response.json()
    _access_token = body['access_token']
    _access_token_expiry = time.time() + body.get('expires_in', 3600)
    return _access_token


def _request(method, path, **kwargs):
    token = _get_access_token()
    headers = kwargs.pop('headers', {})
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request(method, f'{API_BASE}{path}', headers=headers, timeout=30, **kwargs)
    if not response.ok:
        raise GooglePhotosError(f'Google Photos API error ({response.status_code}): {response.text}')
    return response.json() if response.content else {}


def create_session():
    """Returns {'session_id', 'picker_uri'} -- the frontend sends the user
    to picker_uri to select items for this session."""
    body = _request('POST', '/sessions')
    return {'session_id': body['id'], 'picker_uri': body['pickerUri']}


def get_session(session_id):
    """Returns {'media_items_set': bool} -- polled by the frontend until
    True, at which point the picked items can be listed."""
    body = _request('GET', f'/sessions/{session_id}')
    return {'media_items_set': bool(body.get('mediaItemsSet'))}


def list_media_items(session_id):
    """Returns every PickedMediaItem for a completed session, following
    pagination -- a session can hold up to 2000 items (Picker API default
    maxItemCount)."""
    items = []
    page_token = None
    while True:
        params = {'sessionId': session_id}
        if page_token:
            params['pageToken'] = page_token
        body = _request('GET', '/mediaItems', params=params)
        items.extend(body.get('mediaItems', []))
        page_token = body.get('nextPageToken')
        if not page_token:
            return items


def delete_session(session_id):
    """Best-effort cleanup per Google's documented best practice ("delete
    sessions once the user has selected media items and your app has
    retrieved the media item bytes") -- never lets a cleanup failure block
    a sync that otherwise succeeded."""
    try:
        _request('DELETE', f'/sessions/{session_id}')
    except GooglePhotosError:
        pass


def download_media_item(base_url, dest_path):
    """Streams the original-quality bytes for one PickedMediaItem to
    dest_path. `=d` requests the original file rather than a resized
    rendition (baseUrl alone returns a default preview size). baseUrl is
    only valid ~60 minutes after the session completes, so this must run
    immediately after list_media_items, not deferred."""
    token = _get_access_token()
    with requests.get(f'{base_url}=d', headers={'Authorization': f'Bearer {token}'},
                       stream=True, timeout=120) as response:
        if not response.ok:
            raise GooglePhotosError(f'Failed to download media item ({response.status_code}).')
        with open(dest_path, 'wb') as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                out.write(chunk)
