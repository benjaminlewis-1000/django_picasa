"""GPS estimation for photos with no camera GPS at all, backed by a
separate OwnTracks-fed service (not part of this repo) that answers
"where was the phone at timestamp X" from continuous location history.

Deliberately scoped to ONE camera at a time (see S23_CAMERA_MODELS) --
the location history only actually corresponds to one specific phone's
owner, so applying this to every camera in the library would silently
attribute someone else's photos (or a different device's EXIF-copied
metadata -- see the "Family Texts" case found while investigating this)
to a location the actual photographer never visited.

The one detail that would silently produce wrong results if gotten
wrong: this project's own ImageFile.dateTaken/_get_date_taken() throws
away the real UTC offset entirely -- it reads EXIF's naive local-time
string and blindly tags it UTC (see that method's own comment), so
dateTaken in the DB is NOT usable to build the datetime this service
needs. Confirmed via real EXIF on the actual target photos: a real
OffsetTimeOriginal tag is present and varies by real travel location
(-04:00 at home, +01:00/+02:00/+03:00 while traveling) -- re-read fresh
from the file here rather than trusting the DB.
"""
import json
import subprocess

import requests
from django.conf import settings

EXIFTOOL_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 15

# Only the Galaxy S23+ for now, per explicit instruction -- the two
# camera_model strings the same physical phone has written to EXIF (see
# CLAUDE.md's "which cameras have the most/least GPS" investigation).
S23_CAMERA_MODELS = ['Galaxy S23+', 'SM-S916U']

# Flags this coordinate as inferred from location history, not a real
# camera GPS reading -- distinct from the field's own 'exif' default, so
# anything downstream (geocoding, display) can tell the two apart if it
# ever needs to.
GPS_SOURCE_OWNTRACKS = 'owntracks'


class OwnTracksConfigError(Exception):
    """Raised when OWNTRACKS_LOCATION_API_URL/_API_KEY aren't configured."""


def _read_offset_datetime(file_path):
    """Returns (naive_date_str, offset_str) straight from exiftool, or
    (None, None) if either tag is missing/unparseable -- never guesses a
    timezone (e.g. this project's own configured TIME_ZONE) as a
    fallback, since that would be wrong for every travel photo, which is
    exactly the case most likely to be missing GPS in the first place."""
    try:
        result = subprocess.run(
            ['exiftool', '-j', '-DateTimeOriginal', '-OffsetTimeOriginal', file_path],
            capture_output=True, text=True, timeout=EXIFTOOL_TIMEOUT_SECONDS,
        )
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, IndexError):
        return None, None
    if not data:
        return None, None
    entry = data[0]
    date_str = entry.get('DateTimeOriginal')
    offset_str = entry.get('OffsetTimeOriginal')
    if not date_str or not offset_str:
        return None, None
    return date_str, offset_str


def build_capture_iso8601(file_path):
    """Returns a real offset-aware ISO 8601 string for file_path's actual
    capture instant, or None if it can't be built (missing tag, or a
    shape exiftool/strptime don't recognize)."""
    from datetime import datetime

    date_str, offset_str = _read_offset_datetime(file_path)
    if date_str is None:
        return None
    try:
        aware_dt = datetime.strptime(f'{date_str}{offset_str}', '%Y:%m:%d %H:%M:%S%z')
    except ValueError:
        return None
    return aware_dt.isoformat()


def query_nearest_location(iso8601_datetime):
    """Calls the OwnTracks-backed service for the nearest recorded point
    to iso8601_datetime. Returns (lat, lon) on a real match, None if
    nothing was close enough (404) or the call otherwise failed -- the
    caller treats either the same way (skip this photo, safely retryable
    next run). Raises OwnTracksConfigError for a 401 -- a wrong/missing
    key will fail identically for every remaining photo, so the whole
    run aborts loudly rather than silently skipping everything."""
    if not settings.OWNTRACKS_LOCATION_API_URL or not settings.OWNTRACKS_LOCATION_API_KEY:
        raise OwnTracksConfigError(
            "OWNTRACKS_LOCATION_API_URL/OWNTRACKS_LOCATION_API_KEY are not configured."
        )
    try:
        response = requests.get(
            f'{settings.OWNTRACKS_LOCATION_API_URL}/api/nearest_location',
            params={'datetime': iso8601_datetime},
            headers={'X-API-Key': settings.OWNTRACKS_LOCATION_API_KEY},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        settings.LOGGER.warning(f"OwnTracks location lookup failed (network error): {e}")
        return None

    if response.status_code == 401:
        raise OwnTracksConfigError(
            "OwnTracks location service rejected the API key (401) -- check "
            "OWNTRACKS_LOCATION_API_KEY."
        )
    if response.status_code == 404:
        return None
    if response.status_code == 400:
        settings.LOGGER.warning(
            f"OwnTracks location service rejected datetime={iso8601_datetime!r} (400): "
            f"{response.text}"
        )
        return None
    if response.status_code != 200:
        settings.LOGGER.warning(
            f"OwnTracks location lookup returned {response.status_code}: {response.text}"
        )
        return None

    data = response.json()
    # accuracy/date_utc deliberately not used -- accuracy's -1 sentinel
    # isn't a real meters value, and date_utc is a lossy whole-seconds
    # display string per the service's own docs; neither matters for
    # just storing a coordinate.
    return data['latitude'], data['longitude']


def backfill_s23_gps(dry_run=False, limit=None, log=print):
    """One-time (but safely re-runnable) backfill: for every S23_
    CAMERA_MODELS photo still missing GPS, estimate it from OwnTracks
    location history. Returns a dict of outcome counts."""
    from filepopulator.models import ImageFile

    qs = ImageFile.objects.filter(
        camera_model__in=S23_CAMERA_MODELS, gps_lat_decimal=-999,
    ).order_by('dateTaken')
    if limit:
        qs = qs[:limit]

    counts = {'matched': 0, 'no_offset_tag': 0, 'no_nearby_point': 0, 'error': 0}
    for img in qs:
        iso = build_capture_iso8601(img.filename)
        if iso is None:
            counts['no_offset_tag'] += 1
            log(f"SKIP (no OffsetTimeOriginal tag): {img.filename}")
            continue

        try:
            result = query_nearest_location(iso)
        except OwnTracksConfigError:
            raise

        if result is None:
            counts['no_nearby_point'] += 1
            log(f"SKIP (no nearby OwnTracks point): {img.filename} ({iso})")
            continue

        lat, lon = result
        counts['matched'] += 1
        log(f"MATCH: {img.filename} ({iso}) -> {lat}, {lon}")
        if not dry_run:
            ImageFile.objects.filter(pk=img.pk).update(
                gps_lat_decimal=lat, gps_lon_decimal=lon, gps_source=GPS_SOURCE_OWNTRACKS,
            )

    return counts
