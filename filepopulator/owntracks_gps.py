"""GPS estimation for photos with no camera GPS at all, backed by a
separate OwnTracks-fed service (not part of this repo) that answers
"where was the phone at timestamp X" from continuous location history.

Deliberately scoped to an explicit allowlist of cameras (see
CameraGpsBackfillConfig, a DB-backed table a frontend toggle page can
curate directly) -- the location history only actually corresponds to
one specific person's own location, so applying this to every camera
in the library would silently attribute someone else's photos (or a
different device's EXIF-copied metadata -- see the "Family Texts" case
found while investigating this) to a location the actual photographer
never visited. A camera is onboarded by enabling it there explicitly,
never automatically.

The one detail that would silently produce wrong results if gotten
wrong: this project's own ImageFile.dateTaken/_get_date_taken() throws
away the real UTC offset entirely -- it reads EXIF's naive local-time
string and blindly tags it UTC (see that method's own comment), so
dateTaken in the DB is NOT usable to build the datetime this service
needs. Re-read fresh from the file here instead, trying each known
real source of a per-photo offset in turn:
  - Standard EXIF OffsetTimeOriginal (phones) -- confirmed present and
    varying by real travel location on the Galaxy S23+ (-04:00 at
    home, +01:00/+02:00/+03:00 while traveling).
  - Nikon's MakerNotes TimeZone + DaylightSavings (the camera's "World
    Time" setting) -- confirmed present and correct on a real Nikon
    D3200 trip: TimeZone=-05:00 + DaylightSavings=Yes gave the exact
    true -04:00 offset, verified against a real OwnTracks match
    (resolved to Pisa, Italy, to the hundredth of a degree) while the
    camera's raw (unadjusted) clock reading would have been wrong by a
    full 8 hours.
Never falls back to a guessed/assumed timezone (e.g. this project's
own configured TIME_ZONE) -- that would be wrong for exactly the
travel photos most likely to be missing GPS in the first place.
"""
import json
import subprocess

import requests
from django.conf import settings

EXIFTOOL_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 15

# The per-camera allowlist itself now lives in CameraGpsBackfillConfig
# (DB-backed, toggleable from a frontend page) -- see that model's own
# docstring for why this moved out of code. get_enabled_camera_models()
# is the single place anything in this module reads the current
# allowlist from.
def get_enabled_camera_models():
    from filepopulator.models import CameraGpsBackfillConfig
    return list(
        CameraGpsBackfillConfig.objects.filter(enabled=True).values_list('camera_model', flat=True)
    )

# Flags this coordinate as inferred from location history, not a real
# camera GPS reading -- distinct from the field's own 'exif' default, so
# anything downstream (geocoding, display) can tell the two apart if it
# ever needs to.
GPS_SOURCE_OWNTRACKS = 'owntracks'


class OwnTracksConfigError(Exception):
    """Raised when OWNTRACKS_LOCATION_API_URL/_API_KEY aren't configured."""


def _nikon_offset_str(timezone_str, daylight_savings_str):
    """Nikon's World Time setting stores a base UTC offset plus a
    separate yes/no DST flag -- the real effective offset is their sum
    (confirmed against a real file: TimeZone=-05:00 + DaylightSavings=
    Yes -> true -04:00, matching a real OwnTracks match to the hundredth
    of a degree). Returns a '+HH:MM'/'-HH:MM' string, or None if
    timezone_str isn't a recognizable +/-HH:MM shape."""
    import re

    m = re.match(r'^([+-])(\d{2}):(\d{2})$', timezone_str or '')
    if not m:
        return None
    sign, hours, minutes = m.group(1), int(m.group(2)), int(m.group(3))
    # DST always ADDS an hour to the real, SIGNED offset (EST -05:00 ->
    # EDT -04:00, a smaller-magnitude negative number) -- must be applied
    # after the sign, not to the unsigned magnitude (the original bug
    # here: adding 60 to the magnitude before negating made a negative
    # offset MORE negative instead of less, caught by a real test).
    signed_minutes = hours * 60 + minutes
    if sign == '-':
        signed_minutes = -signed_minutes
    if (daylight_savings_str or '').strip().lower() == 'yes':
        signed_minutes += 60
    out_sign = '+' if signed_minutes >= 0 else '-'
    magnitude = abs(signed_minutes)
    return f'{out_sign}{magnitude // 60:02d}:{magnitude % 60:02d}'


def _read_offset_datetime(file_path):
    """Returns (naive_date_str, offset_str) for file_path's real capture
    instant, trying each known offset source in turn (see this module's
    own docstring), or (None, None) if none of them are present/
    parseable -- never guesses a timezone as a fallback."""
    try:
        result = subprocess.run(
            ['exiftool', '-j', '-DateTimeOriginal', '-OffsetTimeOriginal',
             '-Nikon:TimeZone', '-Nikon:DaylightSavings', file_path],
            capture_output=True, text=True, timeout=EXIFTOOL_TIMEOUT_SECONDS,
        )
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, IndexError):
        return None, None
    if not data:
        return None, None
    entry = data[0]
    date_str = entry.get('DateTimeOriginal')
    if not date_str:
        return None, None

    offset_str = entry.get('OffsetTimeOriginal')
    if not offset_str:
        offset_str = _nikon_offset_str(entry.get('TimeZone'), entry.get('DaylightSavings'))
    if not offset_str:
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


def backfill_camera_gps(camera_models=None, start_date=None, end_date=None,
                         retry_failed=False, dry_run=False, limit=None, log=print,
                         include_verified_person=True):
    """One-time-per-photo (but safely re-runnable -- intended to end up
    as a recurring cron/Celery task, picking up each newly-opted-in
    camera's new photos as they're ingested) backfill: for every eligible
    photo still missing GPS, estimate it from OwnTracks location history.
    Returns a dict of outcome counts.

    A photo is eligible two independent ways (OR'd together):
    - its camera_model is in camera_models (defaults to
      get_enabled_camera_models(), every camera currently toggled on in
      CameraGpsBackfillConfig) -- a caller only passes a narrower list to
      test/scope a single camera (see --camera-model on the management
      command), never to permanently widen scope beyond what's been
      explicitly opted in there.
    - include_verified_person=True (the default) and it has a VALIDATED
      Face for any Person with gps_backfill_source=True -- a confirmed
      appearance means that specific person was physically present at
      that moment, which is actually a more direct signal than camera
      ownership (someone else's phone could have taken the photo). See
      Person.gps_backfill_source's own docstring.

    start_date/end_date (date objects) filter on dateTaken -- useful for
    a camera with years of history that predates OwnTracks tracking
    entirely (confirmed real: an unscoped run against a Nikon D3200's
    full ~27k-photo backlog spent its first many calls on 2014 photos
    that could only ever come back "no nearby point," each one still
    paying a real exiftool+HTTP round trip for nothing). dateTaken is
    the UNRELIABLE, offset-blind field for anything needing real
    precision (see this module's own docstring) -- fine here since a
    multi-day scoping window swallows the few hours of error that
    causes, nowhere close to misses at a day boundary mattering.

    By default, any photo with an existing GpsEstimationAttempt record
    is skipped entirely (no exiftool/HTTP call at all) -- see that
    model's own docstring for why. retry_failed=True re-includes rows
    whose existing attempt was 'no_nearby_point' specifically (never
    'no_offset_tag', which can't change)."""
    from django.db.models import Q
    from filepopulator.models import GpsEstimationAttempt, ImageFile

    if camera_models is None:
        camera_models = get_enabled_camera_models()

    eligibility = Q(camera_model__in=camera_models)
    if include_verified_person:
        eligibility |= Q(face__validated=True, face__declared_name__gps_backfill_source=True)

    qs = ImageFile.objects.filter(
        eligibility, gps_lat_decimal=-999,
    ).distinct().order_by('dateTaken')
    if start_date:
        qs = qs.filter(dateTaken__date__gte=start_date)
    if end_date:
        qs = qs.filter(dateTaken__date__lte=end_date)
    if retry_failed:
        qs = qs.exclude(gps_estimation_attempt__reason=GpsEstimationAttempt.REASON_NO_OFFSET_TAG)
    else:
        qs = qs.filter(gps_estimation_attempt__isnull=True)
    if limit:
        qs = qs[:limit]

    counts = {'matched': 0, 'no_offset_tag': 0, 'no_nearby_point': 0, 'error': 0}
    for img in qs:
        iso = build_capture_iso8601(img.filename)
        if iso is None:
            counts['no_offset_tag'] += 1
            log(f"SKIP (no offset source): {img.filename}")
            if not dry_run:
                GpsEstimationAttempt.objects.update_or_create(
                    image=img, defaults={'reason': GpsEstimationAttempt.REASON_NO_OFFSET_TAG},
                )
            continue

        try:
            result = query_nearest_location(iso)
        except OwnTracksConfigError:
            raise

        if result is None:
            counts['no_nearby_point'] += 1
            log(f"SKIP (no nearby OwnTracks point): {img.filename} ({iso})")
            if not dry_run:
                GpsEstimationAttempt.objects.update_or_create(
                    image=img, defaults={'reason': GpsEstimationAttempt.REASON_NO_NEARBY_POINT},
                )
            continue

        lat, lon = result
        counts['matched'] += 1
        log(f"MATCH: {img.filename} ({iso}) -> {lat}, {lon}")
        if not dry_run:
            ImageFile.objects.filter(pk=img.pk).update(
                gps_lat_decimal=lat, gps_lon_decimal=lon, gps_source=GPS_SOURCE_OWNTRACKS,
            )
            GpsEstimationAttempt.objects.filter(image=img).delete()

    return counts
