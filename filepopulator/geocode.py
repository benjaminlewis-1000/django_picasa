"""Reverse geocoding: a precise lookup (Nominatim, rate-limited, cached by
coordinate) and an independent, offline nearest-metro-area fallback.

The two are deliberately separate. Nominatim resolves a coordinate to the
actual place it's in, however small or unrecognizable ("Bothell, WA");
the nearest-metro lookup instead answers "what's the closest place someone
would actually recognize" ("Seattle, WA") by searching a static, offline
dataset of populated places -- no network call, no rate limit, safe to
run for every image at ingestion time rather than just once per unique
coordinate.
"""

import csv
import math
import os
import time

from django.conf import settings
from django.db.models.functions import Round
from django.utils import timezone

MAJOR_PLACES_CSV = os.path.join(os.path.dirname(__file__), 'data', 'major_places.csv')

# Search radius bands, in km, tried in order -- the first band with any
# candidate wins (picking the largest-population place in that band),
# rather than a single flat cutoff. This is what gives "Bothell -> Seattle
# or Bellevue" (both well within the first band or two) while correctly
# leaving somewhere genuinely remote (nothing populous within the last,
# widest band) with no metro-area match at all.
SEARCH_RADIUS_BANDS_KM = [25, 50, 80]

_major_places_cache = None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load_major_places():
    global _major_places_cache
    if _major_places_cache is not None:
        return _major_places_cache

    places = []
    with open(MAJOR_PLACES_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            places.append({
                'name': row['name'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'country_code': row['country_code'],
                'admin1_code': row['admin1_code'],
                'population': int(row['population']),
            })
    _major_places_cache = places
    return places


def find_nearest_metro(lat, lon):
    """Returns (name, distance_km) for the largest populated place within
    the nearest radius band that has any candidate at all, or (None, None)
    if nothing in the dataset is within SEARCH_RADIUS_BANDS_KM[-1]."""
    places = _load_major_places()

    best_by_band = None
    for radius_km in SEARCH_RADIUS_BANDS_KM:
        candidates = []
        for place in places:
            # Cheap pre-filter before the real haversine calculation: 1
            # degree of latitude is ~111km everywhere, so this bounding
            # box can only ever be too permissive, never too strict.
            if abs(place['lat'] - lat) > radius_km / 111.0:
                continue
            if abs(place['lon'] - lon) > radius_km / 111.0:
                continue
            dist = _haversine_km(lat, lon, place['lat'], place['lon'])
            if dist <= radius_km:
                candidates.append((place, dist))

        if candidates:
            best_by_band = max(candidates, key=lambda c: c[0]['population'])
            break

    if best_by_band is None:
        return None, None

    place, dist = best_by_band
    return place['name'], dist


def _get_nominatim_geocode():
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent=settings.NOMINATIM_USER_AGENT)
    return RateLimiter(geolocator.reverse, min_delay_seconds=1.1, max_retries=2, error_wait_seconds=5.0)


def reverse_geocode_precise(lat, lon):
    """Queries Nominatim for the precise place at (lat, lon). Returns a
    dict of the fields GeocodeCache stores, or raises on failure -- callers
    are expected to catch and record lookup_failed/lookup_error, same
    pattern as the rest of this app's failure handling."""
    reverse = _get_nominatim_geocode()
    location = reverse(f"{lat}, {lon}", exactly_one=True, language='en')

    if location is None:
        return {
            'locality': None, 'county': None, 'state': None, 'country': None,
            'display_name': None, 'raw_response': None,
        }

    address = location.raw.get('address', {})
    locality = (
        address.get('city') or address.get('town') or address.get('village')
        or address.get('hamlet') or address.get('suburb')
    )
    return {
        'locality': locality,
        'county': address.get('county'),
        'state': address.get('state'),
        'country': address.get('country'),
        'display_name': location.address,
        'raw_response': location.raw,
    }


def run_geocoding_backfill(limit=None, dry_run=False, log=print):
    """Geocodes every distinct (rounded) GPS coordinate among ImageFiles
    that doesn't already have a GeocodeCache entry, then links matching
    ImageFile rows to it. Shared by the one-time backfill management
    command and the small recurring Celery task that picks up newly
    ingested images -- both just want "catch up whatever's uncached",
    differing only in how large a batch (`limit`) makes sense to run at
    once.

    Safe to interrupt and re-run: a coordinate is only ever processed once
    it has no GeocodeCache row, so a partial run just picks up where it
    left off. A failure geocoding one coordinate is recorded
    (lookup_failed/lookup_error) rather than raised, so it can't abort the
    run for every other coordinate queued behind it.

    Returns a dict of counts: distinct, already_cached, remaining,
    succeeded, failed.
    """
    from filepopulator.models import GeocodeCache, ImageFile

    has_gps = ImageFile.objects.exclude(gps_lat_decimal=-999).exclude(gps_lon_decimal=-999)
    coords = (
        has_gps
        .annotate(rlat=Round('gps_lat_decimal', GeocodeCache.ROUND_DECIMALS),
                  rlon=Round('gps_lon_decimal', GeocodeCache.ROUND_DECIMALS))
        .values_list('rlat', 'rlon')
        .distinct()
    )

    existing = set(GeocodeCache.objects.values_list('lat', 'lon'))
    todo = [(lat, lon) for lat, lon in coords if (lat, lon) not in existing]

    log(f"Distinct coordinates with GPS: {coords.count()}")
    log(f"Already cached: {len(existing)}")
    log(f"Remaining to geocode: {len(todo)}")

    result = {
        'distinct': coords.count(), 'already_cached': len(existing),
        'remaining': len(todo), 'succeeded': 0, 'failed': 0,
    }

    if dry_run:
        log("Dry run -- no changes written.")
        return result

    if limit is not None:
        todo = todo[:limit]

    start = time.time()

    for i, (lat, lon) in enumerate(todo):
        metro_name, metro_distance = find_nearest_metro(lat, lon)

        cache_entry = GeocodeCache(
            lat=lat, lon=lon,
            nearest_metro_name=metro_name,
            nearest_metro_distance_km=metro_distance,
        )

        try:
            precise = reverse_geocode_precise(lat, lon)
            cache_entry.locality = precise['locality']
            cache_entry.county = precise['county']
            cache_entry.state = precise['state']
            cache_entry.country = precise['country']
            cache_entry.display_name = precise['display_name']
            cache_entry.raw_response = precise['raw_response']
            cache_entry.geocoded_at = timezone.now()
            result['succeeded'] += 1
        except Exception as e:
            cache_entry.lookup_failed = True
            cache_entry.lookup_error = str(e)
            result['failed'] += 1
            log(f"Failed to geocode ({lat}, {lon}): {e}")

        cache_entry.save()

        matching = (
            has_gps
            .annotate(rlat=Round('gps_lat_decimal', GeocodeCache.ROUND_DECIMALS),
                      rlon=Round('gps_lon_decimal', GeocodeCache.ROUND_DECIMALS))
            .filter(rlat=lat, rlon=lon)
        )
        matching.update(geocode=cache_entry)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            log(f"{i + 1}/{len(todo)} coordinates processed ({elapsed:.0f}s elapsed)")

    log(f"Done. {result['succeeded']} succeeded, {result['failed']} failed, {time.time() - start:.0f}s total.")
    return result
