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
from django.db import IntegrityError, transaction
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


def search_major_places(query, limit=10):
    """Type-ahead over the offline major_places.csv gazetteer -- case-
    insensitive substring match, largest-population-first. Backs the
    geocode-review tool's editable metro field while typing; unlike
    resolve_named_place below, this is purely offline and cheap enough to
    call on every keystroke (no rate limit to respect)."""
    query = query.strip().lower()
    if not query:
        return []
    places = _load_major_places()
    matches = [p for p in places if query in p['name'].lower()]
    matches.sort(key=lambda p: p['population'], reverse=True)
    return matches[:limit]


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


# Last-resort fallback for reverse_geocode_precise/run_geocoding_backfill
# when Nominatim genuinely has nothing usable for a coordinate (no
# city/town/village/hamlet/suburb tag, and none of the weaker
# municipality/county/state_district/borough tags either) - "the nearest
# real, named place we know of" rather than leaving it permanently
# unknown. Deliberately generous (200km) since this only ever fires when
# every other option has already failed - some real answer, even a
# distant one, communicates more than nothing, as long as the caller
# marks it approximate (GeocodeCache.locality_is_approximate) rather than
# presenting it as the actual place a photo was taken.
NAMED_PLACE_FALLBACK_RADIUS_KM = 200


def find_nearest_named_place(lat, lon, max_radius_km=NAMED_PLACE_FALLBACK_RADIUS_KM):
    """Unlike find_nearest_metro above (largest place within the nearest
    of several fixed radius bands), this picks the single NEAREST place
    in the gazetteer regardless of population, capped at max_radius_km.
    Returns (name, country_code, distance_km), or (None, None, None) if
    nothing in the dataset is within range."""
    places = _load_major_places()
    best = None
    best_dist = None
    for place in places:
        if abs(place['lat'] - lat) > max_radius_km / 111.0:
            continue
        if abs(place['lon'] - lon) > max_radius_km / 111.0:
            continue
        dist = _haversine_km(lat, lon, place['lat'], place['lon'])
        if dist <= max_radius_km and (best_dist is None or dist < best_dist):
            best, best_dist = place, dist

    if best is None:
        return None, None, None
    return best['name'], best['country_code'], best_dist


# Nominatim's usage policy (operations.osmfoundation.org/policies/nominatim)
# singles out "scripts running continuously or at regular intervals" for a
# stricter cap than its general ~1/sec guidance: max 4 requests/minute,
# results cached locally. Our own traffic - a recurring hourly batch task
# (run_geocoding_backfill) plus this tool's occasional interactive
# corrections, both from the same server IP - fits exactly that
# description, and we already cache every result (GeocodeCache, keyed
# per-coordinate) as required. Applied to both RateLimiters below.
NOMINATIM_MIN_DELAY_SECONDS = 15.0  # 60s / 4 requests-per-minute

# Built once per process and reused - see the comment inside
# _get_nominatim_geocode below for why that reuse is the actual fix, not
# just an optimization.
_reverse_rate_limiter = None
_forward_rate_limiter = None


def _get_nominatim_geocode():
    global _reverse_rate_limiter
    if _reverse_rate_limiter is not None:
        return _reverse_rate_limiter

    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    # geopy's own default timeout is a mere 1 second, far too aggressive
    # for a real round trip to a public geocoding server -- found via a
    # real backfill run spuriously failing every lookup with
    # ReadTimeoutError (never surfaced in tests, which mock the network
    # call entirely).
    geolocator = Nominatim(user_agent=settings.NOMINATIM_USER_AGENT, timeout=10)
    # swallow_exceptions=False (geopy's RateLimiter default is True) -
    # found real, live production data 2026-09-10: a real, transient
    # failure (timeout/429/network blip) after exhausting max_retries
    # used to be silently converted to a plain `None` return, which
    # reverse_geocode_precise treats identically to "Nominatim
    # successfully found nothing here" - the two are NOT the same
    # (one is worth retrying, the other isn't), but were indistinguishable
    # in the stored data. With this off, a real failure now raises and
    # gets caught by run_geocoding_backfill's own try/except, which
    # already correctly records lookup_failed/lookup_error - that
    # handling existed from the start, it just never actually ran for
    # this specific failure mode.
    #
    # Built ONCE per process (the module-level global above) rather than
    # fresh on every call, which is what this used to do (a new
    # RateLimiter was constructed inside every reverse_geocode_precise()
    # call, i.e. once per coordinate inside run_geocoding_backfill's own
    # loop). That was a real, previously-undiscovered bug, not just a
    # missed optimization: min_delay_seconds is enforced via *instance*
    # state (RateLimiter._last_call) - a fresh instance has never made a
    # call before, so every single coordinate looked like this limiter's
    # first-ever request and got waved through with zero delay. The
    # "2 seconds between requests" this used to claim was never actually
    # happening across a batch run at all, which goes a long way toward
    # explaining how much harder we were getting rate-limited than a 2s
    # gap should cause (found investigating that 2026-09-10). Reusing one
    # instance for the process's lifetime is what makes min_delay_seconds
    # mean anything across more than a single call.
    _reverse_rate_limiter = RateLimiter(
        geolocator.reverse, min_delay_seconds=NOMINATIM_MIN_DELAY_SECONDS,
        max_retries=2, error_wait_seconds=NOMINATIM_MIN_DELAY_SECONDS,
        swallow_exceptions=False,
    )
    return _reverse_rate_limiter


def _get_nominatim_forward_geocode():
    global _forward_rate_limiter
    if _forward_rate_limiter is not None:
        return _forward_rate_limiter

    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent=settings.NOMINATIM_USER_AGENT, timeout=10)
    # swallow_exceptions=False - same reasoning as _get_nominatim_geocode
    # above: a real failure here (this is the interactive, on-submit path
    # behind the geocode-review tool's "Save correction" button) shouldn't
    # look identical to "not a real place" - api/geocode_views.py's
    # GeocodeReviewActionView catches this specifically to tell a user
    # "try again" apart from "that isn't a place we recognize".
    #
    # max_retries=0 (unlike the reverse/batch RateLimiter above, which
    # retries twice) - confirmed for real 2026-09-10: a rate-limited (429)
    # call retrying twice, 15s apart (see NOMINATIM_MIN_DELAY_SECONDS),
    # would take 30s+ to finally raise, well past axios's 15s client-side
    # timeout in the frontend - so the specific, useful error message
    # never even arrived; the request just looked like a plain network
    # timeout with no response body at all. Retrying against an *active*
    # rate limit is futile anyway (it will almost certainly 429 again
    # seconds later) - this is a single, latency-sensitive, on-submit call
    # with a real user waiting on it, not a patient background batch job.
    # Fail fast and let the "wait about an hour" message
    # (GeocodeReviewActionView) do its job instead of silently eating the
    # time budget first. error_wait_seconds is passed anyway (unused with
    # 0 retries) only because RateLimiter's constructor asserts it's >=
    # min_delay_seconds.
    #
    # Also built once per process, same reasoning as the reverse limiter
    # above - a rapid-fire sequence of manual corrections (unlikely, but
    # possible) should still be spaced out rather than each looking like
    # this limiter's first-ever call.
    _forward_rate_limiter = RateLimiter(
        geolocator.geocode, min_delay_seconds=NOMINATIM_MIN_DELAY_SECONDS,
        max_retries=0, error_wait_seconds=NOMINATIM_MIN_DELAY_SECONDS,
        swallow_exceptions=False,
    )
    return _forward_rate_limiter


def resolve_named_place(query):
    """Forward-geocodes an arbitrary place name via Nominatim -- the
    geocode-review tool's fallback for a correction that isn't in the
    offline major_places.csv gazetteer (a hometown too small to be
    "major", a national park, a landmark, anything OSM has a name for).
    Nominatim actually resolving it *is* the realness check: there's no
    separate list of "valid" places to maintain, since a match is by
    definition real, named OSM data, not free text.

    Unlike search_major_places above, this is a single on-submit network
    call (rate-limited the same as reverse_geocode_precise), not something
    to call on every keystroke.

    Returns {'name', 'lat', 'lon', 'country', 'state', 'display_name'}, or
    None if Nominatim found nothing, or found something with no resolvable
    country (not a real place we can trust), or -- when the resolved
    country is the United States specifically -- no resolvable state
    (the one extra check the user asked for: "valid state if USA").
    """
    forward = _get_nominatim_forward_geocode()
    # addressdetails=True is NOT geopy's default for forward geocode()
    # (unlike reverse(), which returns the structured breakdown either
    # way) - without it, location.raw has no 'address' key at all, so
    # every real, correctly-resolved place (e.g. "Bremerton, WA") looked
    # like it had no resolvable country and got rejected as unrecognized.
    location = forward(query, exactly_one=True, language='en', addressdetails=True)

    if location is None:
        return None

    address = location.raw.get('address', {})
    country = address.get('country')
    state = address.get('state')

    if not country:
        return None
    if country == 'United States' and not state:
        return None

    # Nominatim has no single "the name of this place" field for a mix of
    # settlements, parks, and landmarks the way `address.city` does for a
    # settlement specifically -- `location.raw['name']` is the resolved
    # feature's own name when set (a national park, a named landmark),
    # falling back to the query text itself (a settlement's `name` is
    # often blank in the raw response even though `address.city` isn't).
    name = location.raw.get('name') or query

    return {
        'name': name,
        'lat': location.latitude,
        'lon': location.longitude,
        'country': country,
        'state': state,
        'display_name': location.address,
    }


def _extract_locality(address):
    """The locality-extraction tier logic, pulled out of
    reverse_geocode_precise so a one-off remediation pass can re-run it
    against already-cached raw_response data (see the 2026-09-10 backfill
    migration/cleanup) without needing a fresh Nominatim call - the
    address dict was already fetched and cached once; there's nothing new
    to learn from calling Nominatim again for the SAME coordinate, only
    from extracting more out of what's already there.

    First tier: an actual settlement-level tag. Second, weaker tier
    (relaxed 2026-09-10, real production data): a real Nominatim tag
    still, just a coarser one - a rural landmark/attraction/plantation
    often has an address with no city/town/etc at all, but does have a
    county or municipality, which is still a genuine, useful answer
    rather than nothing. Both tiers are real reverse-geocoded data; only
    find_nearest_named_place (run_geocoding_backfill) is an offline
    approximation, and that's the true last resort."""
    return (
        address.get('city') or address.get('town') or address.get('village')
        or address.get('hamlet') or address.get('suburb')
        or address.get('municipality') or address.get('county')
        or address.get('state_district') or address.get('borough')
    )


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
    return {
        'locality': _extract_locality(address),
        'county': address.get('county'),
        'state': address.get('state'),
        'country': address.get('country'),
        'display_name': location.address,
        'raw_response': location.raw,
    }


# Reuse radius for the proximity check below -- deliberately much wider
# than GeocodeCache.ROUND_DECIMALS' ~11m grid cell (that's a cache *key*
# granularity, not a "these are close enough to share a locality"
# judgment). 150m is tight enough to be confident it's still the same
# address/block (not just "same neighborhood"), which is what actually
# matters here: many of a real photo library's still-uncached coordinates
# are someone wandering the same building/block across separate visits,
# each landing in a different ~11m cell. Confirmed against real
# production data 2026-09-10: of 1278 then-uncached coordinates, 1059
# were within this radius of an already-successful precise geocode.
GEOCODE_REUSE_RADIUS_KM = 0.15


def _find_nearby_precise_geocode(lat, lon, candidates, radius_km=GEOCODE_REUSE_RADIUS_KM):
    """candidates: a list of (lat, lon, dict) tuples for already-known
    PRECISE (real Nominatim tag, not the offline nearest-named-place
    fallback) geocode results -- either loaded from the DB once at the
    start of a run, or appended to during the run as new coordinates
    resolve, so later coordinates in the same run can chain off earlier
    ones too. Returns the nearest match's dict within radius_km, or None.

    Deliberately excludes locality_is_approximate rows (the offline
    fallback) as reuse sources -- propagating an already-approximate guess
    to a second, different coordinate would compound the imprecision
    rather than share a genuine answer."""
    best = None
    best_dist = None
    for clat, clon, data in candidates:
        # Cheap bounding-box pre-filter, same trick find_nearest_metro
        # uses -- can only ever be too permissive, never too strict.
        if abs(clat - lat) > radius_km / 111.0:
            continue
        if abs(clon - lon) > radius_km / 111.0:
            continue
        dist = _haversine_km(lat, lon, clat, clon)
        if dist <= radius_km and (best_dist is None or dist < best_dist):
            best, best_dist = data, dist
    return best


def run_geocoding_backfill(limit=None, dry_run=False, log=print):
    """Geocodes every distinct (rounded) GPS coordinate among ImageFiles
    that doesn't already have a GeocodeCache entry, then links matching
    ImageFile rows to it. Shared by the one-time backfill management
    command and the small recurring Celery task that picks up newly
    ingested images -- both just want "catch up whatever's uncached",
    differing only in how large a batch (`limit`) makes sense to run at
    once.

    Before calling Nominatim for a coordinate, checks for an already-known
    PRECISE geocode (this run's own results so far, plus everything
    already in GeocodeCache) within GEOCODE_REUSE_RADIUS_KM and copies its
    locality/county/state/country/display_name directly if found -- no
    network call, no rate-limit consumption. This is what keeps a large
    backlog from costing one Nominatim call per coordinate: real photo
    libraries have a lot of "wandered the same block across visits" GPS
    noise that the ~11m cache-key rounding alone doesn't catch.

    Safe to interrupt and re-run: a coordinate is only ever processed once
    it has no GeocodeCache row, so a partial run just picks up where it
    left off. A failure geocoding one coordinate is recorded
    (lookup_failed/lookup_error) rather than raised, so it can't abort the
    run for every other coordinate queued behind it.

    Returns a dict of counts: distinct, already_cached, remaining,
    succeeded, failed, reused_nearby (a subset of succeeded that were
    resolved via the proximity check rather than a real Nominatim call).
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
        'remaining': len(todo), 'succeeded': 0, 'failed': 0, 'reused_nearby': 0,
    }

    if dry_run:
        log("Dry run -- no changes written.")
        return result

    if limit is not None:
        todo = todo[:limit]

    # Loaded once, up front -- every already-cached PRECISE (real
    # Nominatim tag, not the offline fallback) result is a candidate reuse
    # source for the proximity check below. Grown in-place as this run
    # resolves its own coordinates, so a later coordinate in the same run
    # can chain off an earlier one too, not just off what predates this
    # run.
    reuse_candidates = [
        (lat, lon, {'locality': locality, 'county': county, 'state': state,
                    'country': country, 'display_name': display_name})
        for lat, lon, locality, county, state, country, display_name in
        GeocodeCache.objects.filter(lookup_failed=False, locality_is_approximate=False)
        .values_list('lat', 'lon', 'locality', 'county', 'state', 'country', 'display_name')
    ]

    start = time.time()

    for i, (lat, lon) in enumerate(todo):
        metro_name, metro_distance = find_nearest_metro(lat, lon)

        cache_entry = GeocodeCache(
            lat=lat, lon=lon,
            nearest_metro_name=metro_name,
            nearest_metro_distance_km=metro_distance,
        )

        nearby = _find_nearby_precise_geocode(lat, lon, reuse_candidates)
        if nearby is not None:
            cache_entry.locality = nearby['locality']
            cache_entry.county = nearby['county']
            cache_entry.state = nearby['state']
            cache_entry.country = nearby['country']
            cache_entry.display_name = nearby['display_name']
            cache_entry.geocoded_at = timezone.now()
            result['succeeded'] += 1
            result['reused_nearby'] += 1
            reuse_candidates.append((lat, lon, nearby))
        else:
            try:
                precise = reverse_geocode_precise(lat, lon)
                cache_entry.locality = precise['locality']
                cache_entry.county = precise['county']
                cache_entry.state = precise['state']
                cache_entry.country = precise['country']
                cache_entry.display_name = precise['display_name']
                cache_entry.raw_response = precise['raw_response']
                cache_entry.geocoded_at = timezone.now()
                # Nominatim genuinely had nothing usable (not a failure -
                # see swallow_exceptions=False's comment above for the
                # failure case) - fall back to the nearest named place we
                # know of offline, rather than leaving this permanently
                # "unknown". Marked approximate so nothing mistakes it for
                # the actual place a photo was taken.
                if not cache_entry.locality:
                    fallback_name, fallback_country, _ = find_nearest_named_place(lat, lon)
                    if fallback_name:
                        cache_entry.locality = fallback_name
                        cache_entry.locality_is_approximate = True
                        if not cache_entry.country:
                            cache_entry.country = fallback_country
                else:
                    # A genuine, precise result -- available for later
                    # coordinates in this same run to reuse.
                    reuse_candidates.append((lat, lon, {
                        'locality': cache_entry.locality, 'county': cache_entry.county,
                        'state': cache_entry.state, 'country': cache_entry.country,
                        'display_name': cache_entry.display_name,
                    }))
                result['succeeded'] += 1
            except Exception as e:
                cache_entry.lookup_failed = True
                cache_entry.lookup_error = str(e)
                result['failed'] += 1
                log(f"Failed to geocode ({lat}, {lon}): {e}")

        try:
            with transaction.atomic():
                cache_entry.save()
        except IntegrityError:
            # Another concurrent run already inserted this exact
            # coordinate between our "what's missing" query (computed
            # once, up front) and this save -- the recurring hourly task
            # and the one-time backfill command can genuinely run at the
            # same time. Benign: adopt the winning row instead of letting
            # an unhandled IntegrityError crash the entire batch over one
            # coordinate, the same anti-pattern this session already
            # fixed elsewhere (face_extraction, assign_faces).
            log(f"Coordinate ({lat}, {lon}) already cached by a concurrent run -- reusing it.")
            cache_entry = GeocodeCache.objects.get(lat=lat, lon=lon)

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

    log(f"Done. {result['succeeded']} succeeded ({result['reused_nearby']} via nearby reuse, "
        f"no Nominatim call), {result['failed']} failed, {time.time() - start:.0f}s total.")
    return result
