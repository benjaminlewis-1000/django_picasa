"""API views backing the frontend's Tools-tab "Fix Geocoding" screen -
letting a person review/correct find_nearest_metro()'s offline guess
(filepopulator/geocode.py) against a photo's precise reverse-geocoded
locality, and either confirm it or point it at a different real place.

Kept in its own module rather than api/views.py (already large) - same
reasoning as mobile_views.py being separate.
"""

import json

from django.db.models import Count, Q
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.http import HttpResponse

from filepopulator.geocode import _haversine_km, resolve_named_place, search_major_places
from filepopulator.models import GeocodeCache


# GeocodeCache is keyed per-coordinate (rounded to ~11m - see its own
# ROUND_DECIMALS), not per place, so a "row per city" review table doesn't
# exist in the data as-is - it has to be built by grouping. Grouped by
# (locality, country, nearest_metro_name) rather than by nearest_metro_name
# alone: the same metro name can legitimately be the correct pick for a
# totally different, unrelated locality elsewhere, so a name-only group
# would let a correction for one place silently "fix" an unrelated
# already-correct one that happens to share a metro name.
def _group_key(row):
    return (row['locality'], row['country'], row['nearest_metro_name'])


def _build_review_groups():
    """Groups every GeocodeCache row with a metro pick by (locality,
    country, nearest_metro_name), computing image/video counts via two
    separate grouped queries rather than combining two Count(..., distinct)
    aggregates over two different to-many relations (images, videos) in
    one annotate() call -- that combination is a known Django ORM
    correctness trap (the join against each relation multiplies the other
    relation's rows, inflating both counts) rather than just a perf
    concern. Three queries total regardless of how many groups there are,
    with the actual grouping/merging done in Python."""
    base_rows = list(
        GeocodeCache.objects
        .filter(nearest_metro_name__isnull=False)
        .values('locality', 'state', 'country', 'locality_is_approximate', 'nearest_metro_name', 'lat', 'lon',
                 'nearest_metro_distance_km', 'metro_validated', 'metro_override')
    )
    image_counts = {
        _group_key(r): r['c'] for r in
        GeocodeCache.objects.filter(nearest_metro_name__isnull=False)
        .values('locality', 'country', 'nearest_metro_name')
        .annotate(c=Count('images', distinct=True))
    }
    video_counts = {
        _group_key(r): r['c'] for r in
        GeocodeCache.objects.filter(nearest_metro_name__isnull=False)
        .values('locality', 'country', 'nearest_metro_name')
        .annotate(c=Count('videos', distinct=True))
    }

    groups = {}
    for row in base_rows:
        key = _group_key(row)
        if key not in groups:
            # Best-effort US state for the metro pick itself - looked up
            # against the same offline gazetteer a correction would use
            # (nearest match if the name exists more than once worldwide),
            # not persisted anywhere. Only major_places.csv's US rows have
            # a human-readable admin1_code (the 2-letter state, e.g. "CA")
            # - every other country's admin1_code is a bare GeoNames
            # number, not worth showing. Silently None for a metro name
            # that came from a Nominatim-resolved correction not in the
            # gazetteer at all (a small hometown, a park) - nothing to
            # look up in that case.
            metro_state = _major_place_state(row['nearest_metro_name'], row['lat'], row['lon'])
            groups[key] = {
                'locality': row['locality'],
                'locality_is_approximate': row['locality_is_approximate'],
                'state': row['state'],
                'country': row['country'],
                'metro_name': row['nearest_metro_name'],
                'metro_state': metro_state,
                'metro_distance_km': row['nearest_metro_distance_km'],
                'lat': row['lat'],
                'lon': row['lon'],
                'metro_validated': row['metro_validated'],
                'metro_override': row['metro_override'],
                'num_coords': 0,
                'num_images': image_counts.get(key, 0) + video_counts.get(key, 0),
            }
        groups[key]['num_coords'] += 1

    # Sort key, in priority order: validated sinks to the bottom; within
    # "not yet validated", a row with no precise locality at all (a
    # failed/empty reverse-geocode - see CLAUDE.md's note on this) sinks
    # below every row that actually has one to compare the metro pick
    # against, since there's nothing to usefully compare there; then
    # alphabetical.
    return sorted(
        groups.values(),
        key=lambda g: (g['metro_validated'], g['locality'] is None, g['locality'] or '', g['country'] or ''),
    )


def _group_filter(locality, country, metro_name):
    # locality is nullable (failed precise lookups still get a metro
    # pick) - Q(locality=None) matches NULL correctly via Django's ORM,
    # same as the explicit __isnull checks used elsewhere in this app.
    return (
        Q(locality=locality) & Q(country=country) & Q(nearest_metro_name=metro_name)
    )


class GeocodeReviewListView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        results = _build_review_groups()
        validated_count = sum(1 for g in results if g['metro_validated'])

        js = {
            'metrics': {'total': len(results), 'validated': validated_count},
            'results': results,
        }
        return HttpResponse(json.dumps(js), content_type='application/json')


class GeocodeReviewSearchPlacesView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        query = request.query_params.get('q', '')
        matches = search_major_places(query, limit=10)
        js = {'results': [
            {
                'name': m['name'], 'country_code': m['country_code'], 'population': m['population'],
                # Human-readable only for US rows - see _major_place_state's
                # own comment. Frontend falls back to showing country_code
                # for everything else.
                'state': m['admin1_code'] if m['country_code'] == 'US' else None,
            }
            for m in matches
        ]}
        return HttpResponse(json.dumps(js), content_type='application/json')


class GeocodeReviewActionView(APIView):
    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):
        data = request.data
        locality = data.get('locality')
        country = data.get('country')
        metro_name = data.get('metro_name')
        action = data.get('action')

        if action not in ('validate', 'correct'):
            return HttpResponse(
                json.dumps({'success': False, 'error': "action must be 'validate' or 'correct'"}),
                content_type='application/json', status=400,
            )

        rows = list(GeocodeCache.objects.filter(_group_filter(locality, country, metro_name)))
        if not rows:
            return HttpResponse(
                json.dumps({'success': False, 'error': 'No matching geocode group found - it may have already been edited.'}),
                content_type='application/json', status=404,
            )

        if action == 'validate':
            for row in rows:
                row.metro_validated = True
                row.save(update_fields=['metro_validated'])
            return HttpResponse(json.dumps({'success': True}), content_type='application/json')

        # action == 'correct'
        corrected_name = (data.get('metro_name_correction') or '').strip()
        if not corrected_name:
            return HttpResponse(
                json.dumps({'success': False, 'error': 'metro_name_correction is required to correct a place.'}),
                content_type='application/json', status=400,
            )

        try:
            resolved = _resolve_correction(corrected_name, rows[0].lat, rows[0].lon)
        except Exception as e:
            # A real Nominatim failure (timeout/429/network) - swallow_
            # exceptions=False on the forward-geocode RateLimiter (see
            # geocode.py) means this now actually reaches here instead of
            # silently looking identical to "not a real place" below.
            # Distinct status/message so the frontend can tell a user
            # "try again" rather than "that place doesn't exist".
            return HttpResponse(
                json.dumps({'success': False, 'error': f'Could not reach the geocoding service - try again in a moment. ({e})'}),
                content_type='application/json', status=503,
            )
        if resolved is None:
            return HttpResponse(
                json.dumps({'success': False, 'error': f'"{corrected_name}" isn\'t a place we could recognize.'}),
                content_type='application/json', status=400,
            )

        for row in rows:
            row.nearest_metro_name = resolved['name']
            row.nearest_metro_distance_km = _haversine_km(row.lat, row.lon, resolved['lat'], resolved['lon'])
            row.metro_validated = True
            row.metro_override = True
            row.save(update_fields=[
                'nearest_metro_name', 'nearest_metro_distance_km', 'metro_validated', 'metro_override',
            ])

        return HttpResponse(json.dumps({
            'success': True,
            'metro_name': resolved['name'],
            'metro_state': resolved['state'],
            'metro_distance_km': _haversine_km(rows[0].lat, rows[0].lon, resolved['lat'], resolved['lon']),
        }), content_type='application/json')


def _exact_major_place_matches(name):
    return [p for p in search_major_places(name, limit=1000) if p['name'].lower() == name.lower()]


def _major_place_state(name, lat, lon):
    """US-only best-effort state lookup for a name already known to be a
    major_places.csv entry (major_places.csv's admin1_code is only a
    human-readable state abbreviation for US rows - every other country's
    is a bare GeoNames number). Returns None if the name isn't in the
    gazetteer at all, or the nearest match isn't in the US."""
    if not name:
        return None
    matches = _exact_major_place_matches(name)
    if not matches:
        return None
    best = min(matches, key=lambda p: _haversine_km(lat, lon, p['lat'], p['lon']))
    return best['admin1_code'] if best['country_code'] == 'US' else None


def _resolve_correction(corrected_name, lat, lon):
    """Tier 1: exact (case-insensitive) match against the offline
    major_places.csv gazetteer, nearest-to-this-group if the name matches
    more than one place worldwide - fast, no network, covers the common
    case of correcting to another well-known metro. Tier 2: Nominatim
    forward-geocode (resolve_named_place) for anything not in that curated
    list - a smaller hometown, a national park, a landmark. Returns
    {'name', 'lat', 'lon', 'state'} (state is US-only, same reasoning as
    _major_place_state above - Nominatim's own 'state' is a real name for
    any country, so tier 2 doesn't have that same limitation) or None if
    neither tier resolves it."""
    exact_matches = _exact_major_place_matches(corrected_name)
    if exact_matches:
        best = min(exact_matches, key=lambda p: _haversine_km(lat, lon, p['lat'], p['lon']))
        state = best['admin1_code'] if best['country_code'] == 'US' else None
        return {'name': best['name'], 'lat': best['lat'], 'lon': best['lon'], 'state': state}

    resolved = resolve_named_place(corrected_name)
    if resolved is None:
        return None
    return {'name': resolved['name'], 'lat': resolved['lat'], 'lon': resolved['lon'], 'state': resolved['state']}
