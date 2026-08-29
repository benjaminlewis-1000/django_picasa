#! /usr/bin/env python

# Views backing the /api/mobile/... endpoints used by the mobile tagging
# app. Split out of api/views.py, which had grown to mix these in with
# the standard ModelViewSets and slideshow-facing endpoints.

import json

from django.conf import settings
from django.db.models import Count, Q
from django.http import HttpResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from face_manager.models import Face, Person


def _face_thumb(host_url, fid):
    """The {id, face_img_url} dict every mobile grid returns per face."""
    return {
        'id': fid,
        'face_img_url': (
            f"{host_url}/keyed_image/face_array/?id={fid}"
            f"&access_key={settings.RANDOM_ACCESS_KEY}"
        ),
    }


def _clamped_limit(request, default=15, maximum=120):
    """Parse ?limit=, falling back to `default` and capping at `maximum`."""
    try:
        n = int(request.query_params.get('limit', default))
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, maximum))


class ConfidentUnlabeledView(APIView):

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        # The tagging app only wants faces where a human has a clean
        # decision to make: declared_name is still the blank sentinel,
        # at least one poss_identN guess points to a *real* person, and
        # *none* of the five poss_identN slots is a sentinel row
        # (.ignore / .realignore / _NO_FACE_ASSIGNED_ / ..., per
        # settings.IGNORED_NAMES). A single sentinel guess anywhere in the
        # top five disqualifies the face. Empty (null) slots are fine.
        #
        # Regression test for a fixed bug: this used to do
        # `unlabeled[0].weight_1` unconditionally, which raised IndexError
        # the moment there were zero unlabeled faces -- the *goal* state
        # of the tagging workflow, not an edge case. It now just returns
        # whatever ids match, including none.
        has_real_guess = Q()
        no_sentinel_guess = Q()
        for i in range(1, Face.NUM_POSSIBLE_IDENTITIES + 1):
            slot = f'poss_ident{i}'
            not_sentinel = ~Q(**{f'{slot}__person_name__in': settings.IGNORED_NAMES})
            has_real_guess |= Q(**{f'{slot}__isnull': False}) & not_sentinel
            # An empty slot is fine; a filled slot must not be a sentinel.
            # (Spelled out rather than a bare ~Q(...__in) so a NULL slot,
            # whose person_name is NULL and fails `NOT IN`, still passes.)
            no_sentinel_guess &= Q(**{f'{slot}__isnull': True}) | not_sentinel

        unlabeled = (
            Face.objects
            .filter(declared_name__person_name=settings.BLANK_FACE_NAME)
            .filter(has_real_guess)
            .filter(no_sentinel_guess)
            .order_by('-weight_1')
        )
        unlabeled_ids = list(unlabeled.values_list('id', flat=True))

        js = {'unlabeled_ids': unlabeled_ids}
        return HttpResponse(json.dumps(js), content_type='application/json')


class LabelingGroupsView(APIView):
    """The mobile main-flow queue, grouped by top guess.

    Unlabeled faces whose poss_ident1 is a *real* person (sentinels
    excluded -- `.ignore`-topped faces belong to the ignore-review
    screen). Faces are grouped by that person; groups come back ordered
    by size (most candidate faces first), and each group's face_ids are
    ordered by weight_1 descending. The app freezes this order for the
    session so the "current person" doesn't shift as counts change.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        qs = (
            Face.objects
            .filter(declared_name__person_name=settings.BLANK_FACE_NAME)
            .filter(poss_ident1__isnull=False)
            .exclude(poss_ident1__person_name__in=settings.IGNORED_NAMES)
            .filter(Q(mobile_review_hidden__isnull=True) | Q(mobile_review_hidden=False))
        )

        counts = list(
            qs.values('poss_ident1', 'poss_ident1__person_name')
              .annotate(c=Count('id'))
              .order_by('-c', 'poss_ident1__person_name')
        )

        faces_by_person = {}
        for pid, fid in qs.order_by('-weight_1', '-id').values_list('poss_ident1', 'id'):
            faces_by_person.setdefault(pid, []).append(fid)

        groups = [
            {
                'person_id': row['poss_ident1'],
                'person_name': row['poss_ident1__person_name'],
                'count': row['c'],
                'face_ids': faces_by_person.get(row['poss_ident1'], []),
            }
            for row in counts
        ]

        js = {'groups': groups}
        return HttpResponse(json.dumps(js), content_type='application/json')


class UnlabeledMobileInfo(APIView):

    # Take a single Face ID from the list returned by ConfidentUnlabeledView,
    # and build all the data it would require to put that image on a page /
    # app screen with possible names and the URLs to assign the face that name.
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):

        selected_id = kwargs['id']

        host_url = f'https://{request.get_host()}/api'

        # Get: URL for full size image
        # URL for small image (face_array)
        # URL for large image (face_source)
        # URL to assign the face to no-one.
        # Names for various assignments and the URL and payload
        # to assign the person to that.

        face_img = f"{host_url}/keyed_image/face_array/?id={selected_id}&access_key={settings.RANDOM_ACCESS_KEY}"
        whole_img = f"{host_url}/keyed_image/face_source/?id={selected_id}&access_key={settings.RANDOM_ACCESS_KEY}"
        ignore_url = f"{host_url}/faces/{selected_id}/ignore_face/"
        ignore_payload = {'ignore_type': 'soft'}

        face_object = Face.objects.get(id = selected_id)

        # Whether this face is still an open tagging task. The mobile
        # queue is a snapshot taken at login/refresh; by the time the app
        # actually reaches a given face it may have been assigned in the
        # web app, ignored, or moved by the classifier. The app uses this
        # to silently skip faces that are no longer up for grabs.
        declared = face_object.declared_name
        declared_name = declared.person_name if declared is not None else None
        is_unassigned = declared_name == settings.BLANK_FACE_NAME

        people_foreign_keys = [face_object.poss_ident1, \
            face_object.poss_ident2, \
            face_object.poss_ident3, \
            face_object.poss_ident4, \
            face_object.poss_ident5, ]

        names = []
        for idx in range(len(people_foreign_keys)):
            weight = face_object.__dict__[f'weight_{idx + 1}']
            # Can't get a foreign key from the __dict__, so
            # use the list above

            person = people_foreign_keys[idx]
            if person is not None:
                name = person.person_name
                pid = person.id
                # store.get('api_url') + '/faces/' + faceId + '/assign_face_to_person/
                person_info = {
                    'name': name,
                    'confirm_patch_url': f"{host_url}/faces/{selected_id}/assign_face_to_person/",
                    'confirm_patch_data': {'declared_name_key': pid},
                    # disassociate_patch_url used to point at
                    # reject_association_app_api(), removed as dead code
                    # (see api/views.py history) -- no frontend this
                    # project has access to ever called it.
                    'person_id': pid,
                    'weight': weight,
                }
                names.append(person_info)

        js = {'face_img_url': face_img,
              'source_img_url': whole_img,
              'names': names,
              'ignore_url': ignore_url,
              'ignore_payload': ignore_payload,
              'is_unassigned': is_unassigned,
              'declared_name': declared_name,
              }

        return HttpResponse(json.dumps(js), content_type='application/json')


class ResetFace(APIView):

    # Given a Face ID, reset it: drop any name assignment and all
    # poss_identN guesses, and put it back in the unassigned pool so the
    # classifier re-processes it.
    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):

        selected_id = kwargs['id']
        face_object = Face.objects.get(id=selected_id)
        face_object.reset_to_pool()

        # Regression test for a fixed bug: this method had no return
        # statement, so DRF's dispatch() got None back instead of a
        # Response and raised AssertionError -- crashed on every call.
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')


class IgnoreCandidatesList(APIView):
    """Still-unlabeled faces whose top classifier guess (poss_ident1) is
    `.ignore`. The mobile "review ignored faces" grid bulk-confirms these
    as `.ignore` -- a rough analogue of the frontend's "confirm row"
    action on the `.ignore` person page.

    Returns a *random* sample each call (there are ~120k candidates; a
    fixed order would just replay the same high-weight faces). The app
    dedupes by id across pages, so the legacy `offset` param is accepted
    but ignored.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        limit = _clamped_limit(request)

        host_url = f'https://{request.get_host()}/api'
        faces = (
            Face.objects
            .filter(
                declared_name__person_name=settings.BLANK_FACE_NAME,
                poss_ident1__person_name=settings.SOFT_IGNORE_NAME,
            )
            .filter(Q(mobile_review_hidden__isnull=True) | Q(mobile_review_hidden=False))
            .order_by('?')
            .values_list('id', flat=True)[:limit]
        )

        js = {'faces': [_face_thumb(host_url, fid) for fid in faces]}
        return HttpResponse(json.dumps(js), content_type='application/json')


class BulkConfirmIgnore(APIView):
    """Bulk-resolve a batch of "poss_ident1 == .ignore" candidates:

      - `confirm_ids`: assign the face to `.ignore` (associate_person) --
        confirms the classifier's guess.
      - `hide_ids`   : just set Face.mobile_review_hidden = True. The face
        is untouched otherwise (still an unlabeled proposed .ignore); it
        simply stops showing up in the mobile review grid.

    Synchronous; stale/mismatched ids are skipped, not errored.
    """

    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):
        confirm_ids = request.data.get('confirm_ids') or []
        # `reject_ids` kept as an accepted alias for older app builds.
        hide_ids = request.data.get('hide_ids') or request.data.get('reject_ids') or []

        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank_name = settings.BLANK_FACE_NAME

        confirmed = skipped = 0

        for fid in confirm_ids:
            face = Face.objects.filter(id=fid).first()
            if (
                face is None
                or face.declared_name is None
                or face.declared_name.person_name != blank_name
                or face.poss_ident1_id != ignore_person.id
            ):
                skipped += 1
                continue
            face.associate_person(ignore_person.id)  # blank -> declared .ignore
            confirmed += 1

        hidden = (
            Face.objects
            .filter(
                id__in=hide_ids,
                declared_name__person_name=blank_name,
                poss_ident1__person_name=settings.SOFT_IGNORE_NAME,
            )
            .update(mobile_review_hidden=True)
        )
        skipped += len(hide_ids) - hidden

        js = {'confirmed': confirmed, 'hidden': hidden, 'skipped': skipped}
        return HttpResponse(json.dumps(js), content_type='application/json')


class HideFromMobile(APIView):
    """Set Face.mobile_review_hidden = True for a batch of faces so the
    mobile app stops surfacing them -- backs the main labeling screen's
    "Skip" (a soft "not now, and don't ask again here"). The face is
    otherwise untouched; the web app and classifier ignore this flag.
    LabelingGroupsView already filters it out, so hidden faces drop off
    the queue on the next refresh/login.

    Body: `{face_ids: [...]}`. Idempotent; unknown ids are simply no-ops.
    """

    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):
        face_ids = request.data.get('face_ids') or []
        hidden = (
            Face.objects
            .filter(id__in=face_ids)
            .update(mobile_review_hidden=True)
        )
        js = {'hidden': hidden}
        return HttpResponse(json.dumps(js), content_type='application/json')


class VerifyCandidatesList(APIView):
    """One *named* person's unconfirmed face assignments, for the mobile
    "verify people" grid. (`.ignore` etc. handled by
    VerifyIgnoreCandidatesList instead.)

    The app pins a person for the session by passing `?person_id=`: as
    long as that person still has unverified faces (and isn't in
    `exclude`), we keep serving them (and their live `unverified_count`).
    Only once they're exhausted (or on the first, unpinned call) do we
    pick the biggest remaining pile -- with a deterministic tiebreaker so
    equal-sized piles don't flip-flop between loads. The app awaits its
    bulk_verify write before reloading, so this count is authoritative.

    Query params: `limit` (default 15), `person_id` (pin), `exclude`
    (comma-separated person ids to skip this session).
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        limit = _clamped_limit(request)

        exclude_ids = []
        for tok in (request.query_params.get('exclude', '') or '').split(','):
            tok = tok.strip()
            if tok.isdigit():
                exclude_ids.append(int(tok))

        pin_id = request.query_params.get('person_id', '')
        pin_id = int(pin_id) if pin_id.strip().isdigit() else None

        unverified = (
            Face.objects
            .filter(validated=False, declared_name__isnull=False)
            .exclude(declared_name__person_name__in=settings.IGNORED_NAMES)
            .exclude(declared_name_id__in=exclude_ids)
        )

        pid = person_name = None
        count = 0
        if pin_id is not None and pin_id not in exclude_ids:
            pinned = (
                unverified.filter(declared_name_id=pin_id)
                .values('declared_name_id', 'declared_name__person_name')
                .annotate(c=Count('id'))
                .order_by('declared_name_id')
                .first()
            )
            if pinned:
                pid = pinned['declared_name_id']
                person_name = pinned['declared_name__person_name']
                count = pinned['c']

        if pid is None:
            top = (
                unverified
                .values('declared_name_id', 'declared_name__person_name')
                .annotate(c=Count('id'))
                .order_by('-c', 'declared_name_id')
                .first()
            )
            if not top:
                js = {'person_id': None, 'person_name': None,
                      'unverified_count': 0, 'faces': []}
                return HttpResponse(json.dumps(js), content_type='application/json')
            pid = top['declared_name_id']
            person_name = top['declared_name__person_name']
            count = top['c']

        host_url = f'https://{request.get_host()}/api'
        face_ids = (
            Face.objects
            .filter(declared_name_id=pid, validated=False)
            .order_by('?')
            .values_list('id', flat=True)[:limit]
        )

        js = {
            'person_id': pid,
            'person_name': person_name,
            'unverified_count': count,
            'faces': [_face_thumb(host_url, fid) for fid in face_ids],
        }
        return HttpResponse(json.dumps(js), content_type='application/json')


class VerifyIgnoreCandidatesList(APIView):
    """Faces already declared `.ignore` but not human-verified. Flat
    random sample (no person grouping -- they're all `.ignore`). The
    "verify ignored" grid confirms the good ones and resets the rest
    (a real person the classifier wrongly ignored) for reprocessing.

    Query param: `limit` (default 15).
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        limit = _clamped_limit(request)

        host_url = f'https://{request.get_host()}/api'
        face_ids = (
            Face.objects
            .filter(declared_name__person_name=settings.SOFT_IGNORE_NAME, validated=False)
            .order_by('?')
            .values_list('id', flat=True)[:limit]
        )
        js = {'faces': [_face_thumb(host_url, fid) for fid in face_ids]}
        return HttpResponse(json.dumps(js), content_type='application/json')


class BulkVerify(APIView):
    """Bulk-resolve a batch of unverified faces (used by both the "verify
    people" and "verify ignored" grids):

      - `verify_ids`: confirm the current assignment
        (verify_person_in_image -> validated = True).
      - `reset_ids` : "wrong" -- send the face to the unassigned pool
        (declared_name = blank sentinel, guesses cleared) for
        re-classification.

    Synchronous; stale/mismatched ids are skipped, not errored.
    """

    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):
        verify_ids = request.data.get('verify_ids') or []
        reset_ids = request.data.get('reset_ids') or []

        verified = reset = skipped = 0

        for fid in verify_ids:
            face = Face.objects.filter(id=fid).first()
            if face is None or face.validated or face.declared_name is None:
                skipped += 1
                continue
            face.verify_person_in_image()  # validated = True, decrements unverified
            verified += 1

        for fid in reset_ids:
            face = Face.objects.filter(id=fid).first()
            if face is None:
                skipped += 1
                continue
            face.reset_to_pool()
            reset += 1

        js = {'verified': verified, 'reset': reset, 'skipped': skipped}
        return HttpResponse(json.dumps(js), content_type='application/json')


class MobileNameList(APIView):
    # Get a list of all defined person names, excluding the sentinel
    # "person" rows (blank/ignore placeholders) that aren't real people.
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        names = list(
            Person.objects.exclude(person_name__in=settings.IGNORED_NAMES)
            .order_by('person_name')
            .values_list('person_name', flat=True)
        )

        js = {'name_list': names}
        return HttpResponse(json.dumps(js), content_type='application/json')
