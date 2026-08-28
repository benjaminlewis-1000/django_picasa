#! /usr/bin/env python

# Views backing the /api/mobile/... endpoints used by the mobile tagging
# app. Split out of api/views.py, which had grown to mix these in with
# the standard ModelViewSets and slideshow-facing endpoints.

import json

from django.conf import settings
from django.db.models import Q
from django.http import HttpResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from face_manager.models import Face, Person


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
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)

        # Decrement the previous *real* owner's counts, if there was one.
        prev = face_object.declared_name
        if prev is not None and prev.person_name != settings.BLANK_FACE_NAME:
            prev.decrement_assigned()
            if not face_object.validated:
                prev.decrement_unverified()

        # Wipe all five poss_identN guesses + weights (this also saves).
        face_object.set_possibles_zero()

        # Put the face back in the unassigned pool as the *blank sentinel
        # Person*, NOT NULL. The old clear_person() left declared_name
        # NULL, which is invisible to both the "Unassigned" bucket and to
        # assign_faces' re-classification -- both filter on
        # declared_name__person_name == BLANK_FACE_NAME -- so a reset face
        # just vanished instead of returning to the pool.
        if face_object.declared_name_id != blank.id:
            face_object.declared_name = blank
            face_object.validated = False
            face_object.written_to_photo_metadata = False
            blank.increment_assigned()
            blank.increment_unverified()
            face_object.save()

        # Regression test for a fixed bug: this method had no return
        # statement, so DRF's dispatch() got None back instead of a
        # Response and raised AssertionError -- crashed on every call.
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')


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
