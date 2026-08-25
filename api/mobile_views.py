#! /usr/bin/env python

# Views backing the /api/mobile/... endpoints used by the mobile tagging
# app. Split out of api/views.py, which had grown to mix these in with
# the standard ModelViewSets and slideshow-facing endpoints.

import json

from django.conf import settings
from django.http import HttpResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from face_manager.models import Face, Person


class ConfidentUnlabeledView(APIView):

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        # Regression test for a fixed bug: this used to do
        # `unlabeled[0].weight_1` unconditionally, which raised IndexError
        # the moment there were zero unlabeled faces -- the *goal* state
        # of the tagging workflow, not an edge case, so this crashed
        # whenever tagging was fully caught up. Just return whatever ids
        # are there, including none.
        unlabeled = Face.objects.filter(declared_name__person_name=settings.BLANK_FACE_NAME).order_by('-weight_1')
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

    # Given a Face ID, reset it - make sure that there are no assigned names for that
    # face. It will trigger a re-classification.
    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):

        selected_id = kwargs['id']

        face_object = Face.objects.get(id = selected_id)
        face_object.clear_person()

        # Regression test for a fixed bug: this method had no return
        # statement, so DRF's dispatch() got None back instead of a
        # Response and raised AssertionError -- this crashed on every
        # single call, not just an edge case.
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
