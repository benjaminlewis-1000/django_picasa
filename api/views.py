#! /usr/bin/env python 

# Authentication: https://simpleisbetterthancomplex.com/tutorial/2018/11/22/how-to-implement-token-authentication-using-django-rest-framework.html
#from api.serializers import UserSerializer, GroupSerializer, ImageFileSerializer, DirectorySerializer, ParameterSerializer
from django.conf import settings
from django.contrib.auth import logout as django_logout
from django.contrib.auth.models import User, Group
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.db import connections
from django.db.models import Count
from django.db.models import Q
from django.http import HttpResponse, Http404, FileResponse
from django.shortcuts import render
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django_filters.rest_framework import DjangoFilterBackend
from face_manager.models import Person, Face
from face_manager.live_counts import compute_live_face_counts
from filepopulator.models import ImageFile, Directory, VideoFile, PhotoCollection
from io import BytesIO
from PIL import Image, ExifTags, ImageDraw
from queue import Queue
from rest_framework import viewsets, filters
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.reverse import reverse, reverse_lazy
from rest_framework.views import APIView
from django.utils.functional import SimpleLazyObject
from face_manager.tasks import api_bulk_operation
from time import sleep
import api.serializers as api_ser
from api.models import ScheduleRule, SlideshowConfig, resolve_active_slideshow_config
from queue import Queue
import base64
import common
import cv2
import subprocess
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import numpy as np
import mimetypes
import os
import PIL
import random
import secrets
import re
import requests
import string
from .permissions import HasSlideshowKeyOrAuthenticated
from .throttles import SlideshowListThrottle, SlideshowMediaThrottle, AccessKeyMediaThrottle
import threading
import threading
import time
from filters.mixins import (
    FiltersMixin,
)


background_queue = Queue()

# How much extra room to leave around a face's own bounding box when
# generating a person's highlight image (FaceViewSet.highlight_from_face).
# 1.0 = crop tight to the face, 1.5 = 50% more space on each side.
HIGHLIGHT_PADDING_FACTOR = 1.6

# Lazy rather than a plain module-level query: this module gets imported at
# Django startup (URL resolution, system checks) well before any request --
# and before any test's setUp has had a chance to seed these sentinel Person
# rows. An eager query here crashed the entire app on any DB that doesn't
# already have '.ignore'/'.realignore'/BLANK_FACE_NAME rows -- which includes
# every fresh install and every CI run (production only avoided this because
# someone seeded them by hand once). SimpleLazyObject defers the query to
# first actual attribute access, by which point the DB is guaranteed to be
# ready. See CLAUDE.md's "Bootstrapping a fresh DB from scratch" note --
# this stops the crash but doesn't auto-create the rows themselves, which is
# a separate gap.
soft_ignore_person = SimpleLazyObject(lambda: Person.objects.filter(person_name='.ignore')[0])
hard_ignore_person = SimpleLazyObject(lambda: Person.objects.filter(person_name='.realignore')[0])
blank_person = SimpleLazyObject(lambda: Person.objects.filter(person_name=settings.BLANK_FACE_NAME)[0])

def bulk_thread(dataframe: dict):
    # print("Got a data load of ", dataframe)

    assert 'face_id_list' in dataframe.keys()
    assert 'operation' in dataframe.keys()
    assert 'current_person_id' in dataframe.keys()

    face_id_list = dataframe['face_id_list']
    operation = dataframe['operation']
    current_person_id = dataframe['current_person_id']

    assert type(face_id_list) is list
    assert type(current_person_id) is int
    assert operation in ['close_unassigned', 'close_ignored', 'close_assigned', 'confirm_proposed', 'verify_face', 'flag_for_review']

    for face_id in face_id_list:
        assert type(face_id) is int
        try:
            face = Face.objects.get(id=face_id)
        except Face.DoesNotExist:
            # Missing `continue` used to let execution fall through to the
            # operation branches below with `face` either unbound (first
            # iteration) or still holding the *previous* iteration's Face
            # (later iterations) -- silently operating on the wrong face,
            # or raising UnboundLocalError, either way swallowed by
            # background_bulk_processor()'s blanket except.
            print(f"Face not found for ID {face_id}")
            continue

        if operation == 'close_unassigned':
            # Previously /api/face_id/ignore_face with {ignore_type: 'soft'}
            # Set it to the soft ignore person, '.ignore'
            face.associate_person(soft_ignore_person.id)
        elif operation == 'close_ignored':
            # Previously /api/face_id/ignore_face with {ignore_type: 'hard'}
            # Set to the hard ignore person, '.realignore', but only
            # if they have previously been set to '.ignore'
            if face.declared_name == soft_ignore_person \
                or face.declared_name == hard_ignore_person \
                or face.poss_ident1 == soft_ignore_person:
                face.associate_person(hard_ignore_person.id)
            else:
                print("Person currently assigned is not .ignore or .realignore.")
        elif operation == 'close_assigned':
            # Previously /api/face_id/reject_association with {unassociate_id: current_person_id}
            #
            # reject_association() only knows how to cross a candidate off
            # the face's poss_identN "possible match" list - it asserts
            # current_person_id is one of those candidates, which is only
            # true when this is declining a proposed match. "Remove from
            # person" (and undoing a confirm) fire this operation on faces
            # that are already DECLARED to current_person_id instead -
            # never in poss_identN - so that assert used to raise every
            # time, get silently swallowed by background_bulk_processor's
            # except below, and the face never actually moved. Route based
            # on which of the two cases this actually is.
            possible_ids = [pid for pid in (
                face.poss_ident1_id, face.poss_ident2_id, face.poss_ident3_id,
                face.poss_ident4_id, face.poss_ident5_id,
            ) if pid is not None]
            if current_person_id in possible_ids:
                # Declining a proposed candidate match - original behavior.
                face.reject_association(current_person_id)
            elif face.declared_name_id == current_person_id:
                # Actually declared to this person - peel off the declared
                # name tag itself, same mechanism close_unassigned uses to
                # reassign a face to '.ignore'. Record the former person as
                # rejected first - associate_person() doesn't touch
                # rejected_fields on its own, and without this,
                # classify_unassigned() would be free to immediately
                # re-propose the exact assignment a human just removed
                # (e.g. re-suggesting '.ignore' right after "Remove from
                # person" took a face out of it).
                face.add_to_rejected_fields(current_person_id)
                face.associate_person(blank_person.id)
            else:
                print(f"close_assigned: face {face.id} is neither declared to nor "
                      f"has a possible match for person {current_person_id} - no-op.")
        elif operation == 'confirm_proposed':
            # Update the face object fields accordingly
            face.associate_person(current_person_id)
        elif operation == 'verify_face':
            face.verify_person_in_image()
        elif operation == 'flag_for_review':
            # Frontend's ".ignore" gallery X button, for a still-proposed
            # (undeclared) candidate specifically (gallery.jsx/lazyImg.jsx,
            # 2026-09-16 per the user's own request) - same mechanism the
            # mobile app's ignore-review flow already uses
            # (mobile_views.py's BulkConfirmIgnore) to mean "looked at,
            # might actually be someone" rather than an outright reject.
            # Doesn't touch poss_ident1/declared_name at all - the face
            # stays exactly the candidate it already was, just moved into
            # the "Flagged for review" sub-queue (PersonParamView's
            # `flagged` param) instead of the main .ignore queue.
            if face.poss_ident1_id == soft_ignore_person.id and not face.mobile_review_hidden:
                face.mobile_review_hidden = True
                face.save(update_fields=['mobile_review_hidden'])
            else:
                print(f"flag_for_review: face {face.id} is not a still-proposed "
                      f".ignore candidate (or is already flagged) - no-op.")

def background_bulk_processor():
    print("Background processor initiating")
    while True:
        if background_queue.empty():
            # Close any DB connection this thread opened while handling the
            # last item -- Django only auto-closes connections at the end
            # of a normal request/response cycle, which this loop never
            # participates in. Left open, it holds a connection to
            # whatever database is active (including a test run's
            # test_picasa) indefinitely, which made `manage.py test`'s own
            # teardown fail with "database is being accessed by other
            # users" even after the thread itself stopped blocking process
            # exit (see the daemon=True note below).
            connections.close_all()
            time.sleep(1)
        else:
            try:
                data = background_queue.get()
                assert type(data) is dict
                # print("Got an item", data)
                bulk_thread(data)
            except Exception as e:
                print(f"Error: {e} in background_bulk_processor, which is called by /api/faces/bulk_operation")


    
# daemon=True: this loop is meant to run forever in the background, never
# to keep the interpreter alive on its own. A non-daemon thread here used
# to block manage.py test/CI's process exit indefinitely after tests
# finished -- Python won't exit while a non-daemon thread is alive, and
# this thread's held-open DB connection also made the test runner's
# post-run `DROP DATABASE test_picasa` fail with "being accessed by other
# users". See CLAUDE.md's "Testing gotcha" note.
#
# Lazily started (NOT as an import-time side effect) -- added 2026-09-18
# after this being an import-time side effect was found to be one of two
# real blockers for `manage.py test --parallel`: Django's --parallel
# forks worker processes AFTER Django/apps are already set up in the
# parent, and POSIX fork() semantics mean only the calling thread
# survives in each child while any DB connection this thread was
# holding gets duplicated (same underlying socket) across parent and
# every fork -- whichever side closes/reuses it first breaks the rest
# (InterfaceError: connection already closed), on essentially whichever
# tests happen to run in that window. Starting the thread lazily, on
# first real use inside an already-fully-forked worker process (or in
# a normal non-parallel run), means no connection is ever open across a
# fork boundary in the first place. See CLAUDE.md's "parallel test
# blockers" investigation for the full diagnosis.
work_thread = None
_work_thread_lock = threading.Lock()


def _ensure_background_worker_started():
    global work_thread
    with _work_thread_lock:
        if work_thread is None or not work_thread.is_alive():
            work_thread = threading.Thread(target=background_bulk_processor, daemon=True)
            work_thread.start()

def render_404(request, message):
    
    response = render(request, '404.html', context = {'error': message})
    response.status_code = 404
    return response

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    # The permission class means that the user must be
    # logged in to see this. 
    permission_classes = (IsAuthenticated,)
    queryset = User.objects.all()
    serializer_class = api_ser.UserSerializer

class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    permission_classes = (IsAuthenticated,)
    queryset = Group.objects.all()
    serializer_class = api_ser.GroupSerializer

class ImageViewSet(FiltersMixin, viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    permission_classes = (IsAuthenticated,)

    queryset = ImageFile.objects.all()
    # def get_queryset(self):
    #     return ImageFile.objects.all()
        
    serializer_class = api_ser.ImageFileSerializer

    filter_backends = (filters.OrderingFilter,)

    ordering_fields = ('id', 'filename')
    ordering = ('id')
    
    filter_mappings = {
        'id': 'id',
        'filename': 'filename__icontains',
    }


class VideoViewSet(FiltersMixin, viewsets.ModelViewSet):
    """Phase 4 (slideshow serving) -- mirrors ImageViewSet. Slideshow-
    reachable (not just IsAuthenticated) since the slideshow client has
    no session auth, same reasoning as filteredImagesView/KeyedImageView
    below."""
    permission_classes = (HasSlideshowKeyOrAuthenticated,)

    queryset = VideoFile.objects.all()
    serializer_class = api_ser.VideoFileSerializer

    filter_backends = (filters.OrderingFilter,)

    ordering_fields = ('id', 'filename')
    ordering = ('id')

    filter_mappings = {
        'id': 'id',
        'filename': 'filename__icontains',
    }

# Use URL filters https://github.com/manjitkumar/drf-url-filters
class DirectoryViewSet(FiltersMixin, viewsets.ModelViewSet):

    permission_classes = (IsAuthenticated,)

    queryset = Directory.objects.all()
    serializer_class = api_ser.DirectorySerializer
    filter_backends = (filters.OrderingFilter,)

    ordering_fields = ('id', 'dir_path', 'mean_datesec')
    ordering = ('id')

    filter_mappings = {
        'id': 'id',
        'dir_path': 'dir_path__icontains',
        'mean_datesec': 'mean_datesec',
        'mean_datesec__lte': 'mean_datesec__lte',
        'mean_datesec__gte': 'mean_datesec__gte',
        'time': 'mean_datesec',
    }
        
class PersonViewSet(viewsets.GenericViewSet):
    # Deliberately NOT a ModelViewSet -- no list/create/retrieve/update/
    # destroy. PersonSerializer (removed) used to expose face_declared as
    # a fully nested list of hyperlinked Face URLs, which for a
    # large-gallery person (the blank sentinel, ~99k faces; .ignore,
    # ~131k faces) meant building six-figure numbers of URLs in pure
    # Python on a single request -- no DB query involved, just CPU/RAM,
    # confirmed hanging 85+ seconds and using 7GB+ RSS on a real request
    # before being killed. This was always latent but never actually
    # reachable: PersonSerializer.get_num_possibilities was commented out
    # while still declared as a SerializerMethodField, so every list/
    # retrieve request 500'd on AttributeError before ever reaching
    # face_declared. Fixing that crash (see live_counts.py) would have
    # made this landmine reachable, so the generic CRUD actions are
    # removed instead -- GenericViewSet keeps get_object()/get_queryset()
    # (used by the two real actions below) without auto-registering any
    # of them.
    permission_classes = (IsAuthenticated,)
    queryset = Person.objects.all()

    @action(detail=True, methods=['put'])
    def toggle_further_unlikely(self, request, pk=None):
        # Accessible as <root>/api/people/<face_id>/toggle_further_unlikely/
        # Accept: HTML PUT
        # Given face_id in the URL, toggle the further_images_unlikely property. 
        person = self.get_object()

        def err_404(message=""):
            msg_start = f'Invalid toggle unlikely request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        unlikely = person.further_images_unlikely
        person.further_images_unlikely = not unlikely
        person.save()

        js = {'success': True, }
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['put'])
    def rename(self, request, pk=None):
        # Accessible as <root>/api/people/<person_id>/rename/
        # Accept: HTML PUT, body: {"person_name": "New Name"}
        person = self.get_object()

        if person.person_name in ('_NO_FACE_ASSIGNED_', 'Unassigned', '.ignore'):
            js = {'success': False, 'error': 'This person cannot be renamed.'}
            return HttpResponse(json.dumps(js), content_type='application/json', status=400)

        new_name = (request.data.get('person_name') or '').strip()
        if not new_name:
            js = {'success': False, 'error': 'Name cannot be empty.'}
            return HttpResponse(json.dumps(js), content_type='application/json', status=400)

        if Person.objects.filter(person_name=new_name).exclude(pk=person.pk).exists():
            js = {'success': False, 'error': 'A person with that name already exists.'}
            return HttpResponse(json.dumps(js), content_type='application/json', status=400)

        person.person_name = new_name
        person.save()

        js = {'success': True, 'person_name': person.person_name}
        return HttpResponse(json.dumps(js), content_type='application/json')

class PersonListView(APIView):
    permission_classes = (IsAuthenticated,)
    def get(self, request, *args, **kwargs):
        params = self.request.query_params

        # ?lightweight=true - net_slideshow's settings page People picker
        # (autocomplete sorted by face count) is the only caller of this
        # branch. The full response below computes num_possibilities/
        # num_unverified_faces/video-image splits via a threaded live-
        # count fan-out plus extra per-sentinel queries, purpose-built
        # for the facewires admin sidebar - none of which this picker
        # uses. This is one plain annotate(Count(...)) aggregate query
        # instead, excludes the blank/.ignore sentinel rows (meaningless
        # as slideshow filters), and skips straight to returning, so it
        # can't regress anything the default (unparemeterized) response
        # already does for existing callers.
        if params.get('lightweight', '').lower() == 'true':
            people = (
                Person.objects.exclude(person_name__in=settings.IGNORED_NAMES)
                .annotate(num_faces=Count('face_declared'))
                .order_by('-num_faces', 'person_name')
                .values('person_name', 'num_faces')
            )
            results = [{'person_name': p['person_name'], 'num_faces': p['num_faces']} for p in people]
            js = {'count': len(results), 'next': None, 'previous': None, 'results': results}
            return HttpResponse(json.dumps(js), content_type='application/json')

        all_people = list(Person.objects.all())
        num_people = len(all_people)
        domain_name = 'https://' + os.environ['API_DOMAIN'] + '/api/people'

        # num_faces/num_possibilities/num_unverified_faces are computed
        # live (no cached columns -- see face_manager/live_counts.py),
        # split across a few DB connections in parallel: each Postgres
        # connection is its own backend process, so this gets genuine
        # multi-core speedup rather than fighting Python's GIL.
        # Benchmarked against real production data (915 people): ~0.2s
        # at 4 threads with the two covering indexes in place, vs. ~1s
        # single-threaded and ~4.5s for a naive per-person query loop.
        live_counts = compute_live_face_counts([p.id for p in all_people], n_threads=4)

        result_list = []
        for p in all_people:
            counts = live_counts[p.id]
            p_dict = {}
            p_dict['num_faces'] = counts['num_faces']
            p_dict['url'] = f'{domain_name}/{p.id}/'
            p_dict['id'] = p.id
            p_dict['person_name'] = p.person_name
            p_dict['further_images_unlikely'] = p.further_images_unlikely

            if p.person_name == settings.BLANK_FACE_NAME:
                # poss_ident1__isnull=False excludes faces with no
                # candidate suggestion at all (2026-10-21, per the user's
                # own call) - these are mostly tiny/low-quality detections
                # the recognition pipeline never produced a guess for,
                # overwhelmingly belonging to people already declared
                # .ignore, that aren't worth surfacing for manual review.
                # This population isn't static (the backend's own pipeline
                # keeps assigning them faces over time), so it's excluded
                # here at display/count time rather than deleted or
                # permanently reclassified - scoped to this web-facing
                # count/listing only; mobile_views.py/assign_faces.py's own
                # Unassigned-population queries are untouched, since they
                # need the true full population to do their jobs.
                blanks_qs = Face.objects.filter(
                    declared_name__person_name=settings.BLANK_FACE_NAME,
                    poss_ident1__isnull=False,
                )
                num_blanks = blanks_qs.count()
                p_dict['num_possibilities'] = num_blanks
                p_dict['num_unverified_faces'] = num_blanks
                p_dict['num_faces'] = num_blanks
                # Unassigned specifically - one extra targeted query
                # (not a per-person loop, see live_counts.py's own
                # reasoning for why that matters) for the frontend's
                # "Confirm from" sidebar split (personSidebar.jsx).
                num_blanks_video = blanks_qs.filter(source_video_file__isnull=False).count()
                p_dict['num_possibilities_video'] = num_blanks_video
                p_dict['num_possibilities_image'] = num_blanks - num_blanks_video
            else:
                p_dict['num_possibilities'] = counts['num_possibilities']
                p_dict['num_possibilities_video'] = counts['num_possibilities_video']
                p_dict['num_possibilities_image'] = counts['num_possibilities'] - counts['num_possibilities_video']
                p_dict['num_unverified_faces'] = counts['num_unverified_faces']

            # Count backing the frontend's ".ignore" sidebar subordinate
            # row ("Flagged for review") - see PersonParamView's
            # `flagged` param above for what this actually counts.
            # Only meaningful for .ignore; every other person gets 0
            # rather than an extra query.
            if p.person_name == settings.SOFT_IGNORE_NAME:
                num_flagged = Face.objects.filter(
                    poss_ident1=p, mobile_review_hidden=True
                ).count()
                p_dict['num_review_flagged'] = num_flagged
                # The main ".ignore" unlabeled screen and its "Flagged
                # for review" subordinate row are a complementary
                # partition of the same poss_ident1 candidates now
                # (see PersonParamView's `flagged` param) - flagged
                # ones are excluded from the main screen's own query,
                # so its sidebar count needs the same exclusion or it
                # would overcount relative to what's actually shown
                # there.
                p_dict['num_possibilities'] = max(0, p_dict['num_possibilities'] - num_flagged)
                # Same exclusion, applied to the video split specifically -
                # num_possibilities_image is derived as the remainder
                # rather than queried separately, so the two always sum
                # back to the already-adjusted num_possibilities above
                # exactly, with no separate rounding/drift risk.
                num_flagged_video = Face.objects.filter(
                    poss_ident1=p, mobile_review_hidden=True, source_video_file__isnull=False
                ).count()
                p_dict['num_possibilities_video'] = max(0, p_dict['num_possibilities_video'] - num_flagged_video)
                p_dict['num_possibilities_image'] = max(0, p_dict['num_possibilities'] - p_dict['num_possibilities_video'])
                # Frontend's "Confirm from" media filter (personSidebar.jsx)
                # needs the "Flagged for review" row's own count split the
                # same way the main possibility count already is - reuses
                # num_flagged_video computed just above rather than a new
                # query.
                p_dict['num_review_flagged_video'] = num_flagged_video
                p_dict['num_review_flagged_image'] = num_flagged - num_flagged_video
                # Backs the verify screen's "Flagged & unverified"
                # subordinate row - faces already confirmed to .ignore,
                # not yet verified, that were ALSO flagged at some point
                # (mobile_review_hidden survives confirm - see
                # PersonParamView's face_declared/do_only_unverified
                # branch). Same complementary-partition reasoning as
                # num_possibilities' adjustment just above, but against
                # num_unverified_faces instead - the main verify screen's
                # own query excludes these, so its sidebar count needs the
                # same exclusion.
                num_flagged_unverified = Face.objects.filter(
                    declared_name=p, validated=False, mobile_review_hidden=True
                ).count()
                p_dict['num_review_flagged_unverified'] = num_flagged_unverified
                p_dict['num_unverified_faces'] = max(0, p_dict['num_unverified_faces'] - num_flagged_unverified)
            else:
                p_dict['num_review_flagged'] = 0
                p_dict['num_review_flagged_video'] = 0
                p_dict['num_review_flagged_image'] = 0
                p_dict['num_review_flagged_unverified'] = 0

            result_list.append(p_dict)

        js = {'count': num_people, 'next': None, 'previous': None, 'results': result_list,}

        return HttpResponse(json.dumps(js), content_type='application/json')
    
class FolderListView(APIView):
    permission_classes = (IsAuthenticated,) 
    def get(self, request, *args, **kwargs):
        params = self.request.query_params

        all_folders = Directory.objects.all()
        f_queue = Queue()

        for f in all_folders:
            f_queue.put(f)

        result_list = []
        domain_name = 'https://' + os.environ['API_DOMAIN'] + '/api/directories'

        def worker():
            while not f_queue.empty():
                fold = f_queue.get()
                f_dict = {}
                f_dict['id'] = fold.id
                f_dict['url'] = f'{domain_name}/{fold.id}/'
                f_dict['top_level_name'] = fold.dir_path.split('/')[-1]
                f_dict['first_datesec'] = fold.first_datesec
                f_dict['mean_datesec'] = fold.mean_datesec
                f_dict['num_images'] = fold.num_images
                f_dict['year'] = datetime.fromtimestamp(fold.first_datesec).year
                result_list.append(f_dict)
                f_queue.task_done()

        for i in range(4):
            threading.Thread(target=worker).start()
        f_queue.join()

        # /directories/?fields=id,url,top_level_name,first_datesec,mean_datesec,num_images,year&limit=2000

        js = {'count': len(all_folders), 'next': None, 'previous': None, 'results': result_list,}

        return HttpResponse(json.dumps(js), content_type='application/json')

class PersonParamView(APIView):
    permission_classes = (IsAuthenticated,) 
    def get(self, request, *args, **kwargs):

        params = self.request.query_params
        
        if 'page' in params.keys():
            try:
                page = int(params['page'])
            except:
                return render_404(request, 'Page requested is not an integer.')
        else:
            page = 1
        page = max(page, 1)

        # Real pagination (this `page` param used to be parsed and then
        # never actually used - the response always contained the entire
        # matching id list). For a queue the size of .ignore (~100k faces)
        # that meant one request returning a ~100k-element JSON array,
        # taking several seconds to query/serialize/download/iterate
        # before the frontend would show a single tile - see CLAUDE.md's
        # write-up. FACE_PAGE_SIZE-sized slices fix that; the frontend
        # (imageScreen.jsx) renders page 1 immediately and streams the
        # rest in behind it.
        page_size = settings.FACE_PAGE_SIZE
        start = (page - 1) * page_size
        end = start + page_size

        if 'only_unverified' in params.keys():
            value = params['only_unverified']
            # Invert it for the DB query.
            do_only_unverified = (value.lower() == 'true')
        else:
            do_only_unverified = False

        # face_poss branch only (below) - lets the frontend's sort-order
        # toggle (imageScreen.jsx) invert the weight_1 ranking. Default
        # 'desc' preserves the original behavior (highest-confidence
        # candidate first).
        sort_ascending = params.get('order', 'desc').lower() == 'asc'

        # face_poss branch only (below) - 'image'/'video' restrict the
        # confirm queue to faces sourced from ordinary photos vs video
        # frames respectively; anything else (including the default,
        # unset) means no restriction.
        media_filter = params.get('media', 'all').lower()

        # Frontend's ".ignore" sidebar subordinate row ("Flagged for
        # review") - faces still proposed as .ignore (poss_ident1) that
        # were reviewed once already via the mobile app's ignore-review
        # flow and marked mobile_review_hidden=True instead of being
        # confirmed as .ignore outright (see api/mobile_views.py's
        # BulkConfirmIgnore) - i.e. "looked at, might actually be
        # someone." Only meaningful on the face_poss branch below.
        do_flagged_only = params.get('flagged', '').lower() == 'true'

        # Populated only for field == 'face_declared' with
        # only_unverified=true - see there for what this is.
        cluster_groups = {}
        # Populated for both face_declared and face_poss (not directory,
        # whose id_list is ImageFile ids, not Face ids) - lets the
        # frontend gate its fast/accurate video-frame double-fetch
        # (api/views.py's face_source `fast` param) to only the faces
        # that actually need it, rather than either always double-
        # fetching (wasteful for the image-sourced majority) or never
        # doing the accurate-frame swap at all.
        video_face_ids = []
        # face_poss, page 1 only - the true total matching this exact
        # filter (media/flagged), for the frontend's sidebar count to
        # display while a non-default media filter is active (see
        # imageScreen.jsx/personSidebar.jsx). None everywhere else - a
        # real COUNT(*) query is only worth paying for once per filter
        # change, not on every background page request.
        total_matching = None

        id_key = kwargs['id']
        field = kwargs['field']

        if field not in ['face_declared', 'face_poss', 'directory']:
            return render_404(request, f"Requested type '{field}' not in ['face_declared', 'face_poss', 'directory']")

        if field in ['face_declared', 'face_poss']:
            try:
                person_obj = Person.objects.get(id=id_key)
            except Exception as e:
                return render_404(request, f"Error: Requested ID {id_key} not available in database.")

            if field == 'face_declared':

                default_query = Q(declared_name=person_obj)
                # Keep this in sync with PersonListView's own BLANK_FACE_NAME
                # count override above - same reasoning (2026-10-21): faces
                # with no candidate suggestion at all are excluded from
                # Unassigned's own gallery/count here, not from the Face
                # table itself.
                if person_obj.person_name == settings.BLANK_FACE_NAME:
                    default_query &= Q(poss_ident1__isnull=False)
                unverified_query = Q(validated=False)
                # verification_cluster_group (face_manager/
                # verification_clustering.py's nightly job) only ever gets
                # set on faces this exact query already targets when
                # do_only_unverified is true (validated=False, declared to
                # a real person) - grouping information the frontend's
                # verify-screen "Group by cluster" mode uses to review a
                # whole visually-coherent batch at once instead of one
                # face at a time. Only clusters of size >=2 get a group id
                # (see that module); a face with no group id is a
                # singleton, always still verified individually.
                if do_only_unverified:
                    faces_qs = Face.objects.filter(default_query & unverified_query)
                    # Frontend's ".ignore" sidebar subordinate row on the
                    # verify screen ("Flagged & unverified") - faces
                    # already confirmed to .ignore but not yet verified,
                    # that were ALSO flagged at some point by the mobile
                    # app's ignore-review flow (mobile_review_hidden).
                    # Nothing clears that flag on confirm (Face.
                    # associate_person et al - checked, none of them touch
                    # it), so a flagged candidate that later gets manually
                    # confirmed as .ignore still carries it. Same
                    # complementary-partition shape as the face_poss
                    # branch's own `flagged` handling below: the main
                    # verify screen and this subordinate row split
                    # .ignore's unverified faces with no overlap either
                    # way, same reasoning (a mobile-skipped candidate
                    # shouldn't just vanish from the person's own gallery -
                    # see the face_poss branch's real-person carve-out).
                    if person_obj.person_name == settings.SOFT_IGNORE_NAME:
                        if do_flagged_only:
                            faces_qs = faces_qs.filter(mobile_review_hidden=True)
                        else:
                            faces_qs = faces_qs.filter(
                                Q(mobile_review_hidden__isnull=True) | Q(mobile_review_hidden=False)
                            )
                    # .order_by('id') -- pagination needs a stable, explicit
                    # order (an unordered queryset isn't guaranteed to come
                    # back in the same order across separate LIMIT/OFFSET
                    # requests, which could silently duplicate or skip ids
                    # between pages). Cluster groups are fetched in the same
                    # sliced query as the ids themselves, rather than a
                    # second scan over the whole queryset like before - both
                    # now cost only as much as one page, not the whole set.
                    page_rows = list(
                        faces_qs.order_by('id')
                        .values_list('id', 'verification_cluster_group')[start:end]
                    )
                    cluster_groups = {
                        str(fid): group for fid, group in page_rows if group is not None
                    }
                    faces = [fid for fid, _ in page_rows]
                else:
                    faces = list(
                        Face.objects.filter(default_query).order_by('id').values_list('id', flat=True)[start:end]
                    )
            else:

                poss_query = Q(poss_ident1=person_obj)
                # mobile_review_hidden is a general-purpose "hide from the
                # mobile app's own queues" flag - set not just by the
                # .ignore-review flow this `flagged` param models, but also
                # by the mobile app's plain "Skip" action on any *real*
                # person's suggestion queue (HideFromMobile/
                # LabelingGroupsView, api/mobile_views.py). Only .ignore
                # has a frontend view that actually surfaces flagged faces
                # (the "Flagged for review" sidebar row) - filtering by it
                # for every other person made a mobile-skipped candidate
                # disappear from their gallery entirely, with nowhere to
                # find it again, while their sidebar count (a separately
                # stored field, untouched by this query) still included
                # it. Scope the filtering to .ignore specifically instead
                # of applying it to every person's face_poss query.
                if person_obj.person_name == settings.SOFT_IGNORE_NAME:
                    if do_flagged_only:
                        poss_query &= Q(mobile_review_hidden=True)
                    else:
                        # Complementary to the flagged branch above - a
                        # face already pulled aside for closer review
                        # shouldn't also keep showing up in .ignore's own
                        # main candidate list. Together these two branches
                        # partition .ignore's poss_ident1 set with no
                        # overlap either way.
                        poss_query &= (Q(mobile_review_hidden__isnull=True) | Q(mobile_review_hidden=False))

                # Frontend's confirm-queue media filter (imageScreen.jsx,
                # 2026-09-18) - lets a person confirm image-sourced faces,
                # video-sourced faces, or both, separately. source_video_file
                # is null for an ordinary photo-sourced face, set for one
                # extracted from a video frame.
                if media_filter == 'image':
                    poss_query &= Q(source_video_file__isnull=True)
                elif media_filter == 'video':
                    poss_query &= Q(source_video_file__isnull=False)

                # Was: fetch every matching row, then sort descending by
                # weight_1 in Python - fine when the whole set was always
                # returned anyway, but that meant materializing and sorting
                # the entire queryset (all of .ignore's ~100k rows) on
                # every single request regardless of how much was actually
                # shown. weight_1 is a plain column, so Postgres can do
                # this ordering directly, which is also what makes real
                # LIMIT/OFFSET slicing possible here.
                weight_order = 'weight_1' if sort_ascending else '-weight_1'
                faces = list(
                    Face.objects.filter(poss_query).order_by(weight_order).values_list('id', flat=True)[start:end]
                )
                if page == 1:
                    total_matching = Face.objects.filter(poss_query).count()

            id_list = list(faces)
            video_face_ids = list(
                Face.objects.filter(id__in=id_list, source_video_file__isnull=False)
                .values_list('id', flat=True)
            )
        else:
            try:
                dir_obj = Directory.objects.get(id=id_key)
            except Exception as e:
                return render_404(request, f"Error: Requested ID {id_key} not available in database.")


            # Newest first - this id_list is what the frontend's Folders tab
            # walks straight through for both tile order and modal prev/next
            # paging, so ordering has to happen here rather than client-side.
            image_set = ImageFile.objects.filter(directory=dir_obj).order_by('-dateTaken').values_list('id', flat=True)
            id_list = list(image_set[start:end])

        # A full page came back -> assume there's more rather than running
        # a separate COUNT(*) query to know for certain. The only cost of
        # being wrong is one extra request landing on an empty next page,
        # when the true total happens to be an exact multiple of
        # page_size - cheap and rare enough not to be worth a second query
        # on every single page request.
        has_more = len(id_list) == page_size

        js = {
            'num_results': len(id_list), 'type': field, 'id_list': id_list,
            'has_more': has_more, 'total_matching': total_matching,
            'cluster_groups': cluster_groups, 'video_face_ids': video_face_ids,
        }

        return HttpResponse(json.dumps(js), content_type='application/json')


class FaceViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,) 

    queryset = Face.objects.all()
    serializer_class = api_ser.FaceSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_serializer_context(self):
        context = super(FaceViewSet, self).get_serializer_context()
        # print("Context for face: ", context, dir(context['request']))
        context.update({"request": self.request})
        return context


    def highlight_from_face(self, face, person_name):
        # Helper method to extract a 500x500 pixel highlight
        # image from the face. Uses the source_image_file
        # of the face to get the original image, extracts the
        # highlight given the face borders, and resizes. Returns
        # the face as a ContentFile, which is used in the 
        # save method for the person.highlight_image

        # Get the borders
        r_left = face.box_left
        r_right = face.box_right
        r_bot = face.box_bottom
        r_top = face.box_top

        # Make the borders a square
        top_to_bot = r_bot - r_top
        left_to_right = r_right - r_left
        lr_cent = r_left + left_to_right // 2
        tb_cent = r_top + top_to_bot // 2

        extent = min(top_to_bot, left_to_right) // 2

        # Pad the crop out beyond the tight face box so the highlight
        # image shows some surrounding context instead of being zoomed
        # in on just the face.
        extent = int(extent * HIGHLIGHT_PADDING_FACTOR)

        r_left = lr_cent - extent
        r_right = lr_cent + extent
        r_top = tb_cent - extent
        r_bot = tb_cent + extent

        # Extract the source file, oriented.
        source_file = face.source_image_file.filename
        pixel_hash = face.source_image_file.pixel_hash
        image = common.open_img_oriented(source_file, as_numpy = False)

        # Clamp the padded box to the source image's bounds - a face
        # near an edge would otherwise crop out of range.
        img_width, img_height = image.size
        r_left = max(0, r_left)
        r_top = max(0, r_top)
        r_right = min(img_width, r_right)
        r_bot = min(img_height, r_bot)

        # Crop out the image, resize, encode in ByteIO, etc.
        img_thmb = image.crop((r_left, r_top, r_right, r_bot))
        img_thmb = np.array(img_thmb)
        img_thmb = cv2.resize(img_thmb, (500, 500))
        # Convert color space
        img_thmb = cv2.cvtColor(img_thmb, cv2.COLOR_BGR2RGB)

        is_success, person_buff = cv2.imencode(".jpg", img_thmb)

        # Save thumbnail to in-memory file as BytesIO
        person_byte_thumbnail = BytesIO(person_buff)

        person_thumb_filename = f'{pixel_hash}_{person_name}.jpg'
        
        person_byte_thumbnail.seek(0)
        person_content_file = ContentFile(person_byte_thumbnail.read())

        # Return the filename to write and the content file.
        return person_thumb_filename, person_content_file


    # Need to set detail=False to enable a URL without a <face_id> value. 
    @action(detail=False, methods=['get'])
    def random_face(self, request):
        # Accessible as <root>/api/faces/random_face/

        # This method is supporting an app to do bite-sized verification.
        # You have to be logged in with JSON web token. It gets a 
        # random unassigned face, then gets up to the top five possible 
        # identities that it can be associated with and its ID. You can
        # then use further requests to query for the face's image
        # at <api_url>/keyed_image/face_array/?id={image_id}, then interact
        # to verify one of the names or none of them. Or have a skip option
        # I guess. 

        unassigned = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
        random_face = Face.objects.filter(unassigned).order_by('?')[0]

        js = {'id': random_face.id}
        
        if random_face.poss_ident1 is not None:
            js['poss_ident1'] = random_face.poss_ident1.person_name
            js['poss_ident1_id'] = random_face.poss_ident1_id
        else:
            js['poss_ident1'] = None
            js['poss_ident1_id'] = None

        if random_face.poss_ident2 is not None:
            js['poss_ident2'] = random_face.poss_ident2.person_name
            js['poss_ident2_id'] = random_face.poss_ident2_id
        else:
            js['poss_ident2'] = None
            js['poss_ident2_id'] = None
            
        if random_face.poss_ident3 is not None:
            js['poss_ident3'] = random_face.poss_ident3.person_name
            js['poss_ident3_id'] = random_face.poss_ident3_id
        else:
            js['poss_ident3'] = None
            js['poss_ident3_id'] = None

        if random_face.poss_ident4 is not None:
            js['poss_ident4'] = random_face.poss_ident4.person_name
            js['poss_ident4_id'] = random_face.poss_ident4_id
        else:
            js['poss_ident4'] = None
            js['poss_ident4_id'] = None

        if random_face.poss_ident5 is not None:
            js['poss_ident5'] = random_face.poss_ident5.person_name
            js['poss_ident5_id'] = random_face.poss_ident5_id
        else:
            js['poss_ident5'] = None
            js['poss_ident5_id'] = None

        # print(js)


        # js = {'num_results': len(id_list), 'type': field, 'id_list': id_list,}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['patch'])
    def verify_face(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/verify_face/
        # Accept: HTML PATCH
        # Set the face's 'verified' status to True. 
        face = self.get_object()
        data = request.data

        # print('verify face', face, face.id)

        def err_404(message=""):
            msg_start = f'Invalid face update request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404
            
        face.verify_person_in_image()

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['patch'])
    def assign_face_to_person(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/assign_face_to_person/
        # Accept: HTML PATCH
        # Requires a declared_name_key parameter in the body which
        # has the pk for a desired person with which to associate
        # the face.
        # Given the face to set to a person, changes the face's 
        # declared_name to point to the person and zeros
        # out any possible identity fields and associated 
        # weights. 

        # Get the face object and the request data
        face = self.get_object()
        data = request.data

        def err_404(message=""):
            msg_start = f'Invalid face update request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'declared_name_key' not in request.data.keys():
            return err_404("declared_name_key not set in body")

        # Get the person to associate and ensure the value
        # passed is an integer
        name_key = request.data['declared_name_key']
        # print(name_key)
        try:
            name_key = int(name_key)
        except:
            return err_404("declared_name_key passed was not an integer.")

        # Given the name key, try to find the appropriate person. 
        # Return a 404 if that person pk is not in the database. 
        try:
            identity = Person.objects.get(id=name_key)
        except:
            return err_404("Desired declared name key is not a key found in the database.")
            
        # Update the face object fields accordingly
        face.associate_person(name_key)

        # Return a json block with success=true, the new key (passed as
        # an argument), and the declared human-meaningful name. 
        js = {'success': True, 'new_key': name_key, 'new_name': face.declared_name.person_name}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['put'])
    def face_to_new_person(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/face_to_new_person/
        # Accept: HTML PUT
        # Given face_id in the URL and person_name in the request
        # body, create a new person of name person_name. Extract
        # the face highlight image from the original source image
        # of face.source_image and resize it to 500 px. Save the
        # highlight image to the new person, assign the name
        # to the new person, and assign this face to the new person
        # and zero out all proposed images. 
        face = self.get_object()
        data = request.data

        def err_404(message=""):
            msg_start = f'Invalid face update request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'person_name' not in data.keys():
            return err_404("person_name not set in body")
        else:
            person_name = data['person_name']


        new_person = Person(person_name = person_name)
        
        filename, person_content_file = self.highlight_from_face(face, person_name)
        # Load a ContentFile into the thumbnail field so it gets saved
        new_person.highlight_img.save(filename, person_content_file) 

        new_person.save()

        face.associate_person(new_person.id)

        js = {'success': True, 'new_id': new_person.id, 'new_name': person_name}
        return HttpResponse(json.dumps(js), content_type='application/json')

    @action(detail=True, methods=['put'])
    def set_as_person_thumbnail(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/set_as_person_thumbnail/
        # Accept: HTML PUT
        # No body parameters
        # Given the face, sets the associated person's thumbnail to be
        # that face. 
        # print("Thumb")
        face = self.get_object()
        person = face.declared_name

        # Capture the OLD highlight file's path before overwriting it --
        # Django's FieldFile never deletes the file it's replacing on a
        # plain reassignment, so every time someone picks a new
        # representative photo for a person, the old one was silently
        # orphaned. Found 2026-10-08 while root-causing a real orphaned-
        # file backlog (323 of 1238 face_highlights files, ~26%, were
        # exactly this).
        try:
            old_highlight_path = person.highlight_img.path
        except ValueError:
            old_highlight_path = None

        # print(person.person_name)
        filename, person_content_file = self.highlight_from_face(face, person.person_name)
        person.highlight_img.save(filename, person_content_file)

        person.save()

        if old_highlight_path:
            try:
                os.remove(old_highlight_path)
            except FileNotFoundError:
                pass

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')


    @action(detail=True, methods=['patch', 'put'])
    def ignore_face(self, request, pk=None):
        # Accessible as <root>/api/faces/<face_id>/ignore_face/
        # Accept: HTML PATCH
        # Body parameter : ignore_type in ['soft', 'hard']
        # Given the face, set the assigned name to '.ignore'
        def err_404(message=""):
            msg_start = f'Invalid face ignore request. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'ignore_type' not in request.data.keys():
            return err_404("ignore_type not set in body")
        else:
            ignore_type = request.data['ignore_type']

        if ignore_type.lower() not in ['soft', 'hard']:
            return err_404("ignore_type body parameter must be either 'soft' or 'hard'.")

        face = self.get_object()

        # print('ignore face', face, face.id)
        if ignore_type == 'soft':
            ignore_person = Person.objects.filter(person_name='.ignore')
        else:
            soft_person = Person.objects.filter(person_name='.ignore')[0]
            hard_person = Person.objects.filter(person_name='.realignore')[0]
            if face.declared_name == soft_person or face.declared_name == hard_person or face.poss_ident1 == soft_person:
                pass
            else:
                #print("Not good")
                js = {'success': False}
                #return HttpResponse(json.dumps(js), content_type='application/json')
                return err_404("Person currently assigned is not .ignore or .realignore.")
            
            ignore_person = Person.objects.filter(person_name='.realignore')
            

        face.associate_person(ignore_person[0].id)

        js = {'success': True}
        return HttpResponse(json.dumps(js), content_type='application/json')



    @action(detail=False, methods=['patch', 'put'])
    def bulk_operation(self, request, pk=None):
        # Accessible as <root>/api/faces/bulk_operation/

        payload = request.data

        if 'face_id_list' not in payload.keys():
            message = 'The key "face_id_list" was not in the keys of the dictionary passed in this request.'
            return render_404(request, message)
        if 'operation' not in payload.keys():
            message = 'The key "operation" was not in the keys of the dictionary passed in this request.'
            return render_404(request, message)
        if 'current_person_id' not in payload.keys():
            message = 'The key "current_person_id" was not in the keys of the dictionary passed in this request.'
            return render_404(request, message)
        if type(payload['face_id_list']) is not list:
            message = 'The type of face_id_list was not a list.'
            return render_404(request, message)
        if type(payload['current_person_id']) is not int:
            message = 'The type of current_person_id was not an int.'
            return render_404(request, message)
        if payload['operation'] not in ['close_unassigned', 'close_ignored', 'close_assigned', 'confirm_proposed', 'verify_face', 'flag_for_review']:
            message = f"Value in 'operation' key was not one of ['close_unassigned', 'close_ignored', 'close_assigned', 'confirm_proposed', 'verify_face', 'flag_for_review'], and is not implemented."
            return render_404(request, message)

        current_person_id = payload['current_person_id']
        
        if not Person.objects.filter(id=current_person_id).exists():
            message = f"The 'current_person_id' value does not currently correspond to the ID of a person in the database."
            response = render(request, '404.html', context = {'error': message})
            response.status_code = 400
            return response
        
        _ensure_background_worker_started()
        background_queue.put(payload)

        js = {'job_submitted': True}
        return HttpResponse(json.dumps(js), content_type='application/json')

def _set_image_cache_headers(response):
    # Every image KeyedImageView serves is immutable for the lifetime of the
    # URL that names it: face_thumbnail/source/full_* files are written once
    # at ingestion time and never reassigned on an existing row (confirmed -
    # nothing outside face_extract_encode.py's initial `new_face.face_thumbnail
    # .save(...)` ever sets that field again), and person.highlight_img is
    # the one field that *can* change, but the frontend already cache-busts
    # that specific case itself via a `&v=<highlightVersion>` query param
    # (see imageScreen.jsx) rather than reusing the same URL - so a plain
    # long-lived cache here is safe for all of these. RANDOM_ACCESS_KEY is a
    # stable server-wide constant too, not per-session, so the URL doesn't
    # change between requests for the same image either. Previously this
    # view set no cache headers at all, so browsers re-fetched (and this
    # view re-decoded/re-resized/re-encoded) the same image on every single
    # page load and re-scroll.
    response['Cache-Control'] = 'public, max-age=604800, immutable'
    return response


def _extract_video_face_frame(face):
    """On-demand full (uncropped) frame extraction for a video-sourced
    Face -- the video equivalent of just opening an image-sourced face's
    source_image_file.filename directly. Deliberately not precomputed/
    stored (would cost real disk space across the whole video library,
    for something reviewed only occasionally) -- re-extracted at request
    time instead, from the exact timestamp the face's current thumbnail/
    box/kps came from (video_thumbnail_frame_seconds). Returns a PIL
    Image, or None if there's no video source, no stored timestamp, or
    extraction fails for any reason (missing file, corrupt video).

    Uses PyAV, not ffmpeg's -ss flag. Real benchmarking (2026-09-09, see
    CLAUDE.md) found -ss-based seeking has a genuine ~30% mismatch rate
    against the frame a face's stored thumbnail actually came from, even
    given the exact correct target timestamp -- ffmpeg's own post-seek
    frame selection doesn't always agree with true sequential decode
    order for some files in this library. video_face_pipeline.py's
    extract_frame_near_timestamp() fixes this by decoding a small window
    around the target and picking whichever frame's own box-crop best
    matches this face's already-stored thumbnail, rather than trusting
    any single seek/index computation to land on the right frame."""
    video = face.source_video_file
    if video is None or face.video_thumbnail_frame_seconds is None:
        return None
    # field_order is cached on VideoFile at ingestion time (from data
    # ffprobe already returns there, no extra call needed) -- falls back
    # to a live ffprobe call only for rows ingested before that field
    # existed (see backfill_video_field_order).
    from video_face_pipeline import ffprobe_info, extract_frame_near_timestamp
    field_order = video.field_order
    try:
        # rotation comes from THIS ffprobe call's own side_data_list
        # parse, not VideoFile.rotation -- that field is populated from
        # exiftool's Rotation tag (filepopulator/video_scripts.py), a
        # different tool with a different sign convention than ffprobe's
        # side_data rotation, which is what extract_frame_near_timestamp
        # (and ffmpeg's own raw-pipe auto-rotation elsewhere in this
        # pipeline) assumes. Confirmed via real production data
        # (2026-09-09): the two disagree in sign for real files (e.g.
        # cached 270 vs. this call's 90), silently rotating the wrong
        # direction if the cached field were used here instead.
        width, height, avg_fps, probed_field_order, rotation, color_transfer, sar_scale_width = (
            ffprobe_info(video.filename)
        )
    except Exception:
        return None
    if field_order is None:
        field_order = probed_field_order
    box = (face.box_left, face.box_top, face.box_right, face.box_bottom)
    try:
        face.face_thumbnail.open('rb')
        thumb_bytes = face.face_thumbnail.read()
    except Exception:
        return None
    finally:
        try:
            face.face_thumbnail.close()
        except Exception:
            pass
    reference_bgr = cv2.imdecode(np.frombuffer(thumb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    try:
        arr = extract_frame_near_timestamp(
            video.filename, face.video_thumbnail_frame_seconds, avg_fps,
            field_order, rotation, box, reference_bgr,
            width=(sar_scale_width or width), height=height, color_transfer=color_transfer,
            sar_scale_width=sar_scale_width,
        )
    except Exception:
        return None
    if arr is None:
        return None
    try:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    except Exception:
        return None


def _extract_video_face_frame_fast(face):
    """Approximate, sub-second variant of _extract_video_face_frame --
    the original -ss-based implementation, kept as the `fast=true`
    branch of face_source's video path. Frontend contract: fetch this
    one first to show something immediately, then fetch the plain
    (accurate) endpoint and swap it in once that resolves -- same
    "placeholder, then real image" pattern as a blurred-thumbnail swap.
    Usually correct, but real benchmarking (2026-09-09, see CLAUDE.md)
    found ffmpeg's own -ss frame selection has a genuine ~30% mismatch
    rate against the frame a face's stored thumbnail actually came from
    -- that's the accurate endpoint's whole reason to exist. Returns a
    PIL Image, or None on any failure."""
    video = face.source_video_file
    if video is None or face.video_thumbnail_frame_seconds is None:
        return None
    field_order = video.field_order
    color_transfer = video.color_transfer
    sar_scale_width = video.sar_scale_width
    if field_order is None or color_transfer is None or sar_scale_width is None:
        from video_face_pipeline import ffprobe_info
        try:
            _, _, _, probed_field_order, _rotation, probed_color_transfer, probed_sar_scale_width = (
                ffprobe_info(video.filename)
            )
        except Exception:
            return None
        if field_order is None:
            field_order = probed_field_order
        if color_transfer is None:
            color_transfer = probed_color_transfer
        if sar_scale_width is None:
            sar_scale_width = probed_sar_scale_width or 0
    from video_face_pipeline import build_vf_filter
    cmd = ['ffmpeg', '-v', 'error',
           '-ss', str(face.video_thumbnail_frame_seconds), '-i', video.filename]
    vf_filter = build_vf_filter(field_order, color_transfer, sar_scale_width=sar_scale_width or None)
    if vf_filter:
        cmd += ['-vf', vf_filter]
    cmd += ['-frames:v', '1', '-f', 'image2', '-q:v', '2', 'pipe:1']
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        return Image.open(BytesIO(result.stdout)).convert('RGB')
    except Exception:
        return None


class KeyedImageView(APIView):

    permission_classes = [AllowAny]

    def get_throttles(self):
        # type='slideshow' is sig-gated (a per-photo, time-boxed
        # token -- the real kiosk's own normal continuous traffic, see
        # SlideshowMediaThrottle's own docstring); every other type
        # here is gated by the shared, unscoped RANDOM_ACCESS_KEY when
        # unauthenticated, a lower-trust path that gets the tighter
        # AccessKeyMediaThrottle instead. self.kwargs is already
        # populated by dispatch() before get_throttles() is called
        # (DRF's initial() -> check_throttles() runs before get()
        # itself). Both throttle classes skip entirely for an
        # authenticated session regardless of type.
        if self.kwargs.get('type') == 'slideshow':
            return [SlideshowMediaThrottle()]
        return [AccessKeyMediaThrottle()]

    def get(self, request, *args, **kwargs):
        '''
        Call to get an image defined by a given key.
        Method: GET
        URL: /api/keyed_image/<key_value>/?id=<int>&access_key=<key>
        Valid types for key_value:
        face_highlight: key type - person. Gets the highlight image of that person
        face_array: key type - face. Gets the cropped image of just that face
        face_source: key type: face. Gets the full image containing that face.
          For a video-sourced face, also accepts &fast=true for an
          approximate, sub-second frame (see _extract_video_face_frame_fast) --
          intended as a placeholder to show immediately while a second,
          plain (accurate, ~1-2s) request is made and swapped in. The
          `video_face_ids` list returned by paginate_obj_ids tells the
          caller which face ids are video-sourced and need this two-step
          fetch at all; every other face_source request is unaffected by
          `fast` (image-sourced faces have no video path to begin with).
        slideshow: key type: image. Gets a reduced resolution image for the slideshow.
        full_small: key type: image. Gets the pre-generated small thumbnail
          (used by the Folders tab's grid view). full_big/full_medium were
          removed 2026-10-08 -- zero real consumers across every frontend/
          mobile client; every "show me the full photo" path goes through
          slideshow (live-resized from the original) instead. See
          CLAUDE.md.
        '''

        params = self.request.query_params
        valid_types = ['face_highlight', 'face_array', 'face_source', 'slideshow', 'full_small']
        
        def err_404(message=""):
            msg_start = f'Invalid request. Url is of format (/keyed_image/<type>/?id=<int>&access_key=<string>). Valid types include {valid_types}. \n\n'
            msg_start += message
            err_404 = render_404(request, msg_start)
            return err_404

        if 'type' not in kwargs.keys():
            return err_404('You did not provide a type variable.')

        image_type = kwargs['type']
        if not image_type in valid_types:
            return err_404('You did not provide a valid type.')

        if 'id' not in params.keys():
            return err_404('ID key not passed as parameter.')
        else:
            id_key = params['id']

        if not request.user.is_authenticated:
            if image_type == 'slideshow':
                # Per-photo, time-boxed token instead of the shared global
                # access_key - see the "Slideshow media tokens" comment
                # above filteredImagesView for why. Minted there, not
                # guessable/reusable for any id other than this one.
                if not _slideshow_sig_valid('image', id_key, params.get('sig')):
                    return err_404('Invalid or expired sig.')
            elif 'access_key' not in params.keys():
                print("no access key")
                print(request.user.is_authenticated)
                return err_404('Access key not provided.')
            else:
                if params['access_key'] != settings.RANDOM_ACCESS_KEY:
                    # print(params['access_key'], settings.RANDOM_ACCESS_KEY)
                    print(settings.RANDOM_ACCESS_KEY)
                    return err_404('Invalid access key.')

        def getAndCheckID(obj_type, id_key):
            # print(obj_type)
            if obj_type == 'person':
                obj = Person
            elif obj_type == 'face':
                obj = Face
            elif obj_type == 'image':
                obj = ImageFile
            # print(obj)
            try:
                return obj.objects.get(id=id_key)
            except:
                return None# err_404(f"Invalid object id of type {image_type}")

        try:
            if image_type == 'face_highlight':
                person = getAndCheckID('person', id_key)
                image = person.highlight_img
                height = 700
                width = 1920 / 1080 * height
            elif image_type == 'face_array':
                face = getAndCheckID('face', id_key)
                # face_thumbnail is a pre-generated, pre-cropped square
                # JPEG (created once at face-extraction time - see
                # face_extract_encode.py's get_square_face_img/
                # cv2.imencode(".jpg", ...)), not the full source photo.
                # Decoded, CLAHE-equalized (common.clahe_equalize_bgr --
                # brightens/evens out dark or unevenly-lit faces, a real
                # user complaint), and re-encoded on every request rather
                # than precomputed/cached at ingestion time: these are
                # tiny crops, so the decode+equalize+encode cost is
                # negligible even on this view's hottest path (one
                # request per gallery tile, every tile, every load), and
                # _set_image_cache_headers' long-lived immutable cache
                # means a browser only pays it once per face anyway.
                face.face_thumbnail.open('rb')
                try:
                    raw_bytes = face.face_thumbnail.read()
                finally:
                    face.face_thumbnail.close()

                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    equalized = common.clahe_equalize_bgr(img)
                    success, encoded = cv2.imencode('.jpg', equalized)
                    if success:
                        response = HttpResponse(encoded.tobytes(), content_type='image/jpeg')
                        return _set_image_cache_headers(response)

                # Decode/equalize/encode failed for some reason (e.g. a
                # corrupted thumbnail file) -- fall back to the original
                # bytes rather than a broken image.
                response = HttpResponse(raw_bytes, content_type='image/jpeg')
                return _set_image_cache_headers(response)
            elif image_type == 'face_source':
                face = getAndCheckID('face', id_key)
                height = 700
                width = 1920 / 1080 * height
                if face.source_video_file_id is not None:
                    # No stored full-context image exists for a video-
                    # sourced face (unlike an image-sourced one, where the
                    # original photo file already IS the full-context
                    # image) -- extracted on demand below instead, from
                    # the same timestamp the current thumbnail came from.
                    image = None
                else:
                    source = face.source_image_file
                    image = source.filename
            elif image_type == 'slideshow':
                img_obj = getAndCheckID('image', id_key)
                image = img_obj.filename
                if 'height' in params.keys():
                    height = int(params['height'])
                else:
                    height = settings.DEFAULT_RESOLUTION_HEIGHT

                width = 1920 / 1080 * height
            elif image_type == 'full_small':
                img_obj = getAndCheckID('image', id_key)
                image = img_obj.thumbnail_small
                    
        except:
            return err_404(f'Bad id for object of type {image_type}')

        # Frontend's full-size modal wants the photo's capture date
        # alongside the image itself - reuses this view's own id/type
        # resolution and access-key/auth check above rather than a
        # separate endpoint, and returns before the (comparatively
        # expensive) decode/resize/redraw below since nothing past this
        # point is needed for a date lookup. Also returns the full file
        # path (same `filename` field the image decode below actually
        # opens) for the frontend's filename label - bundled into this
        # same param/response rather than a separate one, since the
        # frontend always wants both together and the source object is
        # already loaded here either way.
        if params.get('date', '').lower() == 'true':
            if image_type in ('face_array', 'face_source'):
                src = face.source_image_file or face.source_video_file
                date_taken = src.dateTaken if src is not None else None
                filename = src.filename if src is not None else None
            elif image_type in ('slideshow', 'full_small'):
                date_taken = img_obj.dateTaken
                filename = img_obj.filename
            else:
                date_taken = None
                filename = None
            js = {
                'date_taken': date_taken.isoformat() if date_taken else None,
                'filename': filename,
            }
            return HttpResponse(json.dumps(js), content_type='application/json')

        if image_type == 'face_source' and face.source_video_file_id is not None:
            # fast=true: approximate, sub-second placeholder (frontend
            # contract: show this immediately, then fetch the plain/
            # accurate response and swap it in). See
            # _extract_video_face_frame_fast's docstring.
            if params.get('fast', '').lower() == 'true':
                image = _extract_video_face_frame_fast(face)
            else:
                image = _extract_video_face_frame(face)
            if image is None:
                return err_404('Could not extract a context frame for this video-sourced face.')
        else:
            # print("Type of image is ", type(image), image_type, dir(image))
            # if type(image) ==
            image = common.open_img_oriented(image, as_numpy = False)

        # Resize image. Allow for upsampling now. 
        w, h = image.size[:2]

        try:
            width = int(w / h * height)

            try:
                # image.thumbnail( (width, height), Image.ANTIALIAS)
                if width > w : # Upsampling
                    image = image.resize( (width, height) )
                else:
                    image = image.resize( (width, height), resample=Image.Resampling.BICUBIC)
            except:
                pass

        except:
            pass

        # Unverified-faces modal ("R" send-to-other-person aside) wants a
        # box drawn around the specific face being reviewed, not just the
        # whole source photo - only meaningful for face_source (the one
        # type keyed by a Face id in the first place; slideshow/full_*
        # are keyed by ImageFile and have no single face in mind).
        # face.box_left/top/right/bottom are stored in the *original*
        # (pre-resize) image's pixel space, so they need scaling by
        # however much the image actually ended up resized - computed
        # from the real before/after sizes rather than trusting the
        # `height`/`width` locals above, since the resize try/except just
        # above can silently leave `image` at its original size on
        # failure (in which case this correctly computes a scale of 1).
        if image_type == 'face_source' and params.get('highlight_box', '').lower() == 'true':
            if image.mode != 'RGB':
                image = image.convert('RGB')
            scale_x = image.width / w
            scale_y = image.height / h
            box_line_width = 4
            # Drawn flush against the actual detected box, the line itself
            # was covering the edges of the face rather than framing it -
            # push it outward by (at least) 3x its own width in each
            # direction so the box frames the face with room to spare,
            # clamped so it can't run off the image edge.
            box_margin = box_line_width * 3
            box = [
                max(0, face.box_left * scale_x - box_margin),
                max(0, face.box_top * scale_y - box_margin),
                min(image.width, face.box_right * scale_x + box_margin),
                min(image.height, face.box_bottom * scale_y + box_margin),
            ]
            ImageDraw.Draw(image).rectangle(box, outline=(255, 0, 0), width=box_line_width)

        FTYPE = 'JPEG'
        temp_thumb = BytesIO()

        # print(time.time()-s)
        image.save(temp_thumb, FTYPE)
        # print("Here")
        temp_thumb.seek(0)

        return _set_image_cache_headers(HttpResponse(temp_thumb.read(), content_type="image/jpeg"))


# --- Slideshow media tokens -------------------------------------------------
#
# The 'slideshow' KeyedImageView type and KeyedVideoView used to accept any
# request carrying settings.RANDOM_ACCESS_KEY - a single secret shared by
# every unauthenticated caller, valid for the lifetime of the gunicorn
# process, and not scoped to any particular photo/video. That meant anyone
# who read the key out of net_slideshow's public JS bundle could fetch ANY
# id just by incrementing it, not only the ones actually shown to them -
# turning a slideshow into a full-library scraping endpoint.
#
# These tokens replace that for those two endpoints only (face_*/full_small
# KeyedImageView types, used by the mobile app and facewires_frontend, are
# untouched and keep using RANDOM_ACCESS_KEY). A token is derived from the
# specific (kind, id) it's for plus a coarse time bucket, using Django's own
# SECRET_KEY - so it's only ever valid for that one piece of media, and only
# for a limited window, without needing any server-side session/token
# storage. filteredImagesView mints one per id it actually hands back in a
# page; KeyedImageView/KeyedVideoView just recompute and compare.
SLIDESHOW_TOKEN_BUCKET_SECONDS = 20 * 60  # each token is good for ~20-40 min
SLIDESHOW_PAGE_SIZE_DEFAULT = 60
SLIDESHOW_PAGE_SIZE_MAX = 120


def _slideshow_token_buckets():
    now = int(time.time())
    current = now // SLIDESHOW_TOKEN_BUCKET_SECONDS
    # Also accept the previous bucket, so a token minted right before a
    # bucket boundary doesn't go stale mid-slideshow.
    return (current, current - 1)


def _slideshow_sig(kind, media_id, bucket):
    msg = f'{kind}:{media_id}:{bucket}'.encode()
    return hmac.new(settings.SECRET_KEY.encode(), msg, hashlib.sha256).hexdigest()[:16]


def _slideshow_current_sig(kind, media_id):
    return _slideshow_sig(kind, media_id, _slideshow_token_buckets()[0])


def _slideshow_sig_valid(kind, media_id, provided):
    if not provided:
        return False
    return any(
        hmac.compare_digest(_slideshow_sig(kind, media_id, bucket), provided)
        for bucket in _slideshow_token_buckets()
    )


class filteredImagesView(APIView):

    authentication_classes = [] # Optional: keeps session auth available
    permission_classes = [HasSlideshowKeyOrAuthenticated]
    # permission_classes = (HasSlideshowKeyOrAuthenticated,)
    throttle_classes = [SlideshowListThrottle]

    def get(self, request, *args, **kwargs):
        # print(self.request.query_params)
        params = self.request.query_params
        print(params)

        # video_query mirrors p_query but built against VideoFile's own
        # reverse 'face' relation (source_video_file, not
        # source_image_file) -- same person/year filters, since
        # Face.declared_name is orthogonal to image-vs-video (see
        # CLAUDE.md's Phase 3 design note).
        if len(params.keys()) == 0 : # or 'people' not in params.keys():
            # Return all objects!
            obj = ImageFile.objects.all().order_by('dateTaken').values_list('id', 'dateTaken')
            video_obj = VideoFile.objects.all().order_by('dateTaken').values_list('id', 'dateTaken')
            # return render_404(request, "No parameters were passed. Must pass a combination of 'people', 'year_start', 'year_end'...")
        else:
            p_query = None
            years_query = None
            video_years_query = None
            if 'people' in params.keys():
                people = params['people'].split(',')
                if len(people) > 0:
                    people_query = Q(face__declared_name__person_name=people[0])
                for p in people[1:]:
                    people_query = people_query | Q(face__declared_name__person_name=p)

                # people_query = (people_query) & ~Q(face__declared_name__person_name='Charlotte Lewis')
                if p_query is None:
                    p_query = people_query

            if 'year_start' in params.keys():
                year_start = int(params['year_start'])
                year_start_query = Q(face__source_image_file__dateTaken__year__gte = year_start)
                video_year_start_query = Q(face__source_video_file__dateTaken__year__gte = year_start)
                if years_query is None:
                    years_query = year_start_query
                    video_years_query = video_year_start_query
                else:
                    years_query = years_query & year_start_query
                    video_years_query = video_years_query & video_year_start_query

            if 'year_end' in params.keys():
                year_end = int(params['year_end'])
                year_end_query = Q(face__source_image_file__dateTaken__year__lte = year_end)
                video_year_end_query = Q(face__source_video_file__dateTaken__year__lte = year_end)
                if years_query is None:
                    years_query = year_end_query
                    video_years_query = video_year_end_query
                else:
                    years_query = years_query & year_end_query
                    video_years_query = video_years_query & video_year_end_query


            if p_query is None:
                p_query = years_query
                video_query = video_years_query
            else:
                if years_query is not None:
                    p_query = p_query | years_query
                # 'people' filters against Face.declared_name, which is
                # source-agnostic -- the same people_query works
                # unchanged against VideoFile's own reverse 'face'
                # relation, no video-specific rebuild needed for that
                # half.
                video_query = people_query if 'people' in params.keys() else None
                if video_years_query is not None:
                    video_query = video_query | video_years_query if video_query is not None else video_years_query

            print(p_query)
            if p_query is None:
                # Query params were present but none were 'people'/
                # 'year_start'/'year_end' (e.g. just '?key=...') -- p_query
                # never got built. Fall back to "all images", same as the
                # no-params case above, rather than calling
                # ImageFile.objects.filter(None), which raises TypeError.
                obj = ImageFile.objects.all().order_by('dateTaken').values_list('id', 'dateTaken')
                video_obj = VideoFile.objects.all().order_by('dateTaken').values_list('id', 'dateTaken')
            else:
                obj = ImageFile.objects.filter(p_query).exclude(face__declared_name__person_name='Charlotte Lewis').distinct().order_by('dateTaken').values_list('id', 'dateTaken')
                if video_query is None:
                    video_obj = VideoFile.objects.none()
                else:
                    video_obj = VideoFile.objects.filter(video_query).exclude(face__declared_name__person_name='Charlotte Lewis').distinct().order_by('dateTaken').values_list('id', 'dateTaken')

        # include_images/include_videos: dropping a whole kind up front
        # (rather than after full_data fetch) keeps the excluded kind from
        # ever touching ordering, pagination, or total_count.
        image_rows = list(obj) if params.get('include_images') != 'false' else []
        video_candidate_rows = list(video_obj) if params.get('include_videos') != 'false' else []

        # A video with no playable URL yet (still mid-transcode) used to be
        # filtered out client-side, AFTER the full, undisclosed id list had
        # already been handed over. Filtering it out here instead means it
        # never occupies a page/offset slot or counts toward total_count at
        # all - this used to run only when full_data was requested, but it
        # has to run unconditionally now since it affects which ids even
        # exist in the paginated sequence, not just what metadata comes back
        # for them. See VideoFile.get_playback_url's own docstring for what
        # "ready" means (transcoded, or a native codec with playback_path set).
        video_rows = []
        if video_candidate_rows:
            ready_ids = {
                v.id for v in VideoFile.objects.filter(
                    id__in=[row[0] for row in video_candidate_rows]
                )
                if v.get_playback_url() is not None
            }
            video_rows = [row for row in video_candidate_rows if row[0] in ready_ids]

        entries = [{'id': i, 'kind': 'image', 'date': d} for i, d in image_rows]
        entries += [{'id': i, 'kind': 'video', 'date': d} for i, d in video_rows]

        cronological = 'cronological' in params.keys()
        seed = None
        if cronological:
            entries.sort(key=lambda e: (e['date'] is None, e['date']))
        else:
            # A stable order across pages (so later pages don't repeat or
            # skip entries from earlier ones) without keeping any
            # server-side session state: the client generates a seed once
            # and resends it on every subsequent page request for this same
            # slideshow run. random.Random(seed) below is a separate,
            # private PRNG instance - this never touches (or is affected
            # by) any other random.* call elsewhere in the process.
            #
            # The seed STRING itself, though, must NOT be drawn from the
            # global `random` module - picasa/settings.py deliberately
            # reseeds that to today.year*10000+month*100+day at import
            # time (for unrelated reproducibility elsewhere), which is
            # constant for the whole day and gets re-run identically in
            # every gunicorn worker (no --preload, so each worker imports
            # settings.py independently at its own startup). That made
            # every worker's global PRNG start in lockstep, so two workers
            # that happened to reach this line after the same number of
            # prior random.* calls would mint the EXACT SAME "random"
            # seed - confirmed live (identical seeds on genuinely separate
            # requests seconds apart). secrets.choice draws from the OS's
            # CSPRNG, completely unaffected by that global reseed.
            seed = params.get('seed') or ''.join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(16)
            )
            random.Random(seed).shuffle(entries)

        total_count = len(entries)

        try:
            offset = max(0, int(params.get('offset', 0)))
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = int(params.get('limit', SLIDESHOW_PAGE_SIZE_DEFAULT))
        except (TypeError, ValueError):
            limit = SLIDESHOW_PAGE_SIZE_DEFAULT
        # Hard-capped regardless of what's requested - this is what actually
        # stops a single call from ever disclosing the whole library, not
        # just a default a well-behaved client happens to respect.
        limit = max(1, min(limit, SLIDESHOW_PAGE_SIZE_MAX))

        page = entries[offset:offset + limit]

        js = {
            'items': [{'id': e['id'], 'kind': e['kind']} for e in page],
            'total_count': total_count,
            'offset': offset,
            'limit': limit,
        }
        if seed is not None:
            js['seed'] = seed

        if 'full_data' in params.keys():
            page_image_ids = [e['id'] for e in page if e['kind'] == 'image']
            page_video_ids = [e['id'] for e in page if e['kind'] == 'video']

            # Slideshow frontend's date/location overlay, and (new) each
            # id's signed media token - a second, separate query (rather
            # than folding this into the query above) so the common
            # no-full_data path never pays for the geocode join. Only
            # covers THIS page's ids now, not the whole filtered set.
            # Keyed by id (not a list) for O(1) frontend lookups per image.
            meta_rows = ImageFile.objects.filter(id__in=page_image_ids).values(
                'id', 'dateTaken', 'geocode__nearest_metro_name', 'geocode__locality'
            )
            full_data = {}
            for row in meta_rows:
                # nearest_metro_name is the human-reviewed, recognizable
                # pick (e.g. "Seattle") - preferred over the raw precise
                # locality (e.g. "Bothell"), which is only a fallback for
                # coordinates the metro backfill hasn't covered yet.
                # Omit location entirely (rather than sending an empty
                # string) when there's no geocode row for this image at
                # all.
                location = row['geocode__nearest_metro_name'] or row['geocode__locality'] or None
                full_data[row['id']] = {
                    'date': row['dateTaken'].isoformat() if row['dateTaken'] else None,
                    'location': location,
                    'sig': _slideshow_current_sig('image', row['id']),
                }
            js['full_data'] = full_data

            # video_data mirrors full_data, plus the fields a frontend
            # needs to actually play a video (image entries never need
            # these -- the plain 'slideshow' keyed_image type already
            # handles resizing/serving for images). Deliberately NO
            # playback_url here (unlike the Phase 4 design this
            # originally shipped with) -- that would be Apache's direct,
            # Basic-Auth-protected media URL, and this endpoint is
            # reachable with just the low-trust slideshow key, not a
            # real login. Shipping Apache's real media credentials
            # (api/parameters' img_server_user/img_server_pw) to a
            # client that only proves it holds the slideshow key would
            # leak them to anyone who can read that key out of the
            # frontend bundle. Instead the frontend builds a
            # /api/keyed_video/<id>/?sig=... URL itself, exactly like it
            # already does for images - KeyedVideoView below streams the
            # bytes straight from this container's own filesystem mounts,
            # so the real Apache password never needs to leave the
            # backend at all.
            # select_related('geocode'), not .values() -- get_playback_url()
            # needs a real model instance (see VideoFile.get_playback_url's
            # own comment for why this is the single source of truth
            # shared with VideoFileSerializer, rather than re-deriving the
            # transcoded-vs-original-vs-not-ready logic a second time here).
            video_rows_full = VideoFile.objects.filter(id__in=page_video_ids).select_related('geocode')
            video_data = {}
            for video in video_rows_full:
                location = None
                if video.geocode is not None:
                    location = video.geocode.nearest_metro_name or video.geocode.locality or None
                video_data[video.id] = {
                    'date': video.dateTaken.isoformat() if video.dateTaken else None,
                    'location': location,
                    'duration_seconds': video.duration_seconds,
                    'width': video.width,
                    'height': video.height,
                    # Already filtered to playable-only ids above, so every
                    # video that reaches this point is ready by definition.
                    'ready': True,
                    'sig': _slideshow_current_sig('video', video.id),
                }
            js['video_data'] = video_data
        response = HttpResponse(json.dumps(js), content_type='application/json')
        # Carries a fresh random shuffle (or the next page of one) on
        # every call with no explicit seed - nothing here is safe to
        # reuse from a cache. Without this, a plain F5/reload (not a
        # hard refresh) can get served a stale cached response by the
        # browser's own heuristics instead of hitting the network at
        # all, since this response previously had no cache directives
        # of any kind for a browser to go on - landing you on the exact
        # same shuffled order every reload instead of a fresh one.
        response['Cache-Control'] = 'no-store'
        return response


_RANGE_HEADER_RE = re.compile(r'bytes=(\d*)-(\d*)$')


class _LimitedReader:
    """Wraps an already-seeked, open file object so FileResponse's
    internal `iter(lambda: filelike.read(block_size), b"")` loop stops
    after exactly `limit` bytes - i.e. serves exactly one HTTP Range.
    Deliberately exposes neither seek/tell/seekable/name: FileResponse.
    set_headers() only tries to auto-compute Content-Length/Content-Type
    from those, and _ranged_file_response() below sets both explicitly
    instead (auto-computing from this wrapper would see the
    UNDERLYING file's full size/position, not the slice)."""

    def __init__(self, fileobj, limit):
        self._fileobj = fileobj
        self._remaining = limit

    def read(self, size=-1):
        if self._remaining <= 0:
            return b''
        if size < 0 or size > self._remaining:
            size = self._remaining
        chunk = self._fileobj.read(size)
        self._remaining -= len(chunk)
        return chunk

    def close(self):
        self._fileobj.close()


def _ranged_file_response(file_path, content_type, range_header):
    """Django has no built-in Range support for hand-written views -
    neither FileResponse nor even django.views.static.serve() implement
    it (confirmed by reading both; this surprised whoever wrote this
    view first, so it's documented here). Without this, every video
    request downloads the ENTIRE file regardless of what the <video>
    element actually asked for, defeating both seeking/scrubbing and
    Slideshow.jsx's next-video preload (which exists specifically to
    start streaming ahead of time without downloading clips that may
    never even be reached).

    Only a single `bytes=start-end` range is handled (what every real
    browser/media pipeline actually sends for progressive video - multi-
    range requests are vanishingly rare for this use case) - anything
    else (no header, multi-range, malformed) falls back to a plain full
    200 response rather than erroring, since a confused client will
    just re-request without Range anyway."""
    file_size = os.path.getsize(file_path)
    match = _RANGE_HEADER_RE.match((range_header or '').strip())
    if not match:
        response = FileResponse(open(file_path, 'rb'), content_type=content_type)
        response.headers['Accept-Ranges'] = 'bytes'
        return response

    start_str, end_str = match.groups()
    if start_str:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
    else:
        # Suffix form, "bytes=-500" = "the last 500 bytes".
        start = max(file_size - int(end_str), 0)
        end = file_size - 1
    end = min(end, file_size - 1)

    if start > end or start >= file_size:
        response = HttpResponse(status=416)
        response.headers['Content-Range'] = f'bytes */{file_size}'
        return response

    length = end - start + 1
    f = open(file_path, 'rb')
    f.seek(start)
    response = FileResponse(_LimitedReader(f, length), content_type=content_type)
    response.status_code = 206
    response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
    response.headers['Content-Length'] = str(length)
    response.headers['Accept-Ranges'] = 'bytes'
    return response


class KeyedVideoView(APIView):
    """GET /api/keyed_video/<id>/?access_key=<key> - streams a video's
    playable bytes straight from THIS container's own filesystem mounts
    (/videos read-only, /video_transcoded read-write - both already
    mounted here for the transcode pipeline), with hand-rolled Range-
    request support (see _ranged_file_response) so seeking/scrubbing and
    the slideshow's next-video preload both actually work.

    Deliberately NOT a redirect to Apache's direct media URL the way
    filteredImagesView's old playback_url field worked - that would
    require shipping Apache's real Basic Auth credentials
    (img_server_user/img_server_pw) to the browser, and this endpoint
    (like KeyedImageView) is reachable with just the low-trust slideshow
    key. Auth is a per-video, time-boxed `sig` (see the "Slideshow media
    tokens" comment above filteredImagesView), not the old shared,
    unscoped access_key - a leaked sig only ever works for this one
    video, and only for a limited window."""
    permission_classes = [AllowAny]
    throttle_classes = [SlideshowMediaThrottle]

    def get(self, request, id, *args, **kwargs):
        if not request.user.is_authenticated:
            if not _slideshow_sig_valid('video', str(id), request.query_params.get('sig')):
                return render_404(request, 'Invalid or expired sig.')

        try:
            video = VideoFile.objects.get(pk=id)
        except VideoFile.DoesNotExist:
            return render_404(request, 'No video with that id.')

        if video.transcoded_path:
            file_path = os.path.join(settings.VIDEO_TRANSCODE_ROOT, video.transcoded_path)
            content_type = 'video/mp4'
        elif video.codec in settings.WEB_NATIVE_CODECS:
            # Mirrors VideoFile.get_playback_url()'s own transcoded-vs-
            # original-vs-not-ready logic exactly - see that method's
            # docstring for why it's the single source of truth, not
            # re-derived independently here.
            file_path = video.filename
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        else:
            return render_404(request, 'Video not ready for playback yet.')

        if not os.path.exists(file_path):
            return render_404(request, 'Video file missing on disk.')

        return _ranged_file_response(file_path, content_type, request.META.get('HTTP_RANGE'))


# class HelloView(APIView):
#     permission_classes = (IsAuthenticated,)             # <-- And here

#     def get(self, request):
#         content = {'message': 'Hello, World!'}
#         return Response(content)

class CleanLogoutView(APIView):
    # AllowAny since a user hitting logout may already have a half-dead
    # session; we still want to make a best effort at killing both the
    # Authelia and Django sessions rather than 403 them first.
    permission_classes = (AllowAny,)

    def post(self, request, *args, **kwargs):
        try:
            # The API equivalent of clicking "Logout" in the Authelia
            # portal - invalidates the Authelia session server-side.
            # Forwarding the request's cookies (rather than redirecting
            # the browser there) avoids both the CORS gap on Authelia's
            # own endpoints and the double-hop-through-Authelia redirect
            # chain the frontend used to need.
            requests.post(
                'https://auth.exploretheworld.tech/api/logout',
                cookies=request.COOKIES,
                timeout=5,
            )
        except requests.RequestException as e:
            # Don't let Authelia being unreachable block killing the
            # Django session below.
            pass

        django_logout(request)

        return HttpResponse(json.dumps({'success': True}), content_type='application/json')


class AutheliaStateView(APIView):
    # AllowAny is deliberate here: this needs to answer even when the
    # Django session is itself already invalid/absent, since the whole
    # point is cross-checking Authelia's session independently of
    # Django's. Called server-side (not from the browser) specifically
    # to avoid CORS, since Authelia's own /api/state endpoint isn't
    # CORS-enabled for cross-origin callers and /api/verify is meant to
    # be called by a trusted backend/proxy, not browser JS.
    permission_classes = (AllowAny,)

    def get(self, request, *args, **kwargs):
        try:
            resp = requests.get(
                'https://auth.exploretheworld.tech/api/verify',
                cookies=request.COOKIES,
                headers={
                    'X-Forwarded-Host': 'facewire.exploretheworld.tech',
                    'X-Forwarded-Uri': '/',
                    'X-Forwarded-Proto': 'https',
                },
                timeout=5,
            )
            authenticated = resp.status_code == 200
        except requests.RequestException as e:
            return HttpResponse(
                json.dumps({'authenticated': None, 'error': str(e)}),
                content_type='application/json',
                status=502,
            )

        return HttpResponse(json.dumps({'authenticated': authenticated}), content_type='application/json')


###############################################
### NOTE###  Don't Use ViewSet for MODELS!!!
class ParameterViewSet(viewsets.ViewSet):
### NOTE###  Don't Use ViewSet for MODELS!!!
###############################################
    serializer_class = api_ser.ParameterSerializer

    # permission_classes = (IsAuthenticated,)
    permission_classes = (HasSlideshowKeyOrAuthenticated,)

    def list(self, request):
        instance = api_ser.ParameterSerializer.Parameters()
        if not request.user.is_authenticated:
            # The slideshow key alone (unlike a real session) does NOT
            # earn you Apache's actual media credentials - that key is
            # unavoidably public (baked into net_slideshow's frontend
            # bundle, readable by anyone who loads the page), so
            # anything gated only by it has to be treated as public
            # too. random_access_key is fine to hand out here (just a
            # low-stakes nonce gating KeyedImageView/KeyedVideoView,
            # rotated per process) - img_server_user/img_server_pw are
            # real Basic Auth credentials for the whole media library
            # and must stay behind a real login.
            instance.img_server_user = None
            instance.img_server_pw = None
        serializer = api_ser.ParameterSerializer(instance=instance, many=False)
        return Response(serializer.data)

class CsrfCookieView(APIView):
    """GET /api/csrf/ -- exists purely to make Django issue a `csrftoken`
    cookie (CSRF_COOKIE_DOMAIN is '.exploretheworld.tech', so it's
    readable from every subdomain, including net_slideshow's settings
    page). Nothing else in this project ever renders a template with
    {% csrf_token %} or otherwise calls get_token(), so without this,
    a session-authenticated POST/PUT (SlideshowConfigViewSet,
    ScheduleRuleViewSet) has no cookie to read a matching X-CSRFToken
    header from and always 403s with "CSRF token missing" -- the list/
    retrieve GETs those same forms rely on aren't CSRF-protected at
    all, so this failure is invisible until the first actual save.
    Deliberately open to anyone (not just authenticated users): the
    cookie itself is harmless without a session to pair it with, and
    the frontend should be able to fetch this before Authelia login
    has even happened."""
    authentication_classes = []
    permission_classes = [AllowAny]

    @method_decorator(ensure_csrf_cookie)
    def get(self, request, *args, **kwargs):
        return Response({'detail': 'ok'})


class SlideshowConfigViewSet(viewsets.ModelViewSet):
    """CRUD for named slideshow configs (net_slideshow's settings page) -
    session/Authelia-authenticated only, unlike the slideshow-facing
    endpoints below. Building/editing these isn't something the Pi
    itself ever needs to do."""
    permission_classes = (IsAuthenticated,)
    queryset = SlideshowConfig.objects.all().order_by('name')
    serializer_class = api_ser.SlideshowConfigSerializer


class ScheduleRuleViewSet(viewsets.ModelViewSet):
    """CRUD for day/time schedule rules, each pointing at a
    SlideshowConfig. Same auth as SlideshowConfigViewSet - see its
    docstring."""
    permission_classes = (IsAuthenticated,)
    queryset = ScheduleRule.objects.all().order_by('-priority', 'start_time')
    serializer_class = api_ser.ScheduleRuleSerializer


class ActiveSlideshowConfigView(APIView):
    """GET /api/slideshow/active_config/ - what the slideshow should be
    playing right now, resolved from ScheduleRule day/time windows (see
    api/models.py's resolve_active_slideshow_config). Slideshow-key
    reachable like filteredImagesView/ParameterViewSet - the Pi polls
    this periodically with no session of its own.

    Flat response shape deliberately mirrors the URL params
    net_slideshow's App.jsx already understands, so the frontend can
    forward these straight through unchanged. config_id is included so
    the frontend can cheaply detect "did the active config change since
    I last checked" without comparing every field."""
    authentication_classes = []
    permission_classes = [HasSlideshowKeyOrAuthenticated]

    def get(self, request, *args, **kwargs):
        config, rule = resolve_active_slideshow_config()
        if config is None:
            return Response({
                'config_id': None, 'config_name': None, 'rule_id': None,
                'people': '', 'year_start': None, 'year_end': None,
                'cronological': False, 'slide_len': 6, 'show_meta': True,
                'include_videos': True, 'include_images': True, 'debug_limit': None,
            })
        return Response({
            'config_id': config.id,
            'config_name': config.name,
            'rule_id': rule.id if rule else None,
            'people': config.people,
            'year_start': config.year_start,
            'year_end': config.year_end,
            'cronological': config.cronological,
            'slide_len': config.slide_len,
            'show_meta': config.show_meta,
            'include_videos': config.include_videos,
            'include_images': config.include_images,
            'debug_limit': config.debug_limit,
        })


class StatsViewSet(viewsets.ViewSet):
### NOTE###  Don't Use ViewSet for MODELS!!!
###############################################
    serializer_class = api_ser.ServerStatsSerializer

    permission_classes = (IsAuthenticated,)

    def list(self, request):
        stat_serializer = api_ser.ServerStatsSerializer(instance=api_ser.ServerStatsSerializer.Stats(), many=False)
        return Response(stat_serializer.data)


class ApplicationErrorViewSet(viewsets.ReadOnlyModelViewSet):
    """Backend-written, frontend-dismissible error log (common.models.
    ApplicationError / common.error_log.log_error()) -- a curated,
    smaller layer on top of real server logging, for things worth
    surfacing to a human without them having to think to go check log
    files. Read-only + dismiss from the API -- entries are only ever
    created server-side, via log_error()."""
    permission_classes = (IsAuthenticated,)
    serializer_class = api_ser.ApplicationErrorSerializer

    def get_queryset(self):
        from common.models import ApplicationError
        qs = ApplicationError.objects.all()
        if self.request.query_params.get('include_dismissed', '').lower() != 'true':
            qs = qs.filter(dismissed=False)
        return qs

    @action(detail=True, methods=['post'])
    def dismiss(self, request, pk=None):
        from django.utils import timezone
        error = self.get_object()
        error.dismissed = True
        error.dismissed_at = timezone.now()
        error.save()
        # Explicit, opt-in-only (2026-10-30, per the user's own call) --
        # NEVER implied by a plain dismiss. Dismissing just means "I
        # acknowledged this alert," not "the task actually finished";
        # conflating the two would mean a habit of dismissing without
        # checking silently teaches the watchdog to stop watching, which
        # defeats the point of it. This requires the caller to
        # separately assert they've verified it (e.g. a settings-page
        # toggle, not the default dismiss action) AND only has any
        # effect when this specific error really is a heartbeat alert
        # (related_heartbeat_task_name set by check_task_heartbeats) --
        # silently a no-op on any other kind of ApplicationError.
        if request.data.get('clear_heartbeat') and error.related_heartbeat_task_name:
            from common.heartbeat import record_heartbeat_done
            record_heartbeat_done(error.related_heartbeat_task_name)
        return Response(api_ser.ApplicationErrorSerializer(error).data)


class VideoFaceExtractionViewSet(viewsets.ViewSet):
    """On-demand, frontend-triggerable retry of videos that FAILED face
    extraction (VideoFile.face_extraction_failed=True) -- per the
    user's explicit request (not a management command): a list of
    currently-failed videos the frontend can show (with a count and a
    "reprocess all" / "reprocess this one" button), backed by this
    list + create pair.

    Deliberately scoped to already-failed videos only, never the
    general backlog -- the user explicitly does NOT want a way to
    re-trigger a video that already succeeded. `create()` rejects any
    video_id that isn't currently face_extraction_failed=True, and
    with no video_ids given at all, targets every currently-failed
    video rather than the full isProcessed=False backlog.

    Always dispatches via Celery (.delay()) rather than running
    synchronously in the request -- a video can take minutes -- so
    this always returns immediately with 202 and the real work
    happens in the background (added to the normal celery queue). The
    task's own advisory lock makes a duplicate trigger (e.g. the
    frontend firing a second request while one is already running) a
    safe no-op, not a real error, so no extra locking is done here."""

    permission_classes = (IsAuthenticated,)

    def list(self, request):
        from filepopulator.models import VideoFile

        failed = VideoFile.objects.filter(face_extraction_failed=True).order_by('-id')
        rows = [
            {
                'id': v.id,
                'filename': v.filename,
                'duration_seconds': v.duration_seconds,
                'face_extraction_error': v.face_extraction_error,
                'face_extraction_timeout_count': v.face_extraction_timeout_count,
            }
            for v in failed
        ]
        return Response({'count': len(rows), 'results': rows})

    def create(self, request):
        from filepopulator.models import VideoFile
        from face_manager.tasks import process_video_faces

        video_ids = request.data.get('video_ids')
        if video_ids is not None:
            if not isinstance(video_ids, list) or not video_ids or not all(
                isinstance(i, int) for i in video_ids
            ):
                return Response(
                    {'error': 'video_ids must be a non-empty list of integers'},
                    status=400,
                )
            failed_ids = set(
                VideoFile.objects.filter(
                    id__in=video_ids, face_extraction_failed=True
                ).values_list('id', flat=True)
            )
            not_failed = [i for i in video_ids if i not in failed_ids]
            if not_failed:
                return Response(
                    {
                        'error': (
                            'Can only reprocess videos that are currently marked '
                            f'failed. Not currently failed (or unknown): {not_failed}'
                        )
                    },
                    status=400,
                )
        else:
            video_ids = list(
                VideoFile.objects.filter(face_extraction_failed=True).values_list('id', flat=True)
            )
            if not video_ids:
                return Response({'status': 'nothing to do', 'video_ids': []}, status=200)

        max_runtime_seconds = request.data.get('max_runtime_seconds')
        if max_runtime_seconds is not None and not isinstance(max_runtime_seconds, (int, float)):
            return Response({'error': 'max_runtime_seconds must be a number'}, status=400)

        process_video_faces.delay(max_runtime_seconds=max_runtime_seconds, video_ids=video_ids)
        return Response(
            {'status': 'queued', 'video_ids': video_ids, 'max_runtime_seconds': max_runtime_seconds},
            status=202,
        )


class CameraGpsBackfillConfigViewSet(viewsets.ViewSet):
    """Backend for a frontend toggle page (same shape as the existing
    nearest-metro-override tool, built outside this repo) -- lets the
    real device owner curate which cameras' missing-GPS photos may be
    estimated from OwnTracks location history (see filepopulator/
    owntracks_gps.py's own docstring for the privacy reasoning: this
    only ever reflects one person's location, so a camera that was ever
    someone else's device must never be enabled)."""

    permission_classes = (IsAuthenticated,)
    serializer_class = api_ser.CameraGpsBackfillStatusSerializer

    def list(self, request):
        from django.db.models import Count, Q

        from filepopulator.models import CameraGpsBackfillConfig, ImageFile

        rows = (
            ImageFile.objects
            .exclude(camera_model__isnull=True).exclude(camera_model__exact='')
            .values('camera_model')
            .annotate(total=Count('id'), missing_gps=Count('id', filter=Q(gps_lat_decimal=-999)))
            .order_by('-total')
        )

        # One query for every distinct (model, make) pair, grouped in
        # Python into model -> sorted list of makes seen for it -- see
        # CameraGpsBackfillStatusSerializer's own docstring for why this
        # is kept separate from the grouping above.
        makes_by_model = {}
        for pair in ImageFile.objects.exclude(camera_model__isnull=True).exclude(
            camera_model__exact=''
        ).values('camera_model', 'camera_make').distinct():
            makes_by_model.setdefault(pair['camera_model'], set()).add(pair['camera_make'] or '')

        enabled_by_model = dict(
            CameraGpsBackfillConfig.objects.values_list('camera_model', 'enabled')
        )

        data = [
            {
                'camera_model': row['camera_model'],
                'camera_make_examples': sorted(makes_by_model.get(row['camera_model'], [])),
                'total': row['total'],
                'missing_gps': row['missing_gps'],
                'enabled': enabled_by_model.get(row['camera_model'], False),
            }
            for row in rows
        ]
        serializer = api_ser.CameraGpsBackfillStatusSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def set_enabled(self, request):
        # Accessible as <root>/api/camera_gps_backfill_config/set_enabled/
        # Accept: POST, body: {"camera_model": "...", "enabled": true}
        # Upserts by camera_model string, not a numeric pk -- a candidate
        # camera has no CameraGpsBackfillConfig row at all until the
        # first time it's toggled, so a detail-view PUT (which needs an
        # existing pk to target) wouldn't work for a first-time enable.
        from filepopulator.models import CameraGpsBackfillConfig

        payload = request.data
        if 'camera_model' not in payload or 'enabled' not in payload:
            return render_404(request, 'Body must include "camera_model" and "enabled".')
        if not isinstance(payload['enabled'], bool):
            return render_404(request, 'The "enabled" value must be a boolean.')

        config, _ = CameraGpsBackfillConfig.objects.update_or_create(
            camera_model=payload['camera_model'], defaults={'enabled': payload['enabled']},
        )
        return Response({'camera_model': config.camera_model, 'enabled': config.enabled})


class WindowedClassificationConfigViewSet(viewsets.ViewSet):
    """Backend for a frontend toggle page letting a human enable Person.
    use_windowed_classification for a specific gallery -- the per-person
    override built 2026-10-25 after a real false-positive investigation
    (Leslie Williams) found that restricting sim_99th to gallery members
    near the QUERY's own date substantially improves false-positive rate
    for people whose gallery spans a wide date range with genuine
    cross-era appearance drift, but does NOT generalize automatically
    (validated to make things flat-to-worse for most large galleries) --
    see Person.use_windowed_classification's own docstring. This is
    deliberately a manually-curated, per-person toggle: turn it on only
    for a gallery you've directly confirmed benefits (the same empirical
    check described there), not as a blanket setting."""

    permission_classes = (IsAuthenticated,)
    serializer_class = api_ser.WindowedClassificationStatusSerializer

    # Galleries below this many validated faces aren't listed by default
    # (too little data for the toggle to plausibly matter) -- but an
    # already-enabled person is always listed regardless of size, so
    # toggling one off is never blocked by them later dropping below it.
    MIN_GALLERY_SIZE = 50

    def list(self, request):
        from django.db.models import Count, Q

        from face_manager.models import Person

        people = (
            Person.objects.annotate(
                confirmed_count=Count('face_declared', filter=Q(face_declared__validated=True))
            )
            .filter(Q(confirmed_count__gte=self.MIN_GALLERY_SIZE) | Q(use_windowed_classification=True))
            .exclude(person_name__in=[settings.BLANK_FACE_NAME] + list(settings.IGNORED_NAMES))
            .order_by('-confirmed_count')
        )
        data = [
            {
                'person_id': p.id,
                'person_name': p.person_name,
                'confirmed_count': p.confirmed_count,
                'use_windowed_classification': p.use_windowed_classification,
                'classification_window_days': p.classification_window_days,
            }
            for p in people
        ]
        serializer = api_ser.WindowedClassificationStatusSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def set_windowed(self, request):
        # Accessible as <root>/api/windowed_classification_config/set_windowed/
        # Accept: POST, body: {"person_id": N, "enabled": true, "window_days": 182}
        # ("window_days" optional -- only updated if given, so toggling
        # off and back on doesn't require re-specifying it.)
        from face_manager.models import Person

        payload = request.data
        if 'person_id' not in payload or 'enabled' not in payload:
            return render_404(request, 'Body must include "person_id" and "enabled".')
        if not isinstance(payload['enabled'], bool):
            return render_404(request, 'The "enabled" value must be a boolean.')

        try:
            person = Person.objects.get(id=payload['person_id'])
        except Person.DoesNotExist:
            return render_404(request, f'No Person with id {payload["person_id"]}.')

        person.use_windowed_classification = payload['enabled']
        if 'window_days' in payload:
            if not isinstance(payload['window_days'], int) or payload['window_days'] <= 0:
                return render_404(request, 'The "window_days" value must be a positive integer.')
            person.classification_window_days = payload['window_days']
        person.save()

        return Response({
            'person_id': person.id,
            'use_windowed_classification': person.use_windowed_classification,
            'classification_window_days': person.classification_window_days,
        })


class PhotoCollectionConfigViewSet(viewsets.ViewSet):
    """Backend for a frontend settings page letting a human tune a
    PhotoCollection's detection floors -- see filepopulator.models.
    PhotoCollection's own docstring for what min_face_det_score/
    min_face_relative_size gate. Deliberately narrow (list + one
    write action), same reasoning as PersonViewSet above: name/
    root_directory/is_default are read-only here (and in the Django
    admin page for this model) since changing them doesn't re-scope
    already-ingested images. Setting either floor triggers
    PhotoCollection.save()'s own apply_collection_floor_change()
    dispatch automatically -- no special handling needed here."""

    permission_classes = (IsAuthenticated,)
    serializer_class = api_ser.PhotoCollectionFloorStatusSerializer

    def list(self, request):
        collections = PhotoCollection.objects.all().order_by('name')
        data = [
            {
                'collection_id': c.id,
                'name': c.name,
                'root_directory': c.root_directory,
                'is_default': c.is_default,
                'min_face_det_score': c.min_face_det_score,
                'min_face_relative_size': c.min_face_relative_size,
            }
            for c in collections
        ]
        serializer = api_ser.PhotoCollectionFloorStatusSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def set_floors(self, request):
        # Accessible as <root>/api/photo_collection_config/set_floors/
        # Accept: POST, body: {"collection_id": N, "min_face_det_score": 0.6,
        # "min_face_relative_size": null} -- either floor key is optional
        # (only updated if present); passing null clears that floor.
        payload = request.data
        if 'collection_id' not in payload:
            return render_404(request, 'Body must include "collection_id".')

        try:
            collection = PhotoCollection.objects.get(id=payload['collection_id'])
        except PhotoCollection.DoesNotExist:
            return render_404(request, f'No PhotoCollection with id {payload["collection_id"]}.')

        for field in ('min_face_det_score', 'min_face_relative_size'):
            if field not in payload:
                continue
            value = payload[field]
            if value is not None:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return render_404(request, f'"{field}" must be a number or null.')
                if not (0.0 <= value <= 1.0):
                    return render_404(request, f'"{field}" must be between 0 and 1.')
            setattr(collection, field, value)
        collection.save()

        return Response({
            'collection_id': collection.id,
            'min_face_det_score': collection.min_face_det_score,
            'min_face_relative_size': collection.min_face_relative_size,
        })


# Mobile-app-facing views (ConfidentUnlabeledView, UnlabeledMobileInfo,
# ResetFace, MobileNameList) live in api/mobile_views.py, not here.
