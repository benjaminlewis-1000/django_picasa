from __future__ import absolute_import, unicode_literals

# from .scripts import populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection
from .models import Person, Face, clear_confirmed_ignore_face_encodings
from assign_faces import faceAssigner
from verification_clustering import cluster_all_unverified_faces
from celery import shared_task
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Q
from django.utils.crypto import get_random_string
from face_extract_encode import FaceExtractor
from filepopulator.models import ImageFile, VideoFile
from common.advisory_lock import advisory_lock
from picasa import celery_app
import os
import queue
import random
import threading
import time
import traceback


# from image_face_extractor import reencoder, ip_finder 

if not settings.configured:
    settings.configure()

# If you don’t care about the results of a task, be sure to set the ignore_result option,
# as storing results wastes time and resources.
# Shared tasks are to make apps without any concrete app instance. 
# Tasks depend on the app I guess.
# Tasks can be autodiscovered by placing the app in the projec, then creating 
# celery.py in the <PROJECT> directory and putting the following:
# app.config_from_object('django.conf:settings', namespace='CELERY')
# app.autodiscover_tasks() 

@shared_task(ignore_result=True, name='face_manager.face_extraction')
def process_faces():
    starttime = time.time()

    try:
        settings.LOGGER.debug("Starting face extraction...")

        # Overlap protection used to live here as a celery_app.control.
        # inspect().active() check -- a check-then-act race (two tasks
        # starting close together can each see "0 others running" before
        # either registers), and one that only covered other Celery
        # invocations of this same task, not a direct manage.py shell/
        # management-command call. find_and_encode_faces() itself now
        # holds a Postgres advisory lock for its whole run (see
        # common/advisory_lock.py), which is atomic and covers every
        # entry point, so this task no longer needs its own check.

        unprocessed_imgs = ImageFile.objects.filter(isProcessed=False).all()
        unprocessed_count = ImageFile.objects.filter(isProcessed=False).count()

        if unprocessed_count == 0:
            settings.LOGGER.debug("No images to extract! Exiting." )
            return

        extractor = FaceExtractor()
        extractor.find_and_encode_faces()

    except:

        settings.LOGGER.debug("Ending face adding task")
        
@shared_task(ignore_result=True, name='face_manager.video_face_extraction')
def process_video_faces(max_runtime_seconds=None):
    """Phase 3: one Face row per union-merge track-group, per unprocessed
    VideoFile -- see face_manager/video_face_pipeline.py and CLAUDE.md's
    Phase 3 design. Same advisory-lock convention as face_extraction
    (find_and_encode_faces): the lock covers the whole run, atomic across
    every entry point (scheduled task, manage.py shell, management
    command), and is released automatically if the holding connection
    ever drops -- no stale-lock cleanup needed.

    max_runtime_seconds, if given, stops the loop (and releases the lock)
    once elapsed wall time exceeds it -- checked only *between* videos,
    never mid-video, so a video's own Face-group creation is never
    interrupted partway (which would otherwise risk duplicate Face rows
    on a later retry, since isProcessed is only set True in the finally
    block once a video's processing has actually finished). Used to run
    the full-library backfill in bounded windows (e.g. 2.5h) rather than
    pinning every core for days straight, with an external scheduler
    re-invoking this after a cooldown gap."""
    with advisory_lock('face_manager.video_face_extraction') as acquired:
        if not acquired:
            settings.LOGGER.debug("Video face extraction is locked, exiting.")
            return

        unprocessed = VideoFile.objects.filter(isProcessed=False)
        if not unprocessed.exists():
            settings.LOGGER.debug("No videos to extract faces from! Exiting.")
            return

        from video_face_pipeline import VideoFaceExtractor
        extractor = VideoFaceExtractor()
        start_time = time.time()
        for video in unprocessed:
            if max_runtime_seconds is not None and (time.time() - start_time) >= max_runtime_seconds:
                settings.LOGGER.debug(
                    f"Video face extraction: time budget ({max_runtime_seconds}s) reached, "
                    "stopping between videos."
                )
                break
            try:
                faces = extractor.process_video(video)
                settings.LOGGER.debug(
                    f"Video face extraction: {video.filename} -> {len(faces)} face group(s)."
                )
            except Exception:
                settings.LOGGER.error(
                    f"Video face extraction failed for {video.filename}", exc_info=True
                )
            finally:
                # Mark processed regardless of success/failure, same as
                # find_and_encode_faces()'s corrupted-image handling --
                # a video that fails once shouldn't be retried forever on
                # every scheduled run. No separate failure-tracking field
                # for this (unlike FailedVideoFile, which is about
                # ingestion, not face extraction) -- the error is logged,
                # not silently dropped, but not yet surfaced anywhere an
                # operator would see it without checking logs.
                video.isProcessed = True
                video.save()

@shared_task(ignore_result=True, name='face_manager.reencode')
def reencode_missing_faces():
    i = celery_app.control.inspect()
    active_tasks = i.active()
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.reencode':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        settings.LOGGER.debug("Reencode is locked, exiting.")
        settings.LOGGER.warning("Reencode locked!")
        return

    try:
        extractor = FaceExtractor()
        extractor.reencode_missing_faces()
    except:
        settings.LOGGER.debug("Ending reencode task")

@shared_task(ignore_result=True, name='face_manager.clear_ignored_encodings')
def clear_ignored_encodings():
    # Ongoing counterpart to the one-off clear_confirmed_ignore_encodings
    # management command (used for the initial backfill) -- keeps
    # newly-confirmed .ignore/.realignore faces' encodings cleared going
    # forward without needing the command re-run by hand. No FaceExtractor
    # needed here (no ML model involved), just the shared DB update.
    i = celery_app.control.inspect()
    active_tasks = i.active()
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.clear_ignored_encodings':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        settings.LOGGER.debug("Clear ignored encodings is locked, exiting.")
        settings.LOGGER.warning("Clear ignored encodings locked!")
        return

    try:
        updated = clear_confirmed_ignore_face_encodings()
        settings.LOGGER.debug(f"Cleared face_encoding_512 for {updated} confirmed-ignore face(s).")
    except:
        settings.LOGGER.debug("Ending clear_ignored_encodings task")

@shared_task(ignore_result=True, name='face_manager.assign_faces')
def thistask(redo_all=False):

    i = celery_app.control.inspect()
    active_tasks = i.active()
    task_running = False
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.assign_faces':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        # This task will be one, so looking for other tasks.
        settings.LOGGER.debug("Classification is locked, exiting.")
        settings.LOGGER.warning("Classification locked!")
        return

    try:
        classer = faceAssigner()
        classer.execute(redo_all)
    except:
        print("Image classification failed!")
        
@shared_task(ignore_result=True, name='face_manager.reload_classifier_pkl')
def classifier_pkl_file_reload():
    classer = faceAssigner()
    classer.load_encodings(reload_pkl_file = True)

@shared_task(ignore_result=True, name='face_manager.api_bulk_operation')
def api_bulk_operation(input_dict: dict):
    print("Executing a bulk operation task", input_dict)

@shared_task(ignore_result=True, name='face_manager.cluster_unverified_faces')
def cluster_unverified_faces_task():
    i = celery_app.control.inspect()
    active_tasks = i.active()
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        if len(tasks) != 0:
            for tt in tasks:
                if tt['name'] == 'face_manager.cluster_unverified_faces':
                    num_this_task_running += 1

    if num_this_task_running > 1:
        settings.LOGGER.debug("Cluster unverified faces is locked, exiting.")
        settings.LOGGER.warning("Cluster unverified faces locked!")
        return

    try:
        num_people, num_faces = cluster_all_unverified_faces()
        settings.LOGGER.debug(
            f"Clustered {num_faces} unverified face(s) across {num_people} person(s)."
        )
    except:
        settings.LOGGER.debug("Ending cluster_unverified_faces task")
