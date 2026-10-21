from __future__ import absolute_import, unicode_literals

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings
from picasa import celery_app

from celery import shared_task
import time
import os
# import scripts
from .scripts import create_image_file, add_from_root_dir, delete_removed_photos, update_dirs_datetime, check_file_mods, cleanup_orphaned_media_files
from .video_scripts import add_videos_from_root_dir, delete_removed_videos, transcode_pending_videos
from common.advisory_lock import advisory_lock

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

@shared_task(ignore_result=True, name='filepopulator.populate_files_from_root')
def load_images_into_db():

    # This used to have its own celery_app.control.inspect().active()
    # check-then-act guard here, checking for a task named
    # 'face_manager.populate_files_from_root' -- a name that never
    # existed (this task is actually registered as
    # 'filepopulator.populate_files_from_root', see the decorator
    # above), so the check could never trigger. It's also redundant
    # now regardless: add_from_root_dir() below holds its own real
    # Postgres advisory lock (common/advisory_lock.py), which is
    # atomic and covers every entry point, not just concurrent Celery
    # tasks of this same name -- the same reasoning that removed the
    # equivalent racy check from face_manager.tasks.process_faces().
    #
    # Walks every registered PhotoCollection's own root, not just the
    # single hardcoded FILEPOPULATOR_SERVER_IMG_DIR (== PHOTO_ROOT) --
    # the default collection's root_directory equals PHOTO_ROOT (seeded
    # at migration time), so this is behavior-preserving for the
    # existing library and additionally covers any other collection
    # (e.g. a newly-acquired relative's archive) with zero further
    # scheduling changes. See filepopulator/models.py's PhotoCollection
    # for the full design.
    from .models import PhotoCollection
    print("Loading images into database...")
    for collection in PhotoCollection.objects.all():
        add_from_root_dir(collection.root_directory)
    print("Finished adding photos")
    delete_removed_photos()
    print("Done! Finished removing photos that aren't there.")
    # update_dirs_datetime()

    # TODO: Check duplicates

@shared_task(ignore_result=True, name='filepopulator.populate_videos_from_root')
def load_videos_into_db():
    # Own advisory lock (see add_videos_from_root_dir()), own task, own
    # schedule -- matches the existing one-task-per-concern pattern
    # rather than folding this into load_images_into_db() above.
    #
    # Roots are DB-driven (every PhotoCollection's own root_directory,
    # always scanned for interspersed videos -- same precedent
    # PHOTO_ROOT already set; plus each collection's extra_video_roots,
    # for dedicated video-only archives like Our_Home_Videos that aren't
    # under any collection's photo root at all) -- replaces the
    # previously-hardcoded VIDEO_ROOTS/FILEPOPULATOR_SERVER_VIDEO_DIRS
    # settings, 2026-10-21, same motivation as load_images_into_db()
    # above: a new root shouldn't need a code deploy. See
    # filepopulator/models.py's PhotoCollection for the full design.
    from .models import PhotoCollection
    print("Loading videos into database...")
    roots = []
    for collection in PhotoCollection.objects.all():
        roots.append(collection.root_directory)
        roots.extend(collection.extra_video_roots or [])
    add_videos_from_root_dir(roots)
    print("Finished adding videos")
    delete_removed_videos()
    print("Done! Finished removing videos that aren't there.")

@shared_task(ignore_result=True, name='filepopulator.check_mod_dates')
def check_mod_dates():
    check_file_mods()

@shared_task(name='filepopulator.update_dir_dates')
def update_dir_dates():
    print("Updates")
    update_dirs_datetime()

@shared_task(ignore_result=True, name='filepopulator.geocode_new_images')
def geocode_new_images():
    # Small recurring batch, not a one-time backfill: newly-ingested
    # images with GPS accumulate between runs, but nowhere near fast
    # enough to need more than a modest cap per run given Nominatim's
    # ~1 req/sec rate limit (see filepopulator/geocode.py).
    i = celery_app.control.inspect()
    active_tasks = i.active()
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        for tt in tasks:
            if tt['name'] == 'filepopulator.geocode_new_images':
                num_this_task_running += 1

    if num_this_task_running > 1:
        settings.LOGGER.debug("Already running a geocoding task, exiting.")
        return

    from .geocode import run_geocoding_backfill
    run_geocoding_backfill(limit=200, log=settings.LOGGER.debug)

@shared_task(ignore_result=True, name='filepopulator.find_similar_images')
def find_similar_images():
    # Small recurring batch, not a one-time backfill: newly-ingested
    # images accumulate between runs, but the comparison itself is cheap
    # (vectorized numpy popcount(XOR), benchmarked at <1ms per image
    # against a 200k-image population -- see filepopulator/similarity.py),
    # so this cap is generous headroom, not a rate-limit workaround like
    # geocode_new_images' is.
    i = celery_app.control.inspect()
    active_tasks = i.active()
    num_this_task_running = 0
    for k in active_tasks.keys():
        tasks = active_tasks[k]
        for tt in tasks:
            if tt['name'] == 'filepopulator.find_similar_images':
                num_this_task_running += 1

    if num_this_task_running > 1:
        settings.LOGGER.debug("Already running a similarity-check task, exiting.")
        return

    from .similarity import run_similarity_check
    run_similarity_check(limit=2000, log=settings.LOGGER.debug)

@shared_task(ignore_result=True, name='filepopulator.merge_cross_collection_image_duplicates')
def merge_cross_collection_image_duplicates():
    # Recurring counterpart to the one-off merge_cross_collection_image_
    # duplicates management command -- processes only pairs that aren't
    # already resolved (find_unresolved_cross_collection_pairs), so a
    # newly-discovered cross-collection duplicate (e.g. from a future
    # collection being added, or ordinary ongoing ingestion) gets
    # resolved automatically without anyone re-running the command by
    # hand. See filepopulator/cross_collection_dedup.py for the full
    # design.
    with advisory_lock('filepopulator.merge_cross_collection_image_duplicates') as acquired:
        if not acquired:
            settings.LOGGER.debug("Already running cross-collection dedup, exiting.")
            return

        from .cross_collection_dedup import find_unresolved_cross_collection_pairs, resolve_pair
        for pair in find_unresolved_cross_collection_pairs():
            resolve_pair(pair, dry_run=False)

@shared_task(ignore_result=True, name='filepopulator.cleanup_orphaned_media')
def cleanup_orphaned_media():
    # Weekly, not hourly/daily -- this walks every file in 5 real media
    # directories (1.5M+ files found in the real backlog this was built
    # to clear, 2026-10-08), not a cheap per-row check. Orphans accumulate
    # slowly from normal reprocessing/move/restore activity, so a weekly
    # sweep is plenty to keep the backlog from quietly rebuilding, without
    # needing to run anywhere near as often as the hourly tasks above.
    import common
    with common.advisory_lock('filepopulator.cleanup_orphaned_media') as acquired:
        if not acquired:
            settings.LOGGER.debug("cleanup_orphaned_media already running, exiting.")
            return
        cleanup_orphaned_media_files(dry_run=False, min_age_hours=24.0, log=settings.LOGGER.info)

@shared_task(ignore_result=True, name='filepopulator.transcode_videos')
def transcode_videos(max_runtime_seconds=None):
    # Phase 5 (transcode pipeline), added 2026-10-15. Daily, not every
    # few hours like video_face_extraction -- ffmpeg's own re-encode is
    # just as CPU-heavy as the ONNX/torch inference that task does, and
    # trying to interleave two CPU-saturating tasks on a tight cycle
    # isn't worth the complexity; a capped daily window makes steady
    # progress through the one-time backlog without needing to dodge
    # video_face_extraction's near-continuous 3h/2.5h-active cycle.
    import common
    with common.advisory_lock('filepopulator.transcode_videos') as acquired:
        if not acquired:
            settings.LOGGER.debug("Video transcode already running, exiting.")
            return
        done, failed = transcode_pending_videos(max_runtime_seconds=max_runtime_seconds)
        settings.LOGGER.info(f"Video transcode: {done} done, {failed} failed.")
