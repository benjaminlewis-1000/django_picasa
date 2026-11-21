from __future__ import absolute_import, unicode_literals

# from .scripts import populateFromImageMultiGPU, establish_server_connection, establish_multi_server_connection
from .models import (
    Person, Face, clear_confirmed_ignore_face_encodings,
    clear_encodings_below_collection_floor, clear_stale_proposals_with_no_encoding,
)
from assign_faces import faceAssigner
from verification_clustering import cluster_all_unverified_faces
from face_manager.rotation_audit import run_rotation_audit_until_done
from face_manager.rotation_correction import run_rotation_correction
from face_manager.rotation_ml_audit import run_rotation_ml_audit_until_done
from celery import shared_task
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Q
from django.utils.crypto import get_random_string
from face_extract_encode import FaceExtractor
from filepopulator.models import ImageFile, VideoFile, PhotoCollection
from common.advisory_lock import advisory_lock
from common.error_log import log_error
from common.heartbeat import record_heartbeat_start, record_heartbeat_done
from picasa import celery_app
import os
import queue
import random
import signal
import threading
import time
import traceback


# from image_face_extractor import reencoder, ip_finder

# Per-video timeout for process_video_faces() below -- a pathological
# video (an unusually large raw detection/track count blowing up the
# clustering step's distance-matrix computation) can otherwise stall the
# whole batch indefinitely, since max_runtime_seconds is only checked
# *between* videos. Confirmed necessary 2026-09-09: a real production
# video ran for 2+ hours with zero completion (vs. the slowest known-good
# real video, 58 face groups, ~17 minutes) and had to be killed by hand.
# Known limitation: a signal-based alarm can only interrupt at a Python
# bytecode boundary, so it won't preempt a single very-long-running
# C-level call (e.g. one AgglomerativeClustering.fit() invocation)
# mid-call -- it fires as soon as that call returns, which still bounds
# *most* realistic slow-python-loop cases but isn't a hard real-time
# guarantee against every possible blowup.
#
# A flat timeout (1800s, then 2700s -- see git history) kept being wrong
# in both directions: short, completely normal clips were timing out
# purely from CPU contention with another concurrent job (confirmed real
# 2026-09-17), while a real multi-hour video legitimately needs far more
# than 45 minutes regardless of contention. Replaced 2026-11-18 (per the
# user's own request) with a scaled, periodically-recomputed budget:
#   allowed = clamp((BASE + duration_seconds * RATE) * contention, MIN, MAX)
# -- duration-scaled so a longer video gets a proportionally longer
# budget, contention-scaled so a CPU-busy host gets more slack too, and
# clamped on both ends: MIN keeps a short video at least as well off as
# the old flat timeout even with zero detected contention; MAX is a
# final backstop (a genuinely pathological, non-duration-related blowup
# -- the clustering-blowup case this whole mechanism originally existed
# to catch -- still can't run forever). RATE=3.0 sits comfortably above
# the real measured wall-time-per-second-of-footage ratio this pipeline
# has shown even under contention (1.5x-1.76x, see CLAUDE.md's
# 2026-09-12 throughput write-up).
#
# Real incident the same day this was built (video 9575, a 5-minute
# clip, timed out under contention): an earlier draft floored the
# duration-based budget at MIN *before* applying contention, which
# worked out fine for that case but needlessly complicated the formula
# -- simplified to the user's own suggestion (apply contention first,
# then clamp the whole thing), which gives the same floor guarantee at
# baseline while staying one clamp instead of two.
#
# CONTENTION_MAX_MULTIPLIER was originally set to 6.0, informed by this
# pipeline's own real measured contention impact (2026-09-17: one stage
# alone, _detect_and_track, slowed 5-10x under a concurrent CPU-heavy
# job) -- but that was sized to cover a video stalling under sustained
# contention for its ENTIRE run. Once _make_progress_callback() started
# resetting the timeout clock on every completed stage (same day, per
# the user's own follow-up), a video that's merely slow-but-advancing no
# longer depends on this multiplier alone -- the per-stage reset already
# protects it. Backed off 6.0 -> 3.0 accordingly: still real headroom
# for a single stage genuinely running slow under contention, without
# needing to cover a whole video's worst-case sustained slowdown in one
# number.
PER_VIDEO_TIMEOUT_BASE_SECONDS = 300
PER_VIDEO_TIMEOUT_RATE_PER_VIDEO_SECOND = 3.0
PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER = 3.0
PER_VIDEO_TIMEOUT_MIN_SECONDS = 2700
PER_VIDEO_TIMEOUT_MAX_SECONDS = 4 * 3600

# The user's own follow-up: CPU contention can change WHILE a video is
# still processing, not just at its start -- a single timeout value
# computed once at launch can't adapt to that. So instead of one
# signal.alarm(), a repeating signal.setitimer() fires every
# PER_VIDEO_TIMEOUT_CHECK_INTERVAL_SECONDS and re-evaluates the allowed
# budget from scratch using the CURRENT load average each time -- if the
# host gets busier mid-run, later checks see a larger contention
# multiplier and the video simply keeps running; only real, sustained
# elapsed-time-over-budget ever raises.
#
# A second follow-up the same day: "elapsed" above is time since the
# video's CURRENT STAGE started, not time since the video itself
# started -- _make_progress_callback()'s closure resets _video_timeout_
# state['start'] every time VideoFaceExtractor.process_video() advances
# to a new stage (detect_and_track -> pool_representative_frames ->
# clustering -> ...). A video that keeps successfully finishing one
# stage after another is making real, visible progress even if each
# stage is individually slow under contention -- the budget now caps how
# long any ONE stage can run without advancing, not the video's total
# wall time, so it's never penalized for the sum of several genuinely-
# completed stages exceeding what one whole-video budget would allow.
PER_VIDEO_TIMEOUT_CHECK_INTERVAL_SECONDS = 60

# A timeout is left unprocessed (see the timeout except branch below) so
# a plausibly-transient one (resource contention from another concurrent
# job -- confirmed real 2026-09-17) gets retried automatically on a later
# run, rather than requiring a manual reset. But a genuinely-too-long
# video (not just contention) would otherwise retry forever, each attempt
# burning its own dynamically-computed budget -- MAX_VIDEO_TIMEOUT_RETRIES
# caps that: once a video has timed out this many times, it's marked a
# real failure (face_extraction_failed=True) instead of being retried
# again, per the user's explicit call.
MAX_VIDEO_TIMEOUT_RETRIES = 5


class _VideoProcessingTimeout(Exception):
    pass


def _contention_multiplier():
    """How much busier this host is right now than its own core count,
    clamped to a sane range. 1.0 = no extra slack (load at or below core
    count); PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER = fully capped,
    so one extreme spike can't make a timeout effectively infinite."""
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        return 1.0
    cpu_count = os.cpu_count() or 1
    return min(
        PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER,
        max(1.0, load1 / cpu_count),
    )


def _video_timeout_budget_seconds(duration_seconds):
    """The allowed processing budget for a video of this duration, at the
    CURRENT moment's contention level. Pure except for the one real-time
    read (current load average, via _contention_multiplier()) --
    duration_seconds is a plain argument so this is directly unit-
    testable without mocking os.getloadavg(). Deliberately re-reads
    contention on every call rather than caching it -- see
    _check_video_timeout() below, which calls this fresh on every
    periodic tick specifically so a video already in progress gets more
    slack automatically if contention rises partway through, not just
    whatever was true when it started."""
    duration_seconds = duration_seconds or 0
    scaled = (
        PER_VIDEO_TIMEOUT_BASE_SECONDS + duration_seconds * PER_VIDEO_TIMEOUT_RATE_PER_VIDEO_SECOND
    ) * _contention_multiplier()
    return min(max(scaled, PER_VIDEO_TIMEOUT_MIN_SECONDS), PER_VIDEO_TIMEOUT_MAX_SECONDS)


# Mutable state for the currently-processing video, read by
# _check_video_timeout() on every periodic tick -- a plain module-level
# dict rather than a closure/argument, since signal handlers are called
# with a fixed (signum, frame) signature. Set right before each video's
# signal.setitimer() call in process_video_faces() and cleared right
# after, so a stray leftover tick from a prior video (if one somehow fired
# after setitimer(0, 0)) is a harmless no-op.
_video_timeout_state = {}


def _make_progress_callback():
    """Returns a closure VideoFaceExtractor.process_video() calls with a
    short stage-name string right before each major phase starts (see
    its own docstring in video_face_pipeline.py) -- resets _video_
    timeout_state['start'] on every call, per the user's own follow-up
    request (2026-11-18): a video that keeps successfully completing one
    stage after another (detect_and_track, then pool_representative_
    frames, then clustering, ...) is making real, visible progress even
    if each stage is individually slow under contention -- it shouldn't
    be killed just because the SUM of several completed stages happens
    to exceed the whole-video budget. The timeout now effectively caps
    how long any ONE stage can run without advancing, not the video's
    total wall time -- _check_video_timeout below is unchanged, it just
    measures time since the last reset rather than time since the video
    started. Also records the stage name itself, so a timeout's error
    message says WHICH stage it was stuck in.

    Also logs each completed stage's own wall time plus the host's real
    load average/contention multiplier AT THAT MOMENT (2026-11-18, per
    the user's own request for "solid math" backing up a slow/contended
    video rather than just a bare pass/fail) -- this is durable,
    appears in picasa_debug.log for every video regardless of whether it
    times out, and is exactly what distinguishes "this stage took a long
    time under heavy load" from "this stage never finished at all": the
    LAST stage before a real timeout simply never logs its own
    completion line, while every stage before it shows a real number."""
    def _progress(stage):
        now = time.time()
        prev_stage = _video_timeout_state.get('stage')
        prev_start = _video_timeout_state.get('start')
        if prev_stage is not None and prev_start is not None:
            settings.LOGGER.info(
                "video_face_extraction progress: stage '%s' finished in "
                "%.1fs (load avg %.2f / %d cores, contention x%.2f)",
                prev_stage, now - prev_start, os.getloadavg()[0],
                os.cpu_count() or 1, _contention_multiplier(),
            )
        _video_timeout_state['stage'] = stage
        _video_timeout_state['start'] = now
    return _progress


def _check_video_timeout(signum, frame):
    state = _video_timeout_state
    if not state:
        return
    elapsed = time.time() - state['start']
    allowed = _video_timeout_budget_seconds(state['duration_seconds'])
    if elapsed >= allowed:
        stage = state.get('stage')
        stage_note = f", stuck in stage '{stage}'" if stage is not None else ''
        raise _VideoProcessingTimeout(
            f"Video processing exceeded its dynamically-computed "
            f"{allowed:.0f}s budget without advancing to a new stage "
            f"(stalled {elapsed:.0f}s, duration {state['duration_seconds']}s, "
            f"contention x{_contention_multiplier():.2f}{stage_note})"
        )

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

def _log_task_crash(task_name, exc):
    """Shared helper for the scheduled tasks below: a whole-task crash is
    genuinely rare/exceptional (not a routine per-item failure, which
    already has its own dedicated tracking -- FailedImageFile,
    face_extraction_failed, etc.) -- but several of these tasks used to
    wrap their entire body in a bare `except:` that discarded the real
    exception completely, not even writing it to the log files
    (confirmed real, 2026-10-21). Logs the real exception AND records it
    via common.error_log.log_error(), so a genuine crash is visible in
    the frontend instead of only in log files a human has to think to
    go check -- log files that, until now, didn't even have the detail
    in them to find."""
    settings.LOGGER.error(f"{task_name} crashed: {type(exc).__name__}: {exc}", exc_info=True)
    log_error(f'face_manager.tasks.{task_name}', f"{type(exc).__name__}: {exc}")

@shared_task(ignore_result=True, name='face_manager.face_extraction')
def process_faces(max_runtime_seconds=None):
    """max_runtime_seconds, if given, is passed straight through to
    FaceExtractor.find_and_encode_faces() -- stops the loop between
    images (never mid-image) once elapsed wall time exceeds it. Used to
    keep this off CPU overnight (see CELERY_BEAT_SCHEDULE's 'face_add'
    entry) the same way process_video_faces() already bounds the video
    pipeline."""
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
        extractor.find_and_encode_faces(max_runtime_seconds=max_runtime_seconds)

    except Exception as e:
        _log_task_crash('process_faces', e)

@shared_task(ignore_result=True, name='face_manager.video_face_extraction')
def process_video_faces(max_runtime_seconds=None, video_ids=None):
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
    re-invoking this after a cooldown gap.

    video_ids, if given, scopes this run to exactly those VideoFile ids
    (an on-demand, frontend-triggered reprocess -- see
    api.views.VideoFaceExtractionViewSet) instead of the normal
    isProcessed=False backlog, and FORCES a redo regardless of current
    isProcessed/face_extraction_failed state: each named video's
    existing Face rows (if any) are deleted first (per-instance, not a
    bulk queryset delete, so thumbnail files get cleaned up -- same
    reasoning as ImageFile.delete()'s own fix elsewhere in this
    codebase), and its failure/timeout bookkeeping is reset, before
    reprocessing. max_runtime_seconds still applies the same way."""
    with advisory_lock('face_manager.video_face_extraction') as acquired:
        if not acquired:
            settings.LOGGER.debug("Video face extraction is locked, exiting.")
            return

        if video_ids:
            targeted = VideoFile.objects.filter(id__in=video_ids)
            for video in targeted:
                for face in video.face_set.all():
                    face.delete()
                video.face_extraction_failed = False
                video.face_extraction_error = None
                video.face_extraction_timeout_count = 0
                video.isProcessed = False
                video.save()
            unprocessed = VideoFile.objects.filter(id__in=video_ids)
        else:
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
                signal.signal(signal.SIGALRM, _check_video_timeout)
                _video_timeout_state.clear()
                _video_timeout_state['start'] = time.time()
                _video_timeout_state['duration_seconds'] = video.duration_seconds or 0
                signal.setitimer(
                    signal.ITIMER_REAL,
                    PER_VIDEO_TIMEOUT_CHECK_INTERVAL_SECONDS,
                    PER_VIDEO_TIMEOUT_CHECK_INTERVAL_SECONDS,
                )
                try:
                    faces = extractor.process_video(video, progress_cb=_make_progress_callback())
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0, 0)
                    _video_timeout_state.clear()
            except _VideoProcessingTimeout as exc:
                # A timeout is plausibly just resource contention
                # (confirmed real 2026-09-17: several short, completely
                # normal videos timed out purely because a concurrent
                # CPU-heavy reclassification backfill was running at the
                # same time, not because of anything wrong with the file
                # itself) -- NOT marked isProcessed/face_extraction_failed
                # like every other failure below, so it's retried
                # automatically by a future scheduled run instead of
                # needing a manual reset. But a genuinely-too-long video
                # (not just contention) would otherwise retry forever,
                # each attempt burning its own dynamically-computed budget
                # -- face_extraction_timeout_count caps that at
                # MAX_VIDEO_TIMEOUT_RETRIES, per the user's explicit call.
                video.face_extraction_timeout_count += 1
                if video.face_extraction_timeout_count >= MAX_VIDEO_TIMEOUT_RETRIES:
                    settings.LOGGER.error(
                        f"Video face extraction timed out for {video.filename} "
                        f"{video.face_extraction_timeout_count} times -- giving up "
                        f"and marking it a real failure: {exc}"
                    )
                    video.face_extraction_failed = True
                    video.face_extraction_error = (
                        f"Timed out {video.face_extraction_timeout_count} times: {exc}"
                    )[:2000]
                    video.isProcessed = True
                    log_error(
                        'face_manager.tasks.process_video_faces',
                        f"{video.filename} gave up after "
                        f"{video.face_extraction_timeout_count} timeouts: {exc}",
                        related_video_id=video.id,
                    )
                else:
                    settings.LOGGER.warning(
                        f"Video face extraction timed out for {video.filename} "
                        f"(attempt {video.face_extraction_timeout_count}/"
                        f"{MAX_VIDEO_TIMEOUT_RETRIES}) -- leaving unprocessed to "
                        f"retry later rather than marking it a permanent failure: {exc}"
                    )
                video.save()
                continue
            except Exception as exc:
                settings.LOGGER.error(
                    f"Video face extraction failed for {video.filename}", exc_info=True
                )
                # Mark processed regardless of success/failure, same as
                # find_and_encode_faces()'s corrupted-image handling --
                # a video that fails for a real (non-timeout) reason
                # shouldn't be retried forever on every scheduled run.
                # face_extraction_failed/_error make this visible on
                # server_stats and queryable, unlike FailedVideoFile
                # (which is about ingestion, not face extraction).
                video.face_extraction_failed = True
                video.face_extraction_error = f"{type(exc).__name__}: {exc}"[:2000]
                video.isProcessed = True
                log_error(
                    'face_manager.tasks.process_video_faces',
                    f"{video.filename} failed: {type(exc).__name__}: {exc}",
                    related_video_id=video.id,
                )
                video.save()
                continue

            settings.LOGGER.debug(
                f"Video face extraction: {video.filename} -> {len(faces)} face group(s)."
            )
            video.isProcessed = True
            # Reset in case this video had previously timed out a few
            # times before eventually succeeding -- a stale nonzero count
            # would otherwise wrongly count toward MAX_VIDEO_TIMEOUT_RETRIES
            # if this video were ever manually reset and reprocessed again.
            video.face_extraction_timeout_count = 0
            video.save()

            # A prior failure/timeout-giveup for this exact video is no
            # longer applicable now that it just succeeded -- auto-dismiss
            # any ApplicationError row(s) still open about it, so the
            # notification bell doesn't keep flagging a problem that's
            # already resolved (2026-11-18, see
            # ApplicationError.related_video_id's own docstring). Scoped to
            # this video's id and still-undismissed rows only -- never
            # touches a row already dismissed for some other reason.
            from common.models import ApplicationError
            from django.utils import timezone
            ApplicationError.objects.filter(
                related_video_id=video.id, dismissed=False,
            ).update(dismissed=True, dismissed_at=timezone.now())

@shared_task(ignore_result=True, name='face_manager.reencode')
def reencode_missing_faces():
    # Overlap protection used to live here as a celery_app.control.
    # inspect().active() check -- a check-then-act race (two tasks
    # starting close together can each see "0 others running" before
    # either registers), and one that only covered other Celery
    # invocations of this same task, not a direct manage.py shell/
    # management-command call (confirmed as a real gap 2026-10-25: a
    # manual call and this scheduled task ran fully concurrently, each
    # building its own separate FaceExtractor/ONNX sessions, stacking
    # memory on top of each other). FaceExtractor.reencode_missing_
    # faces() itself now holds a Postgres advisory lock for its whole
    # run (see common/advisory_lock.py), which is atomic and covers
    # every entry point, so this task no longer needs its own check
    # for THAT specific collision.
    #
    # A second, real collision found the same day: the one-time
    # parallel backfill (backfill_lvface_parallel) shards the backlog
    # under per-shard lock names, which are deliberately different from
    # this task's own plain lock name so they never block each other --
    # but that also means this task was free to run concurrently and do
    # real, uncoordinated, REDUNDANT work on the same backlog the
    # parallel backfill was already covering, while competing with it
    # for the exact CPU/memory headroom the parallel backfill's own
    # supervisor was trying to budget around. Guarded here with a
    # SEPARATE, shared "is the parallel backfill active at all" marker
    # lock -- the supervisor holds it for its whole run; this task
    # tries to acquire it too (for the duration of its own run, so a
    # supervisor starting mid-task has to wait its turn as well, not
    # just the other direction) and skips entirely if it can't.
    with advisory_lock('face_manager.reencode_missing_faces.parallel_backfill_active') as acquired:
        if not acquired:
            settings.LOGGER.info(
                "reencode_missing_faces: the one-time parallel backfill is active -- "
                "skipping this scheduled run rather than competing with it."
            )
            return
        try:
            extractor = FaceExtractor()
            extractor.reencode_missing_faces()
        except Exception as e:
            _log_task_crash('reencode_missing_faces', e)

@shared_task(ignore_result=True, name='face_manager.apply_collection_floor_change')
def apply_collection_floor_change(collection_id):
    # Fired automatically by PhotoCollection.save() whenever
    # min_face_det_score/min_face_relative_size actually changes -- see
    # that model's save() override. Handles both directions: a tightened
    # floor leaves already-encoded, not-yet-adjudicated faces with a now-
    # stale embedding (cleared here, no model inference needed); a
    # loosened floor leaves faces that were previously skipped still
    # missing one (re-encoded here, which does need real model inference,
    # hence FaceExtractor()). Scoped to just this collection's images via
    # the `collection=` param both functions take -- an unrelated
    # collection's floor changing never touches this collection's faces.
    try:
        collection = PhotoCollection.objects.get(pk=collection_id)
    except PhotoCollection.DoesNotExist:
        return

    with advisory_lock(f'face_manager.apply_collection_floor_change.{collection_id}') as acquired:
        if not acquired:
            settings.LOGGER.warning(
                f"apply_collection_floor_change already running for collection {collection_id}, skipping."
            )
            return

        try:
            scanned, cleared = clear_encodings_below_collection_floor(collection=collection)
            settings.LOGGER.debug(
                f"Collection {collection_id} floor change: cleared {cleared}/{scanned} encodings."
            )
        except Exception as e:
            settings.LOGGER.exception(
                f"Error clearing encodings below the floor for collection {collection_id}."
            )
            log_error(
                'face_manager.tasks.apply_collection_floor_change',
                f"Error clearing encodings below the floor for collection {collection_id}: "
                f"{type(e).__name__}: {e}",
            )

        try:
            # Catch-up for any already-null-encoded face still carrying a
            # stale poss_identN proposal (e.g. a prior run's clearing
            # pass, from before this cleanup existed, or any other gap) --
            # see clear_stale_proposals_with_no_encoding()'s own docstring.
            # Runs every time a threshold changes, not just once, so the
            # backlog can never silently reaccumulate for this collection.
            stale_cleared = clear_stale_proposals_with_no_encoding(collection=collection)
            settings.LOGGER.debug(
                f"Collection {collection_id} floor change: cleared {stale_cleared} stale unencoded proposal(s)."
            )
        except Exception as e:
            settings.LOGGER.exception(
                f"Error clearing stale unencoded proposals for collection {collection_id}."
            )
            log_error(
                'face_manager.tasks.apply_collection_floor_change',
                f"Error clearing stale unencoded proposals for collection {collection_id}: "
                f"{type(e).__name__}: {e}",
            )

        try:
            extractor = FaceExtractor()
            extractor.reencode_missing_faces(collection=collection)
        except Exception as e:
            settings.LOGGER.exception(
                f"Error re-encoding newly-eligible faces for collection {collection_id}."
            )
            log_error(
                'face_manager.tasks.apply_collection_floor_change',
                f"Error re-encoding newly-eligible faces for collection {collection_id}: "
                f"{type(e).__name__}: {e}",
            )

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
    except Exception as e:
        _log_task_crash('clear_ignored_encodings', e)

@shared_task(ignore_result=True, name='face_manager.assign_faces')
def assign_faces_task(redo_all=False):

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

    # Only the weekly redo_all=True run is currently watched for
    # silence/hangs (see common/tasks.py's HEARTBEAT_REGISTRY) -- the
    # hourly incremental run isn't registered there, so recording its
    # heartbeat too is harmless (just an unused row) rather than
    # something that needs its own guard here.
    heartbeat_name = (
        'face_manager.assign_faces.weekly_full' if redo_all else 'face_manager.assign_faces.hourly'
    )
    record_heartbeat_start(heartbeat_name)
    try:
        classer = faceAssigner()
        classer.execute(redo_all)
        settings.LOGGER.info(
            f"assign_faces_task(redo_all={redo_all}) completed."
        )
    except Exception as e:
        _log_task_crash('assign_faces_task', e)
    finally:
        record_heartbeat_done(heartbeat_name)

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
    except Exception as e:
        _log_task_crash('cluster_unverified_faces_task', e)

@shared_task(ignore_result=True, name='face_manager.audit_image_rotation')
def audit_image_rotation_task():
    """Scheduled wrapper around face_manager.rotation_audit.
    run_rotation_audit_until_done() -- see that module and CLAUDE.md's
    2026-11-03/11-05/11-06 orientation-detection write-up for the design.
    Same advisory-lock convention as the other scheduled tasks here;
    covers manual (management-command) and scheduled invocations alike.

    Loops bounded batches (see run_rotation_audit_until_done()'s own
    comment for why that's safe memory-wise) rather than stopping after
    one, so a large backlog is ground through within this one run
    instead of trickling one batch per hourly tick.
    """
    with advisory_lock('face_manager.audit_image_rotation') as acquired:
        if not acquired:
            settings.LOGGER.debug("Image rotation audit is locked, exiting.")
            return
        try:
            result = run_rotation_audit_until_done()
            settings.LOGGER.debug(
                f"audit_image_rotation_task: examined {result['examined']}, "
                f"updated {result['updated']} across {result['batches']} batch(es) "
                f"(hit_time_limit={result['hit_time_limit']}) -- {result['counts']}"
            )
        except Exception as e:
            _log_task_crash('audit_image_rotation_task', e)

@shared_task(ignore_result=True, name='face_manager.audit_image_rotation_ml')
def audit_image_rotation_ml_task():
    """Scheduled wrapper around face_manager.rotation_ml_audit.
    run_rotation_ml_audit_until_done() -- the ML-classifier counterpart
    to audit_image_rotation_task above. Built 2026-11-21 but NOT added
    to CELERY_BEAT_SCHEDULE (see rotation_ml_audit.py's own docstring) --
    this task exists so it's ready to schedule later, but currently only
    ever runs if triggered manually (e.g. via `celery call` or the
    run_rotation_ml_audit management command calling the underlying
    function directly).
    """
    with advisory_lock('face_manager.audit_image_rotation_ml') as acquired:
        if not acquired:
            settings.LOGGER.debug("ML image rotation audit is locked, exiting.")
            return
        try:
            result = run_rotation_ml_audit_until_done()
            if result.get('skipped_no_checkpoint'):
                settings.LOGGER.debug("audit_image_rotation_ml_task: no checkpoint found, skipped.")
                return
            settings.LOGGER.debug(
                f"audit_image_rotation_ml_task: examined {result['examined']}, "
                f"updated {result['updated']} across {result['batches']} batch(es) "
                f"(hit_time_limit={result['hit_time_limit']}) -- {result['counts']}"
            )
        except Exception as e:
            _log_task_crash('audit_image_rotation_ml_task', e)

@shared_task(ignore_result=True, name='face_manager.correct_image_rotation')
def correct_image_rotation_task():
    """Scheduled SAFETY-NET sweep around face_manager.rotation_correction.
    run_rotation_correction() -- catches any CONFIRMED-but-not-yet-
    corrected image the on-demand correct_image_rotation_ids_task below
    missed (e.g. a lost broker message, a worker crash mid-task). The
    common case (a human confirms, sees it corrected within moments) is
    handled by that on-demand task instead -- this one is deliberately
    infrequent (see its own CELERY_BEAT_SCHEDULE entry) since it's only
    ever meant to catch the rare leftover, not deliver the primary UX.
    run_rotation_correction() itself takes a per-image advisory lock, so
    this can safely overlap an in-flight on-demand task without either
    double-processing the same row.
    """
    with advisory_lock('face_manager.correct_image_rotation') as acquired:
        if not acquired:
            settings.LOGGER.debug("Image rotation correction sweep is locked, exiting.")
            return
        try:
            result = run_rotation_correction()
            settings.LOGGER.debug(
                f"correct_image_rotation_task: corrected {result['corrected']}, "
                f"failed {result['failed']}"
            )
        except Exception as e:
            _log_task_crash('correct_image_rotation_task', e)


@shared_task(ignore_result=True, name='face_manager.correct_image_rotation_ids')
def correct_image_rotation_ids_task(image_ids):
    """On-demand (NOT scheduled -- dispatched directly via .delay() right
    after a human confirms a rotation, see api.rotation_review_views'
    RotationReviewBulkActionView) wrapper around run_rotation_correction(),
    scoped to exactly the ids just confirmed. Fast turnaround (the human
    sees the corrected image within moments) is the whole point of this
    task existing separately from the scheduled sweep above -- no
    advisory lock at the TASK level here (a small, specific id list is
    cheap and safe to run concurrently with the sweep or another on-
    demand dispatch; run_rotation_correction()'s own per-image lock is
    what actually prevents double-processing any single row).

    If this task's message is ever lost entirely (broker restart, a
    worker crash before it starts), there's no retry here -- the
    scheduled sweep above is exactly the recourse for that case, just on
    its own (longer) cadence.
    """
    try:
        result = run_rotation_correction(image_ids=image_ids)
        settings.LOGGER.debug(
            f"correct_image_rotation_ids_task({image_ids}): corrected {result['corrected']}, "
            f"failed {result['failed']}"
        )
    except Exception as e:
        _log_task_crash('correct_image_rotation_ids_task', e)
