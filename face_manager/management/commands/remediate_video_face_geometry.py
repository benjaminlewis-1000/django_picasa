#! /usr/bin/env python

# Fixes the real, confirmed bug found 2026-10-29 (see CLAUDE.md and
# audit_video_face_geometry.py's own docstring): a small but real
# fraction of video faces have a box/kps that doesn't actually bound the
# real face at all -- root-caused to _redetect_in_crop() silently
# accepting a low-confidence spurious detection when its padded search
# crop (built around a possibly-already-wrong upstream box) didn't
# contain the tracked person. That bug is now fixed going forward
# (REDETECT_MIN_SCORE fallback chain in video_face_pipeline.py); this
# command repairs the already-extracted faces it affected.
#
# Combines triage (the same thumbnail-based confidence+centeredness
# check audit_video_face_geometry.py uses) and remediation into one
# pass, rather than requiring the separate read-only audit to finish
# first -- a face that passes triage is skipped (cheap, no video
# decode); one that fails gets a real fix attempt via
# locate_confident_face_near_timestamp() (video_face_pipeline.py),
# which searches a window of real frames around the face's own stored
# timestamp -- closest first -- for a confident, correctly-positioned
# detection, not just the original (suspect) moment.
#
# Per the user's explicit call: if a fix is found, box/kps/det_score/
# thumbnail/timestamp are all updated, AND face_encoding_512 is
# recomputed from that one now-confirmed-good frame directly (replacing
# whatever multi-frame trimmed-centroid value was there before) --
# simpler and more trustworthy than trying to reconstruct the original
# multi-frame average, which isn't stored anywhere besides the final
# centroid.
#
# LVFace-B (--lvface-sizes default) is ALSO (re)computed for a fixed
# face, same as face_encoding_512 -- the frame is already decoded and
# new_kps already known at that point, so this is just 1 more small ONNX
# call, not a second expensive decode. B-only by default to match the
# image-side gallery backfill (backfill_verified_kps --lvface-sizes=B)
# this is complementing -- S/L are available via --lvface-sizes but
# aren't the size actually used for real comparisons. This is
# opportunistic, not a general video LVFace backfill: an "already ok"
# (triage-passed) face gets no new decode here at all, so its existing
# LVFace value (if any) is left alone untouched, stale or not -- there
# is currently no dedicated video-side LVFace backfill command (the
# existing backfill_verified_kps/backfill_lvface_targeted/
# backfill_lvface_parallel commands are all image-only,
# source_image_file__isnull=False). Any stale LVFace value on a FIXED
# face is always overwritten here (never skipped), since its kps just
# changed and the old value is now definitely wrong.
#
# A face where NO confident detection turns up anywhere in the search
# window is left completely untouched and reported separately (flagged
# for later, per the user's explicit call) -- never deleted or guessed
# at.
#
# An "already ok" (triage-passed) face ALSO gets a real video decode
# now (2026-10-29, per the user's explicit call) -- not the full
# confident-detection window search the "fixed" path uses (that face's
# box/kps are already trusted, no redetection needed), but the much
# cheaper extract_frame_near_timestamp() the on-demand viewer already
# uses: pixel-matches a small window of real frames against the face's
# own stored thumbnail (its real ground truth) to recover the exact
# frame, at a fraction of the fix path's cost. Once recovered, the real
# frame is run through _redetect_in_crop() around the stored box as a
# SECOND, independent validation beyond thumbnail-only triage (the
# thumbnail is itself just a crop of some frame, generated once at
# extraction time -- re-checking against a freshly-decoded real frame
# can catch a case thumbnail-only triage wouldn't) -- any disagreement
# is flagged (ok_frame_check_failed) for follow-up, not auto-"fixed"
# here, same "flag rather than guess" discipline as unresolved. A face
# that passes both checks gets LVFace-B computed from the real frame +
# the (possibly freshly-redetected, more precise) kps, since there's
# currently no dedicated video-side LVFace backfill at all -- skipped
# (not overwritten) if the field is already set, since the box/kps
# here haven't changed and an existing value isn't stale the way a
# FIXED face's always is.

import signal
import time

import cv2
import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Q

import lvface
from face_manager.models import Face
from video_face_pipeline import (
    VideoFaceExtractor, extract_frame_near_timestamp, ffprobe_info,
    locate_confident_face_near_timestamp,
)

MIN_SCORE = 0.5
CENTER_FRACTION = 0.7
SEARCH_WINDOW_SECONDS = 1.5
# Generous relative to the ~3s of video this normally decodes+searches,
# even under heavy CPU contention -- this project has repeated, real
# history of video/image decode calls hanging outright (not just being
# slow) under contention, hence the same SIGALRM safety net used
# elsewhere (backfill_det_score.py's PER_IMAGE_TIMEOUT_SECONDS,
# face_manager/tasks.py's PER_VIDEO_TIMEOUT_SECONDS).
PER_FACE_TIMEOUT_SECONDS = 60


class _FaceTimeout(Exception):
    pass


def _raise_face_timeout(signum, frame):
    raise _FaceTimeout(f"Face processing exceeded {PER_FACE_TIMEOUT_SECONDS}s")


class Command(BaseCommand):
    help = (
        "Fixes video faces whose box/kps don't actually bound a real face "
        "(see audit_video_face_geometry.py and CLAUDE.md's 2026-10-29 "
        "write-up) -- triages every video face via its thumbnail, and for "
        "any that fail, searches real nearby frames for a confident "
        "detection to replace box/kps/det_score/thumbnail/embedding with. "
        "Anything with no confident match anywhere nearby is left "
        "untouched and reported, not guessed at."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None,
                             help='Only check the first N faces (for a quick spot-check).')
        parser.add_argument('--person-ids', type=str, default=None,
                             help='Comma-separated Person ids -- restrict to faces declared '
                                  'or proposed to these people.')
        parser.add_argument('--lvface-sizes', type=str, default='B',
                             help='Comma-separated LVFace sizes to (re)compute for a fixed '
                                  'face, using the frame/kps already decoded for the fix. '
                                  'Default is B only, matching the image-side gallery '
                                  'backfill (backfill_verified_kps) this is complementing -- '
                                  "no reason to pay for S/L on the video side when they "
                                  "aren't the size actually in use for comparisons. Empty "
                                  'string disables LVFace embedding entirely.')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        person_ids = None
        if options['person_ids']:
            person_ids = [int(s.strip()) for s in options['person_ids'].split(',') if s.strip()]
        lvface_sizes = [s.strip().upper() for s in options['lvface_sizes'].split(',') if s.strip()]

        qs = Face.objects.filter(source_video_file__isnull=False).exclude(face_thumbnail='') \
            .select_related('source_video_file')
        if person_ids is not None:
            qs = qs.filter(Q(declared_name_id__in=person_ids) | Q(poss_ident1_id__in=person_ids))
        qs = qs.order_by('id')
        if limit is not None:
            qs = qs[:limit]

        extractor = VideoFaceExtractor()
        det10g = extractor.det10g_model
        # Triage (one thumbnail, cheap either way) uses the heavier,
        # more accurate det10g -- but the window SEARCH can scan many
        # frames per face, especially ones that end up unresolved (no
        # early exit on failure), so it uses the lightweight det_model
        # instead, same "cheap model for broad search" split this
        # project already uses elsewhere (periodic sampling vs. the
        # few-frame det_10g refinement pass). A window match only needs
        # to clear MIN_SCORE, which det_model already supports reliably.
        det_lightweight = extractor.det_model

        total = 0
        skipped_ok = 0
        ok_lvface_embedded = 0
        ok_frame_check_failed = []
        fixed = 0
        lvface_embedded = 0
        unresolved = []
        decode_failed = []
        timed_out = []
        crashed = []
        t0 = time.time()
        ffprobe_cache = {}

        def get_ffprobe(filename):
            # Many faces share one VideoFile -- avoid re-shelling out to
            # ffprobe for every single one of them.
            if filename not in ffprobe_cache:
                ffprobe_cache[filename] = ffprobe_info(filename)
            return ffprobe_cache[filename]

        for face in qs.iterator(chunk_size=500):
            total += 1
            try:
                signal.signal(signal.SIGALRM, _raise_face_timeout)
                signal.alarm(PER_FACE_TIMEOUT_SECONDS)
                try:
                    try:
                        face.face_thumbnail.open('rb')
                        thumb_bytes = face.face_thumbnail.read()
                    finally:
                        try:
                            face.face_thumbnail.close()
                        except Exception:
                            pass

                    thumb = cv2.imdecode(np.frombuffer(thumb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if thumb is None:
                        decode_failed.append(face.id)
                        continue

                    h, w = thumb.shape[:2]
                    bboxes, _ = det10g.detect(thumb, max_num=0, metric='default')
                    confident = bboxes[:, 4] >= MIN_SCORE if bboxes.shape[0] else np.array([], dtype=bool)
                    ok = False
                    if confident.any():
                        cx_lo, cx_hi = w * (1 - CENTER_FRACTION) / 2, w * (1 + CENTER_FRACTION) / 2
                        cy_lo, cy_hi = h * (1 - CENTER_FRACTION) / 2, h * (1 + CENTER_FRACTION) / 2
                        for i in np.where(confident)[0]:
                            bcx = (bboxes[i, 0] + bboxes[i, 2]) / 2
                            bcy = (bboxes[i, 1] + bboxes[i, 3]) / 2
                            if cx_lo <= bcx <= cx_hi and cy_lo <= bcy <= cy_hi:
                                ok = True
                                break
                    video = face.source_video_file
                    old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

                    if ok:
                        skipped_ok += 1
                        if face.video_thumbnail_frame_seconds is None:
                            continue
                        try:
                            (width, height, avg_fps, probed_field_order, rotation,
                             color_transfer, sar_scale_width) = get_ffprobe(video.filename)
                        except Exception:
                            continue  # no second check/LVFace possible without ffprobe -- not a hard failure
                        field_order = video.field_order if video.field_order is not None else probed_field_order
                        try:
                            real_frame = extract_frame_near_timestamp(
                                video.filename, face.video_thumbnail_frame_seconds, avg_fps,
                                field_order, rotation, old_box, thumb,
                                color_transfer=color_transfer, sar_scale_width=sar_scale_width,
                            )
                        except Exception:
                            continue
                        if real_frame is None:
                            continue

                        redet = extractor._redetect_in_crop(real_frame, old_box)
                        if redet is None:
                            ok_frame_check_failed.append(face.id)
                            continue
                        confirmed_box, confirmed_kps, confirmed_score = redet

                        if lvface_sizes:
                            any_lvface = False
                            for size in lvface_sizes:
                                field = f'face_encoding_lvface_{size.lower()}'
                                if getattr(face, field) is not None:
                                    continue
                                try:
                                    emb = lvface.embed_from_image_array(
                                        real_frame, confirmed_kps, input_is_bgr=True, size=size,
                                    )
                                except Exception as e:
                                    self.stdout.write(f"  LVFace-{size} failed for face {face.id}: {e}")
                                    continue
                                if emb is None:
                                    continue
                                if not dry_run:
                                    setattr(face, field, emb.tolist())
                                any_lvface = True
                            if any_lvface:
                                ok_lvface_embedded += 1
                                if not dry_run:
                                    face.save()
                        continue

                    # Triage failed -- attempt a real fix.
                    if face.video_thumbnail_frame_seconds is None:
                        unresolved.append((face.id, 'no stored timestamp to search around'))
                        continue
                    try:
                        (width, height, avg_fps, probed_field_order, rotation,
                         color_transfer, sar_scale_width) = get_ffprobe(video.filename)
                    except Exception as e:
                        decode_failed.append((face.id, f'ffprobe failed: {e}'))
                        continue
                    field_order = video.field_order if video.field_order is not None else probed_field_order

                    try:
                        result = locate_confident_face_near_timestamp(
                            video.filename, face.video_thumbnail_frame_seconds, field_order, rotation,
                            old_box, det_lightweight, MIN_SCORE, window_seconds=SEARCH_WINDOW_SECONDS,
                            color_transfer=color_transfer, sar_scale_width=sar_scale_width,
                        )
                    except Exception as e:
                        decode_failed.append((face.id, f'window search failed: {e}'))
                        continue

                    if result is None:
                        unresolved.append((face.id, 'no confident detection anywhere in the search window'))
                        continue

                    frame, new_box, new_kps, new_det_score, new_ts = result
                    fixed += 1
                    if not dry_run:
                        extractor._set_face_box_and_thumbnail(
                            face, frame, new_box, new_kps, timestamp_seconds=new_ts, det_score=new_det_score,
                        )
                        new_embedding = extractor._encode_insightface(frame, new_kps)
                        face.face_encoding_512 = new_embedding.tolist()
                        # frame/new_kps are already decoded for the fix above --
                        # computing LVFace here too is ~free (1 more small ONNX
                        # call per size) vs. a second expensive decode later via
                        # a dedicated backfill (which doesn't exist for video
                        # faces anyway). Always overwrite: kps just changed, so
                        # any existing LVFace value is now stale.
                        any_lvface = False
                        for size in lvface_sizes:
                            field = f'face_encoding_lvface_{size.lower()}'
                            try:
                                emb = lvface.embed_from_image_array(
                                    frame, new_kps, input_is_bgr=True, size=size,
                                )
                            except Exception as e:
                                self.stdout.write(f"  LVFace-{size} failed for face {face.id}: {e}")
                                continue
                            if emb is None:
                                continue
                            setattr(face, field, emb.tolist())
                            any_lvface = True
                        if any_lvface:
                            lvface_embedded += 1
                        face.save()
                finally:
                    signal.alarm(0)
            except _FaceTimeout:
                timed_out.append(face.id)
                self.stdout.write(f"  TIMEOUT on face {face.id} -- skipping")
                continue
            except Exception as e:
                # Real, confirmed gap (2026-10-30): a missing thumbnail
                # file on disk for one face (FileNotFoundError) crashed
                # this ENTIRE multi-day run, since only _FaceTimeout was
                # caught at this level -- same "one bad item shouldn't
                # kill the whole batch" discipline this project already
                # enforces everywhere else (face_extraction, assign_
                # faces.py's per-face error handling, etc.), just missing
                # here until a real crash exposed it.
                crashed.append((face.id, f'{type(e).__name__}: {e}'))
                self.stdout.write(f"  CRASH on face {face.id}: {type(e).__name__}: {e} -- skipping")
                continue

            if total % 100 == 0:
                elapsed = time.time() - t0
                self.stdout.write(
                    f"  ...{total} checked ({total/elapsed:.2f}/s), {skipped_ok} already ok "
                    f"({ok_lvface_embedded} w/ LVFace, {len(ok_frame_check_failed)} frame-check-failed), "
                    f"{fixed} fixed ({lvface_embedded} w/ LVFace), {len(unresolved)} unresolved, "
                    f"{len(decode_failed)} decode-failed, {len(timed_out)} timed out, "
                    f"{len(crashed)} crashed"
                )

        elapsed = time.time() - t0
        self.stdout.write(
            f"\nChecked {total} video face(s) in {elapsed:.0f}s{' (dry run)' if dry_run else ''}.\n"
            f"  {skipped_ok} already fine ({ok_lvface_embedded} also got LVFace embeddings; "
            f"{len(ok_frame_check_failed)} disagreed on a second, real-frame check -- flagged below).\n"
            f"  {fixed} fixed via a confident nearby detection ({lvface_embedded} also got LVFace embeddings).\n"
            f"  {len(unresolved)} unresolved -- no confident detection anywhere nearby, left untouched.\n"
            f"  {len(decode_failed)} decode/ffprobe failures.\n"
            f"  {len(timed_out)} timed out (>{PER_FACE_TIMEOUT_SECONDS}s) -- left untouched, safe to retry later.\n"
            f"  {len(crashed)} crashed on an unexpected error -- left untouched, logged below, safe to retry later."
        )
        if crashed:
            self.stdout.write("\nCrashed face ids (first 50), flagged for later:")
            for fid, reason in crashed[:50]:
                self.stdout.write(f"  {fid}: {reason}")
        if ok_frame_check_failed:
            self.stdout.write(
                f"\nFaces that passed thumbnail triage but failed the real-frame second check "
                f"(first 50), flagged for later -- never auto-touched:"
            )
            for fid in ok_frame_check_failed[:50]:
                self.stdout.write(f"  {fid}")
        if unresolved:
            self.stdout.write("\nUnresolved face ids (first 50), flagged for later:")
            for fid, reason in unresolved[:50]:
                self.stdout.write(f"  {fid}: {reason}")
