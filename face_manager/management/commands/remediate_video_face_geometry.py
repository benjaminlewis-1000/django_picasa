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
# centroid. A face still carrying face_encoding_lvface_b/s/l before this
# fix will naturally fall out of that population (since kps changed) and
# get picked up again by the existing, separate LVFace backfill --
# not re-encoded here, out of scope for a geometry fix.
#
# A face where NO confident detection turns up anywhere in the search
# window is left completely untouched and reported separately (flagged
# for later, per the user's explicit call) -- never deleted or guessed
# at.

import time

import cv2
import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Q

from face_manager.models import Face
from video_face_pipeline import VideoFaceExtractor, ffprobe_info, locate_confident_face_near_timestamp

MIN_SCORE = 0.5
CENTER_FRACTION = 0.7
SEARCH_WINDOW_SECONDS = 1.5


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

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        person_ids = None
        if options['person_ids']:
            person_ids = [int(s.strip()) for s in options['person_ids'].split(',') if s.strip()]

        qs = Face.objects.filter(source_video_file__isnull=False).exclude(face_thumbnail='') \
            .select_related('source_video_file')
        if person_ids is not None:
            qs = qs.filter(Q(declared_name_id__in=person_ids) | Q(poss_ident1_id__in=person_ids))
        qs = qs.order_by('id')
        if limit is not None:
            qs = qs[:limit]

        extractor = VideoFaceExtractor()
        det10g = extractor.det10g_model

        total = 0
        skipped_ok = 0
        fixed = 0
        unresolved = []
        decode_failed = []
        t0 = time.time()

        for face in qs.iterator(chunk_size=500):
            total += 1
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
            if ok:
                skipped_ok += 1
                continue

            # Triage failed -- attempt a real fix.
            video = face.source_video_file
            if face.video_thumbnail_frame_seconds is None:
                unresolved.append((face.id, 'no stored timestamp to search around'))
                continue
            try:
                width, height, avg_fps, probed_field_order, rotation, color_transfer, sar_scale_width = (
                    ffprobe_info(video.filename)
                )
            except Exception as e:
                decode_failed.append((face.id, f'ffprobe failed: {e}'))
                continue
            field_order = video.field_order if video.field_order is not None else probed_field_order
            old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

            try:
                result = locate_confident_face_near_timestamp(
                    video.filename, face.video_thumbnail_frame_seconds, field_order, rotation,
                    old_box, det10g, MIN_SCORE, window_seconds=SEARCH_WINDOW_SECONDS,
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
                face.save()

            if total % 100 == 0:
                elapsed = time.time() - t0
                self.stdout.write(
                    f"  ...{total} checked ({total/elapsed:.2f}/s), {skipped_ok} already ok, "
                    f"{fixed} fixed, {len(unresolved)} unresolved, {len(decode_failed)} decode-failed"
                )

        elapsed = time.time() - t0
        self.stdout.write(
            f"\nChecked {total} video face(s) in {elapsed:.0f}s{' (dry run)' if dry_run else ''}.\n"
            f"  {skipped_ok} already fine, no action needed.\n"
            f"  {fixed} fixed via a confident nearby detection.\n"
            f"  {len(unresolved)} unresolved -- no confident detection anywhere nearby, left untouched.\n"
            f"  {len(decode_failed)} decode/ffprobe failures."
        )
        if unresolved:
            self.stdout.write("\nUnresolved face ids (first 50), flagged for later:")
            for fid, reason in unresolved[:50]:
                self.stdout.write(f"  {fid}: {reason}")
