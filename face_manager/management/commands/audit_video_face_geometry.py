#! /usr/bin/env python

# Read-only audit: for every video-sourced Face with a stored thumbnail,
# runs the same detector _redetect_in_crop() uses (det_10g) directly on
# the already-saved face_thumbnail FILE -- no video decode needed at all,
# since _square_thumbnail() already centers a correctly-placed box's face
# in the middle of the thumbnail with generous margin (see that function's
# own docstring: the face occupies roughly 50-100% of the thumbnail,
# never off to one side, when the box is actually right).
#
# Built 2026-10-29 after a real investigation (see CLAUDE.md) found a
# small number of video faces whose box/kps don't bound the actual face
# at all -- root-caused to _redetect_in_crop() silently accepting a
# low-confidence spurious detection when its padded search crop (built
# around a possibly-wrong upstream box) didn't contain the real face.
# That bug is now fixed going forward (REDETECT_MIN_SCORE fallback chain
# in video_face_pipeline.py); this command finds which ALREADY-EXTRACTED
# faces were affected, including ones just recovered by the kps restore
# (restoring a stale-but-internally-consistent box+kps pair doesn't fix a
# box that was already wrong at extraction time).
#
# A face "passes" if a confident (>=MIN_SCORE) detection exists in its
# thumbnail with its center landing in the thumbnail's central region --
# not simply "a face was found somewhere", since a badly-placed box's
# thumbnail can still show a DIFFERENT, unrelated face at the edge (e.g.
# another person partially in frame) without that meaning the box itself
# is correct.
#
# Does not modify any data -- a strict read-only scan. See CLAUDE.md for
# how flagged faces are remediated once this scan's real scope is known.

import time

import numpy as np
from django.core.management.base import BaseCommand
from insightface.app import FaceAnalysis

from face_manager.models import Face
from video_face_pipeline import decode_face_thumbnail

MIN_SCORE = 0.5
# Fraction of the thumbnail's own width/height the detected face's
# center must fall within, centered on the thumbnail's own center --
# 0.7 means the central 70% region, leaving a 15% margin on each side
# as the "edge" zone a badly-placed box's face would land in instead.
CENTER_FRACTION = 0.7


class Command(BaseCommand):
    help = (
        "Read-only audit: runs detection directly on every video-sourced "
        "Face's saved thumbnail file, flagging any without a confident, "
        "roughly-centered face -- the signature of a box/kps that doesn't "
        "actually bound the real face. No video decode, no data written."
    )

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=None,
                             help='Only check the first N faces (for a quick spot-check).')
        parser.add_argument('--person-ids', type=str, default=None,
                             help='Comma-separated Person ids -- restrict to faces declared '
                                  'or proposed to these people.')

    def handle(self, *args, **options):
        limit = options['limit']
        person_ids = None
        if options['person_ids']:
            person_ids = [int(s.strip()) for s in options['person_ids'].split(',') if s.strip()]

        qs = Face.objects.filter(source_video_file__isnull=False).exclude(face_thumbnail='')
        if person_ids is not None:
            from django.db.models import Q
            qs = qs.filter(Q(declared_name_id__in=person_ids) | Q(poss_ident1_id__in=person_ids))
        qs = qs.order_by('id')
        if limit is not None:
            qs = qs[:limit]

        face_analysis = FaceAnalysis(
            name='buffalo_l', providers=['CPUExecutionProvider'],
            allowed_modules=['detection'],
        )
        face_analysis.prepare(ctx_id=-1)
        det_model = face_analysis.models['detection']

        total = 0
        flagged = []
        t0 = time.time()

        for face in qs.iterator(chunk_size=500):
            total += 1
            img = decode_face_thumbnail(face)
            if img is None:
                flagged.append((face.id, 'thumbnail missing, unreadable, or undecodable'))
                continue

            h, w = img.shape[:2]
            bboxes, _ = det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                flagged.append((face.id, 'no detection in thumbnail at all'))
                continue

            confident = bboxes[:, 4] >= MIN_SCORE
            if not confident.any():
                best_score = float(bboxes[:, 4].max())
                flagged.append((face.id, f'no confident detection (best={best_score:.2f})'))
                continue

            cx_lo, cx_hi = w * (1 - CENTER_FRACTION) / 2, w * (1 + CENTER_FRACTION) / 2
            cy_lo, cy_hi = h * (1 - CENTER_FRACTION) / 2, h * (1 + CENTER_FRACTION) / 2
            centered_and_confident = False
            for i in np.where(confident)[0]:
                bcx = (bboxes[i, 0] + bboxes[i, 2]) / 2
                bcy = (bboxes[i, 1] + bboxes[i, 3]) / 2
                if cx_lo <= bcx <= cx_hi and cy_lo <= bcy <= cy_hi:
                    centered_and_confident = True
                    break
            if not centered_and_confident:
                flagged.append((face.id, 'confident detection found, but off-center'))

            if total % 1000 == 0:
                elapsed = time.time() - t0
                self.stdout.write(
                    f"  ...{total} checked ({total/elapsed:.1f}/s), {len(flagged)} flagged so far"
                )

        elapsed = time.time() - t0
        self.stdout.write(
            f"\nChecked {total} video face(s) in {elapsed:.0f}s. "
            f"{len(flagged)} flagged ({100*len(flagged)/total if total else 0:.2f}%)."
        )
        if flagged:
            self.stdout.write("\nFlagged face ids (first 50):")
            for fid, reason in flagged[:50]:
                self.stdout.write(f"  {fid}: {reason}")
