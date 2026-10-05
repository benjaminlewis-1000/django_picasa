#! /usr/bin/env python

# Stage 1 (cheap, read-only) of a two-stage quality rescan for video-face
# thumbnails -- triggered by a real user report (2026-10-02): face
# 1104963's stored thumbnail/box/timestamp were all internally consistent
# (a fresh decode at the exact same frame/box reproduces the stored
# thumbnail almost exactly, diff~3) -- NOT a data-corruption bug -- but
# that frame is dark and motion-blurred, while a much clearer frame of
# the same real, confirmed person exists only ~0.8s later in the same
# detection track. The pipeline's own frame-selection (provisional =
# largest box; matched-person re-pick = highest gallery similarity,
# neither of which accounts for brightness/sharpness) can land on a
# genuinely worse moment than was available.
#
# This stage does NOT touch the source video at all -- it only reads each
# face's already-saved thumbnail JPEG (small, on local disk) and scores
# it on two standard, cheap metrics:
#   - brightness: mean grayscale pixel value (flags underexposed/dark
#     frames)
#   - sharpness: variance of the Laplacian (a standard blur detector --
#     a sharp image has a lot of high-frequency edge content, a blurry
#     one doesn't)
# Writes one row per face to a CSV-like output file so the real
# threshold can be picked by looking at the actual distribution (this
# project's own established discipline -- see CLAUDE.md's empirical-
# validation-before-deploying pattern) rather than guessed blind.
#
# Stage 2 (expensive -- full video redetection, only for whatever subset
# this stage's output says is worth a closer look) is a separate command.
import cv2
import numpy as np
from django.core.management.base import BaseCommand

from face_manager.models import Face


class Command(BaseCommand):
    help = (
        "Stage 1, read-only: score every video face's ALREADY-SAVED "
        "thumbnail on brightness/sharpness, no video decode needed. "
        "Writes results to --out for later threshold calibration. Does "
        "not modify any data."
    )

    def add_arguments(self, parser):
        parser.add_argument('--out', type=str, default='/tmp/video_thumbnail_quality.csv')
        parser.add_argument('--limit', type=int, default=None)

    def handle(self, *args, **options):
        out_path = options['out']
        limit = options['limit']

        qs = Face.objects.filter(source_video_file__isnull=False).exclude(face_thumbnail='')
        if limit is not None:
            qs = qs[:limit]

        total = qs.count() if limit is None else limit
        self.stdout.write(f'{total} video face(s) to score.')

        written = 0
        decode_failed = 0
        with open(out_path, 'w') as out:
            out.write('face_id,video_id,declared_name,brightness,sharpness\n')
            for i, face in enumerate(qs.iterator(), start=1):
                try:
                    face.face_thumbnail.open('rb')
                    data = face.face_thumbnail.read()
                    face.face_thumbnail.close()
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError('decode returned None')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness = float(np.mean(gray))
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                except Exception:
                    decode_failed += 1
                    continue

                name = face.declared_name.person_name if face.declared_name_id else ''
                out.write(f'{face.id},{face.source_video_file_id},{name},{brightness:.2f},{sharpness:.2f}\n')
                written += 1

                if i % 5000 == 0:
                    self.stdout.write(f'  ... {i}/{total} scored')

        self.stdout.write(
            f'\nDone. {written} face(s) scored, {decode_failed} skipped on decode failure. '
            f'Results written to {out_path}.'
        )
