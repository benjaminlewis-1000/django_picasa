#! /usr/bin/env python

# Human-review surface for face_manager.rotation_audit's FLAGGED images --
# list proposed rotation corrections with a live-rotated preview URL
# (KeyedImageView's 'rotation_preview' type), and bulk confirm/dismiss them.
#
# Confirming is a cheap DB write ONLY -- the image already shows correctly
# everywhere the instant rotation_scan_state flips to CONFIRMED, via the
# shared decode chokepoint (ImageFile.confirmed_rotation_override_degrees,
# common.open_img_oriented). What still needs real (if now lightweight --
# no model inference) background work is the one-time Face box/kps
# geometry migration (face_manager.rotation_correction.
# run_rotation_correction()) -- dispatched here as an immediate on-demand
# Celery task (face_manager.correct_image_rotation_ids_task) for fast
# turnaround, with a separate, much-less-frequent scheduled sweep
# (face_manager.correct_image_rotation_task) as a safety net for whatever
# that dispatch ever misses. See CLAUDE.md's 2026-11-03/11-05/11-06
# orientation-detection write-up.

from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

import common
from api.mobile_views import _clamped_limit
from face_manager.tasks import correct_image_rotation_ids_task
from filepopulator.models import ImageFile


def _preview_url(host_url, image):
    return (
        f"{host_url}/keyed_image/rotation_preview/?id={image.id}"
        f"&access_key={settings.RANDOM_ACCESS_KEY}"
    )


class RotationReviewListView(APIView):
    """Lists ImageFiles currently FLAGGED for human rotation review.

    GET /api/rotation_review/?limit=&offset=
    Returns [{id, filename, proposed_rotation_degrees, face_count,
    preview_url}, ...], ordered by id for stable pagination (unlike the
    mobile `.ignore`-review grids, which deliberately randomize -- this
    is a bounded, finite backlog a reviewer is expected to work through
    to completion, not an endless stream to sample from).
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        limit = _clamped_limit(request, default=25, maximum=200)
        try:
            offset = max(0, int(request.query_params.get('offset', 0)))
        except (TypeError, ValueError):
            offset = 0

        host_url = f'https://{request.get_host()}/api'
        queryset = (
            ImageFile.objects
            .filter(rotation_scan_state=ImageFile.RotationScanState.FLAGGED)
            .order_by('id')
        )
        total = queryset.count()
        page = list(queryset[offset:offset + limit])

        results = [
            {
                'id': image.id,
                'filename': image.filename,
                'proposed_rotation_degrees': image.proposed_rotation_degrees,
                'face_count': image.face_set.count(),
                'preview_url': _preview_url(host_url, image),
            }
            for image in page
        ]
        return Response({'count': total, 'results': results})


class RotationReviewBulkActionView(APIView):
    """Bulk-resolve a batch of images -- not restricted to ones the audit
    actually FLAGGED: a human can confirm a rotation the audit never
    detected at all (no faces to measure from, a wrong/missed proposal),
    as long as it hasn't already been baked in (rotation_correction_
    applied=False) -- see rotation_signal_source.HUMAN.

      - `confirm`: [{"id": int, "degrees": float}, ...] -- `degrees` is
        the human's FINAL chosen correction (always explicit, even if it
        happens to match what the audit proposed -- there's no implicit
        "just accept the proposal" shorthand, so the frontend's rotate-
        left/right UI has one single code path regardless of whether the
        reviewer touched those buttons). Must be a real, nonzero multiple
        of 90 (90/180/270/-90/-180/-270/etc, within a small floating-point
        tolerance -- see common.is_cardinal_rotation) -- the only shape a
        rotate-left/right UI can ever actually produce; anything else
        (too close to "no rotation", or a non-cardinal angle) is rejected
        outright rather than silently snapped to an ambiguous nearest
        bucket -- use `dismiss_ids` for "no rotation needed". Normalized
        (common.snap_to_canonical_rotation) before being stored, so
        proposed_rotation_degrees always ends up exactly 90.0/-90.0/180.0
        for a human-confirmed row, never a near-multiple like 89.97.
        Overwrites proposed_rotation_degrees with that normalized value
        and sets
        rotation_scan_state -> CONFIRMED; an on-demand Celery task is
        then dispatched for exactly the ids that succeeded, to migrate
        their existing Faces' box/kps into the new coordinate space --
        not applied here, synchronously, inside this HTTP request.
      - `dismiss_ids`: rotation_scan_state -> DISMISSED_NOT_ROTATED,
        proposed_rotation_degrees cleared (the image isn't rotated at
        all, whether or not the audit ever proposed otherwise).

    Either action is rejected per-id (reported in `rejected`, never a
    silent no-op) once rotation_correction_applied=True for that image --
    re-correcting an already-corrected image isn't supported yet.
    """

    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):
        confirm_entries = request.data.get('confirm') or []
        dismiss_ids = request.data.get('dismiss_ids') or []
        ScanState = ImageFile.RotationScanState

        confirmed_ids = []
        rejected = []
        for entry in confirm_entries:
            try:
                image_id = int(entry['id'])
                degrees = float(entry['degrees'])
            except (KeyError, TypeError, ValueError):
                rejected.append({'id': entry.get('id') if isinstance(entry, dict) else None,
                                  'reason': 'missing or invalid id/degrees'})
                continue

            if not common.is_cardinal_rotation(degrees):
                rejected.append({
                    'id': image_id,
                    'reason': (
                        'degrees must be a nonzero multiple of 90 -- use dismiss_ids for '
                        '"no rotation needed"'
                    ),
                })
                continue

            updated = (
                ImageFile.objects
                .filter(id=image_id, rotation_correction_applied=False)
                .update(
                    rotation_scan_state=ScanState.CONFIRMED,
                    proposed_rotation_degrees=common.snap_to_canonical_rotation(degrees),
                    rotation_signal_source=ImageFile.RotationSignalSource.HUMAN,
                )
            )
            if updated:
                confirmed_ids.append(image_id)
            else:
                rejected.append({
                    'id': image_id,
                    'reason': 'no such image, or already corrected',
                })

        dismissed = (
            ImageFile.objects
            .filter(id__in=dismiss_ids, rotation_correction_applied=False)
            .update(rotation_scan_state=ScanState.DISMISSED_NOT_ROTATED, proposed_rotation_degrees=None)
        )

        if confirmed_ids:
            correct_image_rotation_ids_task.delay(confirmed_ids)

        return Response({'confirmed': len(confirmed_ids), 'dismissed': dismissed, 'rejected': rejected})
