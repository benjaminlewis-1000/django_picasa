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
    """Bulk-resolve a batch of FLAGGED images.

      - `confirm_ids`: rotation_scan_state -> CONFIRMED (the image shows
        correctly everywhere immediately, see this module's own
        docstring), then an on-demand Celery task is dispatched for
        exactly these ids to migrate their existing Faces' box/kps into
        the new coordinate space -- not applied here, synchronously,
        inside an HTTP request.
      - `dismiss_ids`: rotation_scan_state -> DISMISSED_NOT_ROTATED (the
        image isn't actually rotated; the audit's proposal was wrong).

    Only rows currently FLAGGED are touched -- an id that's already been
    resolved (by this call or a concurrent one) or was never flagged is
    silently skipped, not errored, matching BulkConfirmIgnore's convention.
    """

    permission_classes = (IsAuthenticated,)

    def patch(self, request, *args, **kwargs):
        confirm_ids = request.data.get('confirm_ids') or []
        dismiss_ids = request.data.get('dismiss_ids') or []
        ScanState = ImageFile.RotationScanState

        confirmed = (
            ImageFile.objects
            .filter(id__in=confirm_ids, rotation_scan_state=ScanState.FLAGGED)
            .update(rotation_scan_state=ScanState.CONFIRMED)
        )
        dismissed = (
            ImageFile.objects
            .filter(id__in=dismiss_ids, rotation_scan_state=ScanState.FLAGGED)
            .update(rotation_scan_state=ScanState.DISMISSED_NOT_ROTATED)
        )

        # Passing the raw confirm_ids (not a re-queried "which ones
        # actually flipped" list) is safe and deliberate -- run_rotation_
        # correction()'s own queryset re-filters to CONFIRMED-and-not-
        # yet-applied, so an id that didn't actually change state here
        # (already resolved, never flagged) just harmlessly doesn't match
        # and costs nothing.
        if confirm_ids:
            correct_image_rotation_ids_task.delay(confirm_ids)

        return Response({'confirmed': confirmed, 'dismissed': dismissed})
