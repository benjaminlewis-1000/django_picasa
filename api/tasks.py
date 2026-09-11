from __future__ import absolute_import, unicode_literals

import os
import shutil
from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from api.models import UploadSession


@shared_task(ignore_result=True, name='api.cleanup_stale_uploads')
def cleanup_stale_uploads():
    """Deletes any UploadSession (api/upload_views.py's chunked-upload
    flow) older than settings.UPLOAD_SESSION_TTL_HOURS, regardless of
    status -- an abandoned in-progress session (closed tab, permanently
    dropped connection) would otherwise sit forever with its chunk
    scratch data never cleaned up, and a completed/failed session's own
    DB row has no further purpose once its chunks are already gone.
    Removes the chunk scratch directory defensively (ignore_errors) in
    case it's somehow still present -- the normal complete/fail path
    already cleans it up itself."""
    cutoff = timezone.now() - timedelta(hours=settings.UPLOAD_SESSION_TTL_HOURS)
    stale = UploadSession.objects.filter(created_at__lt=cutoff)
    count = 0
    for session in stale:
        shutil.rmtree(
            os.path.join(settings.UPLOAD_CHUNK_SCRATCH_DIR, str(session.upload_id)),
            ignore_errors=True,
        )
        count += 1
    stale.delete()
    settings.LOGGER.debug(f"cleanup_stale_uploads: removed {count} stale upload session(s).")
