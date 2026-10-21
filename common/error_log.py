def log_error(source, message):
    """Records a persistent, frontend-visible ApplicationError row.

    Use this for a small, curated set of things worth a human's
    attention via the frontend -- not every exception. Real logging
    (settings.LOGGER) stays the source of truth for full detail/
    tracebacks and should still be called alongside this, not replaced
    by it.

    source: a short, stable string identifying roughly where this came
    from (e.g. 'face_manager.assign_faces.execute'), not a full
    traceback.
    """
    from common.models import ApplicationError
    ApplicationError.objects.create(source=source, message=message)
