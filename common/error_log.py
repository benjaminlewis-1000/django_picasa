# A hard, central cap on ApplicationError.message -- enforced here,
# not left to each individual call site to remember on its own. A
# per-call-site cap (e.g. truncating an id list to the first 50) is
# still worth doing for readability, but this is the real backstop: if
# some future (or existing, overlooked) caller builds an unbounded
# message -- a huge id list, a giant traceback dump -- it gets cut down
# to size here regardless, so one bad call site can never flood the
# log with a single giant row.
MAX_MESSAGE_LENGTH = 4000


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

    message is truncated to MAX_MESSAGE_LENGTH -- see that constant's
    own comment for why this is enforced here rather than trusted to
    every caller.
    """
    from common.models import ApplicationError
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH] + f"... [truncated, {len(message)} chars total]"
    ApplicationError.objects.create(source=source, message=message)
