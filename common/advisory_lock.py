"""Postgres advisory locks -- a lightweight, crash-safe mutex for making
sure only one invocation of a given task runs at a time, regardless of
entry point (scheduled Celery task, manage.py shell, a management
command). Built to close a real race: face_manager.face_extraction's
find_and_encode_faces() had no locking of its own -- only the Celery task
wrapper checked celery_app.control.inspect().active(), a classic
check-then-act race (two invocations starting close together can each see
"0 others running" before either registers), and that check didn't apply
at all to a direct call bypassing the Celery task. Two concurrent runs
processing the same never-before-processed image each saw n_existing=0,
both ran detection (deterministic, so both got pixel-identical boxes),
and both inserted new Face rows -- confirmed against real production
data: ~72% of found same-image duplicate-Face-row pairs had adjacent or
near-adjacent ids, exactly the signature of two near-simultaneous inserts.

An advisory lock is tied to the database SESSION: if the holding process
crashes, is OOM-killed, or the container restarts, Postgres releases the
lock the moment that connection drops. No stale lock can be left behind,
so unlike a lockfile (settings.LOCKFILE) or a "mark done, unmark on
failure" per-row claim, there is no timeout/heartbeat/unclaim logic to
get wrong.

The lock key is scoped only to whatever name is passed in -- it does not
touch any table, row, or other Postgres locking machinery, and has no
effect on any other lock name. Give each task its own name; reusing a
name across unrelated tasks would serialize them against each other.
"""
from contextlib import contextmanager
import zlib

from django.db import connection


def _lock_key(name: str) -> int:
    # pg_try_advisory_lock takes a bigint; crc32 gives a stable 32-bit
    # unsigned value that comfortably fits, so distinct names alias only
    # in the extremely unlikely case of a full crc32 collision.
    return zlib.crc32(name.encode())


@contextmanager
def advisory_lock(name: str):
    """Non-blocking: yields True if this call acquired the named lock,
    False if another process already holds it (in which case the caller
    should skip its exclusive work, not wait). Usage:

        with advisory_lock('face_manager.face_extraction') as acquired:
            if not acquired:
                return
            ... do the exclusive work ...

    The lock is released automatically on exit from the `with` block
    (including via exception), and also automatically by Postgres itself
    if the underlying connection ever drops without a clean exit.
    """
    key = _lock_key(name)
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_try_advisory_lock(%s)", [key])
        acquired = cursor.fetchone()[0]
    try:
        yield acquired
    finally:
        if acquired:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", [key])
