from django.utils import timezone


def record_heartbeat_start(task_name):
    """Records that task_name just started its real work -- see
    common.models.TaskHeartbeat's own docstring. Call this at the very
    start of a recurring task's real work, before anything that could
    fail, and unconditionally (every invocation, not just successful
    ones) -- it must run regardless of what happens next, to prove the
    task actually started.

    task_name: a short, stable identifier distinguishing this specific
    recurring schedule entry, not just the underlying task function --
    e.g. 'face_manager.thistask.weekly_full' vs
    'face_manager.thistask.hourly', since both invoke the same
    thistask() function with different args and need to be watched for
    staleness/hangs independently.
    """
    from common.models import TaskHeartbeat
    TaskHeartbeat.objects.update_or_create(
        task_name=task_name, defaults={'last_started_at': timezone.now()},
    )


def record_heartbeat_done(task_name):
    """Records that task_name's most recent run actually finished --
    call this once the task's real work is over, in a `finally` (or
    equivalently at both the success point and in the except branch),
    so both a clean success and an already-logged crash are marked
    'finished, not stuck.' Only a hang (never reaching this call at
    all) leaves last_completed_at behind last_started_at, which is
    exactly what check_task_heartbeats watches for."""
    from common.models import TaskHeartbeat
    TaskHeartbeat.objects.filter(task_name=task_name).update(last_completed_at=timezone.now())
