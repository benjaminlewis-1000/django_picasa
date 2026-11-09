from datetime import timedelta

from celery import shared_task
from celery.exceptions import WorkerShutdown
from django.conf import settings
from django.utils import timezone

from common.advisory_lock import advisory_lock
from common.error_log import log_error
from common.models import TaskHeartbeat
from common.schema_health import find_stale_models

# Registry of recurring tasks worth watching for silence (see
# TaskHeartbeat's own docstring for why this is a different failure mode
# than a try/except-catchable crash). Each task gets two independent
# thresholds:
# - max_age_since_start: how long since the LAST invocation is still
#   normal -- generously above the real schedule interval (enough
#   buffer to absorb ordinary jitter/a brief outage without false-
#   alarming), catching "this hasn't even started in a long time."
# - max_runtime: how long a single run should reasonably take before
#   it's more likely stuck than still legitimately working -- catches
#   "this started but never finished" (a hang -- an error that DOES
#   raise is already caught immediately by common.error_log.log_error()
#   at its own call site, not by this watchdog).
HEARTBEAT_REGISTRY = {
    # Weekly (Monday 1:04am). 8 days covers one missed week's slot
    # before alarming. 24h runtime budget: a full redo_all=True pass
    # reclassifies the entire unassigned population (potentially
    # 100k+ faces), far more than the hourly incremental batch -- sized
    # generously above that, not measured against a real completed run
    # yet.
    'face_manager.assign_faces.weekly_full': {
        'max_age_since_start': timedelta(days=8),
        'max_runtime': timedelta(hours=24),
    },
}


@shared_task(ignore_result=True, name='common.check_task_heartbeats')
def check_task_heartbeats():
    with advisory_lock('common.check_task_heartbeats') as acquired:
        if not acquired:
            settings.LOGGER.debug("check_task_heartbeats already running, exiting.")
            return

        now = timezone.now()
        for task_name, cfg in HEARTBEAT_REGISTRY.items():
            heartbeat = TaskHeartbeat.objects.filter(task_name=task_name).first()

            if heartbeat is None:
                message = f"{task_name} has never recorded a heartbeat at all."
            elif now - heartbeat.last_started_at > cfg['max_age_since_start']:
                message = (
                    f"{task_name} hasn't started a new run since "
                    f"{heartbeat.last_started_at.isoformat()}, more than "
                    f"{cfg['max_age_since_start']} ago."
                )
            elif (
                (heartbeat.last_completed_at is None
                 or heartbeat.last_completed_at < heartbeat.last_started_at)
                and now - heartbeat.last_started_at > cfg['max_runtime']
            ):
                message = (
                    f"{task_name} started at {heartbeat.last_started_at.isoformat()} and still "
                    f"hasn't finished, more than {cfg['max_runtime']} later -- possibly stuck."
                )
            else:
                continue

            settings.LOGGER.error(f"check_task_heartbeats: {message}")
            log_error('common.tasks.check_task_heartbeats', message, related_heartbeat_task_name=task_name)


@shared_task(ignore_result=True, name='common.check_schema_freshness')
def check_schema_freshness():
    """Catches a real, confirmed failure mode (2026-11-09, see
    CLAUDE.md): a long-running celery worker process keeps its Django
    models loaded in memory from whenever it started -- if a migration
    adds a new NOT-NULL column while it's still running, every write to
    that table silently fails (caught by whichever per-row try/except
    happens to wrap it, logged, and moved past) until a human notices
    and manually restarts the worker, which could take hours. This runs
    independently of any specific error message or call site -- see
    common/schema_health.py's own docstring for why a periodic
    structural self-check was chosen over chasing every existing (and
    future) swallowed-exception site individually.

    Raises WorkerShutdown, not os._exit() -- deliberate. Celery's
    prefork pool executes tasks in child processes forked from the
    master, which inherit the master's already-imported (stale) Python
    modules -- os._exit() here would only kill the one pool child
    running this task, and the master would just fork a REPLACEMENT
    child from its own still-stale state, silently looping forever
    without ever picking up fresh code. WorkerShutdown is celery's own
    public API for "the whole worker (master + pool) needs to come
    down," which is what actually lets the supervising process (see
    dockerize/supervise_celery_worker.py) relaunch a genuinely fresh
    process that re-imports everything from disk.
    """
    stale = find_stale_models()
    if not stale:
        return

    message = "Stale model(s) detected -- the live DB has columns this process's code doesn't know about: " + "; ".join(
        f"{model} missing {sorted(cols)}" for model, cols in sorted(stale.items())
    )
    settings.LOGGER.error(f"check_schema_freshness: {message}")
    log_error('common.tasks.check_schema_freshness', message)
    raise WorkerShutdown()
