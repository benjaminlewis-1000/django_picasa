from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from common.advisory_lock import advisory_lock
from common.error_log import log_error
from common.models import TaskHeartbeat

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
    'face_manager.thistask.weekly_full': {
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
            log_error('common.tasks.check_task_heartbeats', message)
