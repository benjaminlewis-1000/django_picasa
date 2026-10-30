from django.db import models


class ApplicationError(models.Model):
    """A persistent, dismissible log of noteworthy backend errors, meant
    to surface problems that would otherwise only be visible in server
    log files (/var/log/picasa/*) to someone who has to think to go
    check them. NOT a replacement for real logging -- settings.LOGGER
    stays the source of truth for full detail/tracebacks; this is a
    smaller, curated layer on top, written deliberately by specific call
    sites (see common.error_log.log_error()) when something is worth a
    human's attention, not every log line.
    """
    # A short, stable identifier for roughly where this came from (e.g.
    # 'face_manager.assign_faces.execute') -- not a full traceback (that
    # stays in the real logs), just enough to recognize/triage at a
    # glance without opening a log file.
    source = models.CharField(max_length=256)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    dismissed = models.BooleanField(default=False)
    dismissed_at = models.DateTimeField(null=True, blank=True)

    # Set only for errors raised by common.tasks.check_task_heartbeats
    # (2026-10-30) -- a structured link back to the specific
    # TaskHeartbeat.task_name this alert is about, so the frontend can
    # offer a "this wasn't actually stuck, clear it" action on dismiss
    # without parsing the free-text message. NULL for every other kind
    # of ApplicationError (most of them -- this is deliberately a niche
    # field, not something every error needs). See
    # ApplicationErrorViewSet.dismiss()'s own `clear_heartbeat` param.
    related_heartbeat_task_name = models.CharField(max_length=256, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"ApplicationError({self.source}, {self.created_at})"


class TaskHeartbeat(models.Model):
    """Durable "this scheduled task actually got invoked, and finished"
    record, for recurring tasks where SILENCE is itself a failure mode
    a try/except can't catch:
    - never invoked at all (Celery beat down, a bad schedule edit, the
      task deregistered) -- nothing ever reaches a try/except to log a
      crash, so a try/except-based error (common.error_log.
      ApplicationError) can only ever catch a task that DID start.
    - invoked, but stuck/hung (not actually raising -- an infinite
      loop, waiting forever on a lock/resource) -- again, nothing a
      try/except would ever see.

    task_name is a short, stable identifier distinguishing this
    specific recurring schedule entry, not just the underlying task
    function (see common.heartbeat.record_heartbeat_start()'s own
    docstring). last_started_at is updated unconditionally at the very
    start of the task's real work, before anything that could fail;
    last_completed_at is updated once that work actually finishes
    (success OR a caught exception -- either way, not stuck). A row
    where last_started_at is newer than last_completed_at means a run
    is (or was) genuinely still in progress. common.tasks.
    check_task_heartbeats (a separate, low-frequency watchdog task)
    uses both fields to tell "hasn't run recently enough" apart from
    "started but never finished," and records an ApplicationError for
    either."""
    task_name = models.CharField(max_length=256, unique=True)
    last_started_at = models.DateTimeField()
    last_completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"TaskHeartbeat({self.task_name}, started={self.last_started_at})"
