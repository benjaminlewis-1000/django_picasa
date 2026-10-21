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

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"ApplicationError({self.source}, {self.created_at})"
