import uuid

from django.contrib.auth.models import User
from django.db import models, transaction
from django.db.models import Q
from django.utils import timezone


class UploadSession(models.Model):
    """Tracks an in-progress chunked upload (api/upload_views.py).

    Real DB rows, not in-memory state -- different chunks of the same
    upload can be handled by different gunicorn WORKER PROCESSES across
    separate HTTP requests (that's the whole point of chunking: no single
    request has to stay open for the length of the whole transfer), so
    the state has to live somewhere every worker can see.
    """
    STATUS_IN_PROGRESS = 'in_progress'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_IN_PROGRESS, 'In progress'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    upload_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False, db_index=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    filename = models.CharField(max_length=1024)
    total_size = models.BigIntegerField()
    # Whole-file SHA-256, hex -- declared at init time, re-verified once
    # all chunks are reassembled (api/upload_views.py's
    # CompleteChunkedUploadView). Per-chunk checksums (UploadChunk below)
    # catch corruption early and cheaply; this is the final, defense-in-
    # depth check against the actual assembled artifact.
    checksum = models.CharField(max_length=64)
    chunk_size = models.BigIntegerField()
    total_chunks = models.IntegerField()
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_IN_PROGRESS)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"UploadSession({self.upload_id}, {self.filename!r}, {self.status})"


class UploadChunk(models.Model):
    """One received chunk of an UploadSession. A separate row per chunk
    index -- rather than a shared list field on UploadSession -- so that
    concurrent chunk uploads (the frontend may send several chunks in
    parallel for speed) never race on a shared read-modify-write; each
    chunk index is its own independent insert/update."""
    session = models.ForeignKey(UploadSession, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.IntegerField()
    size = models.IntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['session', 'chunk_index'], name='unique_session_chunk_index'),
        ]

    def __str__(self):
        return f"UploadChunk(session={self.session_id}, index={self.chunk_index})"


class GooglePhotosCredential(models.Model):
    """Singleton (always pk=1, see load()) holding the OAuth client
    registered in Google Cloud Console plus the refresh token minted by
    the Tools tab's "Connect Google Photos" button
    (api/photos_watch_views.py's OAuth start/callback views). A DB row
    rather than settings.py/.env values, deliberately - lets these be
    entered and rotated entirely through the app's own GUI instead of
    editing server config and restarting a container."""
    client_id = models.CharField(max_length=255, blank=True, default='')
    client_secret = models.CharField(max_length=255, blank=True, default='')
    refresh_token = models.CharField(max_length=512, blank=True, default='')
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return f"GooglePhotosCredential(configured={bool(self.client_id)}, connected={bool(self.refresh_token)})"


class GooglePhotosWatchedAlbum(models.Model):
    """A user-named entry in the Tools-tab "Google Photos" list
    (api/photos_watch_views.py). Google's Picker API returns no album/
    sharer metadata at all (see CLAUDE.md) - `title` is purely what the
    user typed when adding it, not anything fetched from Google."""
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    last_synced_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"GooglePhotosWatchedAlbum({self.title!r})"


class SlideshowConfig(models.Model):
    """A named, reusable bundle of slideshow filter/display settings -
    the same knobs net_slideshow's App.jsx already accepts as URL
    params (see its CLAUDE.md), just saved server-side so the settings
    page can build/update them without anyone hand-editing a URL, and
    so ScheduleRule can reference them by name for time-based
    switching."""
    name = models.CharField(max_length=128, unique=True)
    # Comma-separated person names, matching the slideshow's own
    # people= filter format - stored as the same flat string rather than
    # an M2M to Person, since this is just forwarded straight through to
    # filteredImagesView, never queried against directly.
    people = models.CharField(max_length=512, blank=True, default='')
    year_start = models.IntegerField(null=True, blank=True)
    year_end = models.IntegerField(null=True, blank=True)
    # Mirrors the slideshow's `cronological` URL param name (including
    # its existing misspelling) so the active_config response can be
    # forwarded through unchanged.
    cronological = models.BooleanField(default=False)
    slide_len = models.PositiveIntegerField(default=6)
    show_meta = models.BooleanField(default=True)
    include_videos = models.BooleanField(default=True)
    include_images = models.BooleanField(default=True)
    debug_limit = models.PositiveIntegerField(null=True, blank=True)
    # The fallback config when no enabled ScheduleRule matches the
    # current day/time (or no rules exist at all yet). At most one row
    # may have this set - enforced at the DB level by the partial unique
    # constraint below, AND defensively in save() so flipping a second
    # row to default through the API demotes the previous one instead of
    # raising an IntegrityError.
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['is_default'], condition=Q(is_default=True),
                name='unique_default_slideshow_config',
            ),
        ]

    def save(self, *args, **kwargs):
        if self.is_default:
            with transaction.atomic():
                SlideshowConfig.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
                super().save(*args, **kwargs)
        else:
            super().save(*args, **kwargs)

    def __str__(self):
        return f"SlideshowConfig({self.name!r}, default={self.is_default})"


class ScheduleRule(models.Model):
    """Plays `config` whenever the current local time falls inside this
    rule's day-of-week + time-of-day window. Multiple enabled rules can
    match at once (e.g. overlapping windows); `priority` is the explicit,
    user-controlled tie-break (see resolve_active_slideshow_config)
    rather than leaving it to insertion order or rule id, which would be
    invisible and unintuitive to someone editing rules in the settings
    page."""
    WEEKDAY_ABBREVIATIONS = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

    config = models.ForeignKey(SlideshowConfig, on_delete=models.CASCADE, related_name='schedule_rules')
    name = models.CharField(max_length=128, blank=True, default='')
    # Comma-separated subset of WEEKDAY_ABBREVIATIONS - a single row can
    # cover several days ("weekday mornings") instead of needing one row
    # per day. Validated/normalized in api/serializers.py, not here -
    # this field intentionally has no model-level choices/validators so
    # a plain CharField read stays simple.
    days = models.CharField(max_length=27)
    start_time = models.TimeField()
    end_time = models.TimeField()
    priority = models.IntegerField(default=0)
    enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def day_list(self):
        return [d.strip().upper() for d in self.days.split(',') if d.strip()]

    def contains(self, local_dt):
        """Whether `local_dt` (an aware datetime already in local time -
        see resolve_active_slideshow_config) falls inside this rule's
        day+time window. end_time < start_time is treated as an
        overnight window (e.g. 22:00-06:00) that wraps past midnight,
        rather than an always-false empty range."""
        if self.WEEKDAY_ABBREVIATIONS[local_dt.weekday()] not in self.day_list():
            return False
        t = local_dt.time()
        if self.start_time <= self.end_time:
            return self.start_time <= t < self.end_time
        return t >= self.start_time or t < self.end_time

    def __str__(self):
        return f"ScheduleRule({self.name!r}, {self.days}, {self.start_time}-{self.end_time} -> {self.config_id})"


def resolve_active_slideshow_config(now=None):
    """What should be playing right now. Returns (config, rule) - rule
    is None when falling back to the default config (or when there's no
    config at all yet, in which case config is also None).

    Among every enabled rule whose window contains `now`, the highest
    `priority` wins; a tie is broken by the most-recently-created rule
    (higher id) - an explicit, documented tie-break rather than
    whatever order the DB happens to return rows in.
    """
    local_now = timezone.localtime(now or timezone.now())
    matching = [r for r in ScheduleRule.objects.filter(enabled=True).select_related('config')
                if r.contains(local_now)]
    if matching:
        best = max(matching, key=lambda r: (r.priority, r.id))
        return best.config, best
    return SlideshowConfig.objects.filter(is_default=True).first(), None


class GooglePhotosSyncedItem(models.Model):
    """One already-downloaded item for a GooglePhotosWatchedAlbum.
    google_media_item_id is Google's PickedMediaItem.id, documented as a
    persistent identifier stable across sessions - the whole dedup
    mechanism, since the Picker API itself has no "only what's new"
    concept and the user reselects an album's full contents every sync."""
    watched_album = models.ForeignKey(
        GooglePhotosWatchedAlbum, on_delete=models.CASCADE, related_name='synced_items')
    google_media_item_id = models.CharField(max_length=255, unique=True)
    filename = models.CharField(max_length=1024)
    downloaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"GooglePhotosSyncedItem({self.google_media_item_id}, {self.filename!r})"
