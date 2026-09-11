import uuid

from django.contrib.auth.models import User
from django.db import models


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
