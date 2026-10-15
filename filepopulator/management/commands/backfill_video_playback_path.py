#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for VideoFile.playback_path
# on rows ingested before that field existed (every row at the time this
# was added, 2026-10-15 -- see Phase 4's "slideshow serving" write-up in
# CLAUDE.md). Pure string computation, no I/O beyond the DB -- cheap
# enough to run against the whole library in one pass, unlike the
# ffprobe/exiftool-based video backfills elsewhere in this directory.

from django.core.management.base import BaseCommand

from filepopulator.models import VideoFile
from filepopulator.video_scripts import compute_video_playback_path


class Command(BaseCommand):
    help = "Backfill VideoFile.playback_path for rows created before that field existed."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        qs = VideoFile.objects.filter(playback_path__isnull=True)
        total = 0
        updated = 0
        for video in qs.iterator():
            total += 1
            playback_path = compute_video_playback_path(video.filename)
            updated += 1
            if not dry_run:
                VideoFile.objects.filter(pk=video.pk).update(playback_path=playback_path)

        self.stdout.write(
            f"{'Would update' if dry_run else 'Updated'} {updated}/{total} row(s)."
        )
