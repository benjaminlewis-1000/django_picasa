#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for VideoFile.field_order on
# rows ingested before that field existed. Cheap: one ffprobe call per
# video, same data create_video_file() now reads at ingestion time.

from django.core.management.base import BaseCommand

from filepopulator.models import VideoFile
from filepopulator.video_scripts import _run_ffprobe


class Command(BaseCommand):
    help = "Backfill VideoFile.field_order for rows created before that field existed."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        qs = VideoFile.objects.filter(field_order__isnull=True)
        total = 0
        updated = 0
        failed = 0
        for video in qs.iterator():
            total += 1
            probe = _run_ffprobe(video.filename)
            if probe is None:
                failed += 1
                continue
            video_stream = next(
                (s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), None
            )
            if video_stream is None:
                failed += 1
                continue
            field_order = video_stream.get('field_order', 'unknown')
            updated += 1
            if not dry_run:
                VideoFile.objects.filter(pk=video.pk).update(field_order=field_order)

        self.stdout.write(
            f"{'Would update' if dry_run else 'Updated'} {updated}/{total} row(s) "
            f"({failed} ffprobe failure(s), left NULL)."
        )
