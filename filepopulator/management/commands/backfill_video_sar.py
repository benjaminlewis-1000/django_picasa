#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for VideoFile.sar_scale_width
# on rows ingested before that field existed. Cheap: one ffprobe call per
# video, same data create_video_file() now reads at ingestion time.
# Mirrors backfill_video_field_order.py/backfill_video_color_transfer.py.

from django.core.management.base import BaseCommand

from filepopulator.models import VideoFile
from filepopulator.video_scripts import _run_ffprobe


class Command(BaseCommand):
    help = "Backfill VideoFile.sar_scale_width for rows created before that field existed."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        qs = VideoFile.objects.filter(sar_scale_width__isnull=True)
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
            sar = video_stream.get('sample_aspect_ratio', '1:1')
            sar_num, _, sar_den = sar.partition(':')
            try:
                sar_ratio = float(sar_num) / float(sar_den) if sar_den and float(sar_den) != 0 else 1.0
            except ValueError:
                sar_ratio = 1.0
            if sar_ratio != 1.0:
                corrected = round(video.width * sar_ratio)
                sar_scale_width = corrected + (corrected % 2)
            else:
                sar_scale_width = 0
            updated += 1
            if not dry_run:
                VideoFile.objects.filter(pk=video.pk).update(sar_scale_width=sar_scale_width)

        self.stdout.write(
            f"{'Would update' if dry_run else 'Updated'} {updated}/{total} row(s) "
            f"({failed} ffprobe failure(s), left NULL)."
        )
