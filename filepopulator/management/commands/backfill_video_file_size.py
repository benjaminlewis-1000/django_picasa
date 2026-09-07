import os

from django.core.management.base import BaseCommand

from filepopulator.models import VideoFile


class Command(BaseCommand):
    help = (
        "One-time backfill for VideoFile.file_size_bytes on rows created "
        "before the field existed (os.path.getsize() per file; safe to "
        "re-run, only touches rows still NULL)."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        qs = VideoFile.objects.filter(file_size_bytes__isnull=True)
        total = qs.count()
        updated = 0
        missing = 0
        for video in qs.iterator():
            try:
                size = os.path.getsize(video.filename)
            except OSError:
                missing += 1
                continue
            updated += 1
            if not dry_run:
                VideoFile.objects.filter(pk=video.pk).update(file_size_bytes=size)

        self.stdout.write(
            f"{'Would update' if dry_run else 'Updated'} {updated}/{total} rows "
            f"({missing} file(s) no longer on disk, left NULL)."
        )
