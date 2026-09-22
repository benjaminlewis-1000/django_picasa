import time

from django.core.management.base import BaseCommand
from django.db.models import Count

from filepopulator.models import VideoFile
from filepopulator.video_scripts import compute_video_content_hash

# Progress checkpoint cadence -- a full-library run is ~70 minutes
# (real, benchmarked throughput ~130-138 MB/s against a 542GB library,
# 2026-09-22), so a time-based checkpoint (not a fixed row count) keeps
# progress output meaningful regardless of whether the current file is
# a tiny clip or a multi-GB one.
_PROGRESS_INTERVAL_SECONDS = 30


class Command(BaseCommand):
    help = (
        "One-time backfill of VideoFile.content_hash (real MD5 of file "
        "bytes) for every row created before that field existed. Does NOT "
        "merge/delete anything itself -- once this has run, use "
        "merge_duplicate_videofiles to actually resolve any duplicate "
        "groups it reveals. Safe to re-run: only touches rows still NULL, "
        "and a file that's vanished from disk since ingestion is skipped "
        "(left NULL) rather than erroring the whole run."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument(
            '--limit', type=int, default=None,
            help='Restrict to the first N rows missing content_hash (for testing).',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']

        qs = VideoFile.objects.filter(content_hash__isnull=True).order_by('pk')
        total = qs.count()
        if limit is not None:
            qs = qs[:limit]

        self.stdout.write(
            f"{qs.count() if limit else total}/{total} rows missing content_hash "
            f"{'(dry run)' if dry_run else ''}"
        )

        hashed = 0
        missing = 0
        t0 = last_checkpoint = time.time()

        for video in qs.iterator():
            try:
                content_hash = compute_video_content_hash(video.filename)
            except OSError:
                missing += 1
                continue

            hashed += 1
            if not dry_run:
                VideoFile.objects.filter(pk=video.pk).update(content_hash=content_hash)

            now = time.time()
            if now - last_checkpoint >= _PROGRESS_INTERVAL_SECONDS:
                elapsed = now - t0
                self.stdout.write(
                    f"  ...{hashed + missing} processed (hashed={hashed} "
                    f"missing={missing}), {elapsed:.0f}s elapsed"
                )
                last_checkpoint = now

        elapsed = time.time() - t0
        self.stdout.write(
            f"DONE: {elapsed:.0f}s. hashed={hashed} missing_from_disk={missing}"
        )

        if not dry_run and hashed:
            dupe_groups = (
                VideoFile.objects.exclude(content_hash__isnull=True)
                .values('content_hash')
                .annotate(n=Count('id'))
                .filter(n__gt=1)
            )
            n_groups = dupe_groups.count()
            n_rows = sum(g['n'] for g in dupe_groups)
            self.stdout.write(
                f"Found {n_groups} duplicate content_hash group(s) covering {n_rows} "
                f"VideoFile row(s) total -- run merge_duplicate_videofiles to resolve them."
            )
