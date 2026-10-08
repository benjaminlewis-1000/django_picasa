import os
import time

from django.conf import settings
from django.core.management.base import BaseCommand

from filepopulator.models import ImageFile, VideoFile
from face_manager.models import Face, Person


def _referenced_paths(queryset, field_name):
    """Absolute on-disk paths every current row's field actually points at
    -- built from a plain values_list (just the stored name string), not
    by instantiating full model objects, since this needs to scale to
    hundreds of thousands of rows."""
    paths = set()
    rel_names = (
        queryset.exclude(**{field_name: ''}).exclude(**{field_name: None})
        .values_list(field_name, flat=True)
    )
    for rel_name in rel_names:
        if rel_name:
            paths.add(os.path.join(settings.MEDIA_ROOT, rel_name))
    return paths


class Command(BaseCommand):
    help = (
        "Deletes orphaned thumbnail/highlight files -- ones sitting on "
        "disk that no current row references anymore. Real, substantial "
        "backlog as of 2026-10-08: thumbnails_big/thumbnails_med are now "
        "entirely dead (their fields were removed outright -- every file "
        "left in them is an orphan by definition); thumbnails_small, "
        "face_thumbnails, and face_highlights each accumulate real "
        "orphans too (re-processing/restore incidents leaving behind a "
        "file whose old filename a row no longer points to). A file "
        "newer than --min-age-hours is never touched, regardless of "
        "whether it looks orphaned -- a brand-new thumbnail is written to "
        "disk moments before its owning row's save() actually commits, "
        "so a file that looks unreferenced in a query that ran a moment "
        "ago could just be mid-flight, not a real orphan."
    )

    # (directory name, model queryset + field to check against, or None
    # for a directory whose field no longer exists at all -- every file
    # in it is unconditionally orphaned).
    GROUPS = [
        ('thumbnails_big', None),
        ('thumbnails_med', None),
        ('thumbnails_small', [(ImageFile, 'thumbnail_small'), (VideoFile, 'thumbnail_small')]),
        ('face_thumbnails', [(Face, 'face_thumbnail')]),
        ('face_highlights', [(Person, 'highlight_img')]),
    ]

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument(
            '--min-age-hours', type=float, default=24.0,
            help='Never touch a file modified more recently than this (default 24h) -- '
                 'guards against the race between a file landing on disk and its row committing.',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        min_age_seconds = options['min_age_hours'] * 3600
        now = time.time()

        grand_deleted = 0
        grand_bytes = 0

        for dirname, field_specs in self.GROUPS:
            root = os.path.join(settings.MEDIA_ROOT, dirname)
            if not os.path.isdir(root):
                self.stdout.write(f'{dirname}: directory does not exist, skipping.')
                continue

            referenced = set()
            if field_specs is not None:
                for model, field_name in field_specs:
                    referenced |= _referenced_paths(model.objects.all(), field_name)

            deleted = 0
            skipped_too_new = 0
            freed_bytes = 0
            for dirpath, _dirs, filenames in os.walk(root):
                for fname in filenames:
                    full_path = os.path.join(dirpath, fname)
                    if full_path in referenced:
                        continue
                    try:
                        stat_result = os.stat(full_path)
                    except FileNotFoundError:
                        continue
                    if now - stat_result.st_mtime < min_age_seconds:
                        skipped_too_new += 1
                        continue
                    freed_bytes += stat_result.st_size
                    deleted += 1
                    if not dry_run:
                        try:
                            os.remove(full_path)
                        except FileNotFoundError:
                            pass

            grand_deleted += deleted
            grand_bytes += freed_bytes
            self.stdout.write(
                f"{dirname}: {'would delete' if dry_run else 'deleted'} {deleted} orphaned "
                f"file(s) ({freed_bytes / (1024 ** 3):.2f} GB), {skipped_too_new} skipped as too new."
            )

        self.stdout.write(
            f"TOTAL: {'would free' if dry_run else 'freed'} {grand_bytes / (1024 ** 3):.2f} GB "
            f"across {grand_deleted} file(s), dry_run={dry_run}."
        )
