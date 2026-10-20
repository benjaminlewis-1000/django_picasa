#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for Face.relative_size on
# rows created before that field existed (every row at the time this
# was added, 2026-10-20). Pure arithmetic from already-stored columns
# (box_*, and the source image's width/height) -- no decode, no model
# inference, no I/O beyond the DB, so this is cheap even at real scale
# (~680k Face rows). bulk_update() in batches, not Django's .update():
# the computation needs min(box_w, box_h) / min(image_w, image_h),
# and Django's .update() does not support F() expressions that span a
# join (confirmed directly -- raises "Joined field references are not
# permitted in this query") -- image width/height live on a related
# ImageFile row, so the division has to happen in Python.

from django.core.management.base import BaseCommand

from face_manager.models import Face

BATCH_SIZE = 5000


class Command(BaseCommand):
    help = "Backfill Face.relative_size for rows created before that field existed."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        rows = Face.objects.filter(
            relative_size__isnull=True,
            source_image_file__isnull=False,
            box_left__isnull=False, box_top__isnull=False,
            box_right__isnull=False, box_bottom__isnull=False,
        ).values(
            'id', 'box_left', 'box_top', 'box_right', 'box_bottom',
            'source_image_file__width', 'source_image_file__height',
        )

        to_update = []
        skipped = 0
        total = 0
        for row in rows.iterator(chunk_size=BATCH_SIZE):
            total += 1
            img_w, img_h = row['source_image_file__width'], row['source_image_file__height']
            image_min_dim = min(img_w or 0, img_h or 0)
            if image_min_dim <= 0:
                skipped += 1
                continue
            box_w = row['box_right'] - row['box_left']
            box_h = row['box_bottom'] - row['box_top']
            relative_size = min(box_w, box_h) / image_min_dim
            to_update.append(Face(pk=row['id'], relative_size=relative_size))

            if not dry_run and len(to_update) >= BATCH_SIZE:
                Face.objects.bulk_update(to_update, ['relative_size'], batch_size=BATCH_SIZE)
                to_update = []

        if not dry_run and to_update:
            Face.objects.bulk_update(to_update, ['relative_size'], batch_size=BATCH_SIZE)

        updated = total - skipped
        self.stdout.write(
            f"{'Would update' if dry_run else 'Updated'} {updated}/{total} row(s) "
            f"({skipped} skipped -- image width/height unavailable)."
        )
