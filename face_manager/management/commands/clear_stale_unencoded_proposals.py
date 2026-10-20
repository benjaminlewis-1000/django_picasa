#! /usr/bin/env python

# One-time (but safely re-runnable) catch-up: clear_encodings_below_floor
# only clears poss_identN/weight_N on a face it ITSELF nulls the encoding
# on -- a face already NULL-encoded before that fix existed is invisible
# to it. This sweeps the opposite condition directly: any never-
# adjudicated face missing an encoding but still carrying a stale
# poss_identN proposal, regardless of why the encoding went missing.
#
# PhotoCollection.save() now also runs this automatically (scoped to just
# the changed collection) via the face_manager.apply_collection_floor_
# change Celery task -- see clear_stale_proposals_with_no_encoding()'s
# own docstring in face_manager/models.py for the actual write logic,
# shared by both.

from django.core.management.base import BaseCommand

from face_manager.models import clear_stale_proposals_with_no_encoding


class Command(BaseCommand):
    help = (
        "Clear poss_identN/weight_N on never-adjudicated faces that are "
        "missing an encoding but still carry a stale proposal."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        count = clear_stale_proposals_with_no_encoding(dry_run=dry_run)
        self.stdout.write(
            f"{'Would clear' if dry_run else 'Cleared'} stale poss_identN/weight_N "
            f"on {count} unencoded face(s)."
        )
