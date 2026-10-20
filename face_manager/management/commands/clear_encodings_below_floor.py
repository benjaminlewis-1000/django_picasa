#! /usr/bin/env python

# One-time (but safely re-runnable) sweep: an existing Face already carries
# a real face_encoding_512, computed before its collection's
# min_face_det_score/min_face_relative_size floor existed, or before the
# floor was tightened. below_collection_floor() already gates NEW faces
# (add_new_face()) and re-encodes (reencode_missing_faces()) live -- but
# neither of those ever touches a face that already has a usable embedding,
# so retuning a floor down (e.g. setting the default collection to 2%) has
# no effect on the existing backlog until this sweep runs.
#
# This command is the manual/one-off entry point over the WHOLE library;
# PhotoCollection.save() now also triggers this automatically (scoped to
# just the changed collection) via the face_manager.apply_collection_
# floor_change Celery task -- see clear_encodings_below_collection_floor()'s
# own docstring in face_manager/models.py for the actual write logic,
# shared by both.

from django.core.management.base import BaseCommand

from face_manager.models import clear_encodings_below_collection_floor


class Command(BaseCommand):
    help = (
        "Clear face_encoding_512 for never-adjudicated faces that are below "
        "their collection's current min_face_det_score/min_face_relative_size floor."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        scanned, cleared = clear_encodings_below_collection_floor(dry_run=dry_run)
        self.stdout.write(
            f"{'Would clear' if dry_run else 'Cleared'} face_encoding_512 on "
            f"{cleared}/{scanned} never-adjudicated face(s) below their collection's floor."
        )
