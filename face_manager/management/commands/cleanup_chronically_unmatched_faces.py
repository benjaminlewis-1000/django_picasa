#! /usr/bin/env python

# One-time (but safely re-runnable) data cleanup: faces on orientation
# 6/8 images that are reencoded=True but still carry the
# NON_DETECTED_FACE_ENCODING sentinel are "chronically unmatched" -- they
# were never geometrically verified under the corrected decode path (see
# face_manager/face_extract_encode.py's update_list_of_no_matching_detects()
# fix) and, since a wrongly-positioned box can never match a real
# detection, would otherwise carry a stale/wrong box forever with no
# self-correcting mechanism. Deletes them outright and marks their images
# unprocessed so a fresh face_extraction pass can redetect any real face
# that's actually there.
#
# Safe to re-run: once the underlying images have been reprocessed, this
# query returns nothing.

from django.conf import settings
from django.core.management.base import BaseCommand

from face_manager.models import Face
from filepopulator.models import ImageFile


class Command(BaseCommand):
    help = "Delete chronically-unmatched, geometrically-unverified faces on orientation 6/8 images."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many faces/images would be affected; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        qs = Face.objects.filter(
            reencoded=True,
            source_image_file__orientation__in=[6, 8],
            face_encoding_512__0=settings.NON_DETECTED_FACE_ENCODING[0],
        )

        face_ids = list(qs.values_list('id', flat=True))
        image_ids = list(qs.values_list('source_image_file_id', flat=True).distinct())

        self.stdout.write(f"Chronically-unmatched faces: {len(face_ids)}")
        self.stdout.write(f"Images affected: {len(image_ids)}")

        if not face_ids:
            self.stdout.write(self.style.SUCCESS("Nothing to clean up."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(
                f"Delete {len(face_ids)} face(s) and mark {len(image_ids)} image(s) "
                "isProcessed=False for redetection? y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        deleted = 0
        for face_id in face_ids:
            try:
                Face.objects.get(pk=face_id).delete()
                deleted += 1
            except Face.DoesNotExist:
                pass

        updated = ImageFile.objects.filter(pk__in=image_ids).update(isProcessed=False)

        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} face(s)."))
        self.stdout.write(self.style.SUCCESS(f"Marked {updated} image(s) isProcessed=False for redetection."))
