#! /usr/bin/env python

# One-time (but safely re-runnable) sync, needed alongside
# backfill_implausible_exif_dates.py: Face.dateTakenUTC is only ever
# copied from its source ImageFile.dateTakenUTC ONCE, at face-creation
# time (face_extract_encode.py's add_new_face()) -- it is never kept in
# sync afterward. Correcting an image's own dateTakenUTC (e.g. via
# backfill_implausible_exif_dates) does nothing for Face rows already
# extracted from it; they keep carrying the old (here, implausible)
# value forever unless explicitly synced.
#
# Scoped to exactly the faces that actually need it: dateTakenUTC still
# implying a pre-1990 date (the same implausibility cutoff
# backfill_implausible_exif_dates uses) AND currently different from
# their source image's own (now-corrected) value. Uses a single
# queryset .update() with a Subquery/OuterRef, not a per-row Python
# loop, since this is a pure bulk field sync with no business logic
# per row -- a direct F('source_image_file__dateTakenUTC') was tried
# first and rejected: Django's .update() hard-disallows joined-field
# F() expressions ("Joined field references are not permitted in this
# query"), confirmed via a real test failure, not assumed. A
# correlated Subquery is not a "join" in that same restricted sense
# and works correctly in .update().
#
# Safe to re-run: once synced, a face's dateTakenUTC either matches its
# image's real recovered date (>= 1990, dropping out of this command's
# own query) or matches the image's honest "unknown" placeholder (also
# dropping out, since that's also >= 1990 -- generated via timezone.now()).

from django.core.management.base import BaseCommand
from django.db.models import OuterRef, Subquery

from face_manager.models import Face
from filepopulator.models import ImageFile

_CUTOFF_UTC_TIMESTAMP = 631152000.0  # 1990-01-01 00:00:00 UTC


class Command(BaseCommand):
    help = (
        "Sync Face.dateTakenUTC from its source image's (corrected) "
        "dateTakenUTC, for faces still carrying an implausible pre-1990 "
        "value left stale by backfill_implausible_exif_dates."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many rows would be updated; don't write anything.",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        # Scoped purely to the face's OWN implausible value -- a face
        # still showing a pre-1990 dateTakenUTC always needs correcting
        # regardless of what its image's current value is (the image's
        # own fix, backfill_implausible_exif_dates, always leaves it
        # >= 1990 by construction, so this is already the right scope).
        qs = Face.objects.filter(
            dateTakenUTC__lt=_CUTOFF_UTC_TIMESTAMP,
            source_image_file__isnull=False,
        )

        count = qs.count()
        self.stdout.write(f"{count} face(s) with a stale/implausible dateTakenUTC, out of sync with their source image.")

        if dry_run:
            self.stdout.write(self.style.WARNING(f"Dry run -- would sync {count}. Wrote nothing."))
            return

        source_date = Subquery(
            ImageFile.objects.filter(pk=OuterRef('source_image_file_id')).values('dateTakenUTC')[:1]
        )
        updated = qs.update(dateTakenUTC=source_date)
        self.stdout.write(self.style.SUCCESS(f"Synced dateTakenUTC for {updated} face(s) from their source image."))
