#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for the real bug fixed
# 2026-10-28: ImageFile._get_date_taken() used to accept ANY EXIF
# DateTime* field that parsed successfully, including implausible
# pre-1990 dates -- confirmed against 2,376 real images (1,815 from one
# "Mom Phone") all carrying the exact same "1900-01-01 01:01:01" EXIF
# value, a well-known device-clock-reset default, not a real capture
# date. Those rows are currently stuck with dateTakenValid=True and an
# implausible dateTaken/dateTakenUTC -- the live ingestion-time fix only
# protects NEWLY-ingested/reprocessed files going forward, it does
# nothing for rows already written with the bad value.
#
# This targets exactly that population (dateTakenValid=True AND
# dateTakenUTC implying a date before 1990) and, for each:
#   - tries guess_date_from_filename() first (a real recovered date is
#     strictly better than an admittedly-unknown one) -- if found, sets
#     dateTaken/dateTakenUTC to the real recovered date. dateTakenValid
#     stays False either way, matching create_image_file()'s own
#     ingestion-time convention that a filename guess is never
#     EXIF-grade confidence.
#   - if no filename guess is found, resets dateTaken/dateTakenUTC to
#     "now" (the same neutral placeholder _get_date_taken() itself
#     would produce for a file with no real date at all) rather than
#     leaving the misleading implausible date sitting there with just
#     the valid flag flipped -- anything reading dateTakenUTC
#     numerically without first checking dateTakenValid would otherwise
#     still see a wrong, implausible value.
#
# Uses .update() rather than instance.save(), same rationale as
# backfill_dates_from_filename.py: no reason to pay for ImageFile.save()'s
# unconditional pixel-hash re-decode for a pure date correction.
#
# Safe to re-run: only touches rows still matching the implausible-date
# condition: once corrected, a row either has a real recovered date
# (dateTakenUTC >= 1990) or has been reset to "now" with
# dateTakenValid=False, dropping it out of this command's own query.

from django.core.management.base import BaseCommand
from django.utils import timezone

from filepopulator.models import ImageFile, guess_date_from_filename

# 1990-01-01 00:00:00 UTC, matching _get_date_taken()'s own implausibility
# cutoff -- digital EXIF didn't exist before the mid-1990s, so anything
# earlier is never a real capture date.
_CUTOFF_UTC_TIMESTAMP = 631152000.0


class Command(BaseCommand):
    help = (
        "For images with an implausible pre-1990 EXIF date that was "
        "wrongly trusted (dateTakenValid=True), try to recover a real "
        "date from the filename, or reset to an honest 'unknown' "
        "placeholder (now(), dateTakenValid=False) if no better guess "
        "is available."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many rows would be updated; don't write anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many rows (useful for a quick test run).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']

        qs = ImageFile.objects.filter(
            dateTakenValid=True, dateTakenUTC__lt=_CUTOFF_UTC_TIMESTAMP,
        ).only('id', 'filename')
        if limit:
            qs = qs[:limit]

        total = 0
        recovered = 0
        reset_unknown = 0
        for obj in qs:
            total += 1
            guess = guess_date_from_filename(obj.filename)
            if guess is not None:
                recovered += 1
                if not dry_run:
                    ImageFile.objects.filter(pk=obj.pk).update(
                        dateTaken=guess, dateTakenUTC=guess.timestamp(), dateTakenValid=False,
                    )
            else:
                reset_unknown += 1
                if not dry_run:
                    now = timezone.now()
                    ImageFile.objects.filter(pk=obj.pk).update(
                        dateTaken=now, dateTakenUTC=now.timestamp(), dateTakenValid=False,
                    )

        self.stdout.write(f"Checked {total} image(s) with an implausible pre-1990 EXIF date.")
        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"Dry run -- would recover a real date for {recovered}, "
                f"reset {reset_unknown} to an honest 'unknown' placeholder. Wrote nothing."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Recovered a real filename-based date for {recovered} image(s); "
                f"reset {reset_unknown} to an honest 'unknown' placeholder (no filename guess available)."
            ))
