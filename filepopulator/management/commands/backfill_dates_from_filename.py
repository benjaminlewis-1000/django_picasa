#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: for every ImageFile/VideoFile
# with no valid EXIF date (dateTakenValid=False), dateTaken currently holds
# whatever timezone.now() was at ingestion/reprocessing time -- a
# meaningless placeholder, not a real date. This tries
# guess_date_from_filename() against each one and replaces that placeholder
# with a real guess wherever the filename actually embeds a plausible date
# (see the function's own docstring for which naming conventions it
# recognizes). dateTakenValid is deliberately left False either way -- a
# filename guess is not EXIF-grade confidence, matching create_image_file()/
# create_video_file()'s own ingestion-time convention.
#
# Also updates dateTakenUTC to match -- a real gap fixed 2026-09-09: the
# original (image-only) version of this command updated dateTaken but never
# kept dateTakenUTC in sync, leaving it stale/wrong after a backfill.
#
# Uses .update() rather than instance.save() on purpose: ImageFile.save()
# unconditionally re-decodes the image to recompute its pixel hash (and
# phash), which this backfill has no reason to pay for -- same rationale as
# backfill_phash.py. VideoFile has no such re-decode cost on save(), but
# .update() is used there too for consistency/symmetry.
#
# Extended 2026-09-09 to also cover VideoFile, prompted by a real, sizable
# gap found while investigating one specific video with no date: 272 of 977
# (28%) videos with an invalid date match a newly-added M-D-YY date-only
# filename pattern (see guess_date_from_filename()) that this command can
# now recover.
#
# Safe to re-run: only touches rows still missing a valid EXIF date, and
# only overwrites their dateTaken/dateTakenUTC if a filename guess is
# actually found.

from django.core.management.base import BaseCommand

from filepopulator.models import ImageFile, VideoFile, guess_date_from_filename


class Command(BaseCommand):
    help = "For images/videos with no valid EXIF date, replace the now()-placeholder dateTaken/dateTakenUTC with a filename-derived guess where possible."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many rows would be updated; don't write anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many rows per model (useful for a quick test run).",
        )

    def _backfill_model(self, model, dry_run, limit):
        qs = model.objects.filter(dateTakenValid=False).only('id', 'filename', 'dateTaken')
        if limit:
            qs = qs[:limit]

        total = 0
        updated = 0
        for obj in qs:
            total += 1
            guess = guess_date_from_filename(obj.filename)
            if guess is None:
                continue
            updated += 1
            if not dry_run:
                model.objects.filter(pk=obj.pk).update(dateTaken=guess, dateTakenUTC=guess.timestamp())

        return total, updated

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']

        for model, label in ((ImageFile, 'image'), (VideoFile, 'video')):
            total, updated = self._backfill_model(model, dry_run, limit)
            self.stdout.write(f"Checked {total} {label}(s) with no valid EXIF date.")
            if dry_run:
                self.stdout.write(self.style.WARNING(f"Dry run -- would update {updated}, wrote nothing."))
            else:
                self.stdout.write(self.style.SUCCESS(f"Updated dateTaken/dateTakenUTC for {updated} {label}(s) from their filename."))
