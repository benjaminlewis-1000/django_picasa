#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: for every ImageFile with no
# valid EXIF date (dateTakenValid=False), dateTaken currently holds
# whatever timezone.now() was at ingestion/reprocessing time -- a
# meaningless placeholder, not a real date. This tries
# ImageFile.guess_date_from_filename() against each one and replaces that
# placeholder with a real guess wherever the filename actually embeds a
# plausible date (see the function's own docstring for which naming
# conventions it recognizes). dateTakenValid is deliberately left False
# either way -- a filename guess is not EXIF-grade confidence.
#
# Uses .update() rather than instance.save() on purpose: ImageFile.save()
# unconditionally re-decodes the image to recompute its pixel hash (and
# now phash), which this backfill has no reason to pay for -- same
# rationale as backfill_phash.py.
#
# Safe to re-run: only touches rows still missing a valid EXIF date, and
# only overwrites their dateTaken if a filename guess is actually found.

from django.core.management.base import BaseCommand

from filepopulator.models import ImageFile, guess_date_from_filename


class Command(BaseCommand):
    help = "For images with no valid EXIF date, replace the now()-placeholder dateTaken with a filename-derived guess where possible."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many images would be updated; don't write anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many images (useful for a quick test run).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        qs = ImageFile.objects.filter(dateTakenValid=False).only('id', 'filename', 'dateTaken')
        if options['limit']:
            qs = qs[:options['limit']]

        total = 0
        updated = 0
        for obj in qs:
            total += 1
            guess = guess_date_from_filename(obj.filename)
            if guess is None:
                continue
            updated += 1
            if not dry_run:
                ImageFile.objects.filter(pk=obj.pk).update(dateTaken=guess)

        self.stdout.write(f"Checked {total} image(s) with no valid EXIF date.")
        if dry_run:
            self.stdout.write(self.style.WARNING(f"Dry run -- would update {updated}, wrote nothing."))
        else:
            self.stdout.write(self.style.SUCCESS(f"Updated dateTaken for {updated} image(s) from their filename."))
