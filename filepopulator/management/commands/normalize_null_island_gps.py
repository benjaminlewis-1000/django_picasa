#! /usr/bin/env python

# One-time data cleanup: ImageFile._init_image() used to fall back to
# gps_lat_decimal/gps_lon_decimal = 0 (instead of the -999 "no GPS"
# sentinel used everywhere else) whenever EXIF GPS decimal conversion
# produced NaN -- see filepopulator/models.py, now fixed. (0, 0) is a
# real, valid-looking coordinate (off the coast of Africa), so those rows
# look like "no GPS" data everywhere except a straight equality check
# against -999. This command finds any pre-existing rows still carrying
# that stale (0, 0) value and normalizes them to the real sentinel.

from django.core.management.base import BaseCommand

from filepopulator.models import ImageFile


class Command(BaseCommand):
    help = "Normalize ImageFile rows with GPS (0, 0) ('null island') to the -999 no-GPS sentinel."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many rows would be affected; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        qs = ImageFile.objects.filter(gps_lat_decimal=0, gps_lon_decimal=0)
        count = qs.count()
        self.stdout.write(f"ImageFile rows with GPS (0, 0): {count}")

        if count == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to normalize."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(f"Normalize {count} row(s) from (0, 0) to (-999, -999)? y/N: ")
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        updated = qs.update(gps_lat_decimal=-999, gps_lon_decimal=-999)
        self.stdout.write(self.style.SUCCESS(f"Normalized {updated} row(s)."))
