#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: estimates GPS for Galaxy
# S23+ photos with no camera GPS at all, from OwnTracks location
# history (a separate service, see filepopulator/owntracks_gps.py for
# the real contract/caveats). Deliberately scoped to this one camera --
# see that module's own docstring for why applying it to every camera
# would be wrong, not just unwanted.

from django.core.management.base import BaseCommand, CommandError

from filepopulator.owntracks_gps import OwnTracksConfigError, backfill_s23_gps


class Command(BaseCommand):
    help = "Estimate GPS for Galaxy S23+ photos missing it, from OwnTracks location history."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print what would be matched; don't write or query-write anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many photos (useful for a quick test run).",
        )

    def handle(self, *args, **options):
        try:
            counts = backfill_s23_gps(
                dry_run=options['dry_run'], limit=options['limit'], log=self.stdout.write,
            )
        except OwnTracksConfigError as e:
            raise CommandError(str(e))

        self.stdout.write(
            f"{'Would match' if options['dry_run'] else 'Matched'} {counts['matched']} photo(s). "
            f"{counts['no_offset_tag']} skipped (no OffsetTimeOriginal tag), "
            f"{counts['no_nearby_point']} skipped (no nearby OwnTracks point)."
        )
