#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: reverse-geocodes every
# distinct GPS coordinate in the library that isn't already cached. See
# filepopulator/geocode.py's run_geocoding_backfill() for the actual
# logic, shared with the small recurring Celery task that picks up newly
# ingested images.

from django.core.management.base import BaseCommand

from filepopulator.geocode import run_geocoding_backfill


class Command(BaseCommand):
    help = "Reverse-geocode every distinct GPS coordinate in the library not already cached."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many coordinates would be processed; don't write or query Nominatim.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many coordinates (useful for a quick test run).",
        )

    def handle(self, *args, **options):
        run_geocoding_backfill(limit=options['limit'], dry_run=options['dry_run'], log=self.stdout.write)
