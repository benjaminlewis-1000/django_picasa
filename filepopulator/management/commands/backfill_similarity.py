#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: compares every ImageFile
# not yet marked similarity_checked against the full hashed population,
# recording near-duplicate pairs. See filepopulator/similarity.py's
# run_similarity_check() for the actual logic, shared with the small
# recurring Celery task that picks up newly ingested images.

from django.core.management.base import BaseCommand

from filepopulator.similarity import run_similarity_check


class Command(BaseCommand):
    help = "Find near-duplicate images (via phash Hamming distance) not yet checked."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many images would be checked; don't write anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only check this many images (useful for a quick test run).",
        )

    def handle(self, *args, **options):
        run_similarity_check(limit=options['limit'], dry_run=options['dry_run'], log=self.stdout.write)
