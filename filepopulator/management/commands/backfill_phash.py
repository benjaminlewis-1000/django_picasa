#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: computes phash for every
# ImageFile ingested before phash computation was added to
# ImageFile.save(). Every image saved from here on gets one
# automatically -- see filepopulator/similarity.py's
# run_phash_backfill() for the actual logic.

from django.core.management.base import BaseCommand

from filepopulator.similarity import run_phash_backfill


class Command(BaseCommand):
    help = "Compute phash for every ImageFile that doesn't have one yet."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many images would be processed; don't write or decode anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many images (useful for a quick test run).",
        )
        parser.add_argument(
            '--processes', type=int, default=1,
            help="Number of worker processes for the CPU-bound decode+hash step (default 1).",
        )

    def handle(self, *args, **options):
        run_phash_backfill(
            limit=options['limit'], dry_run=options['dry_run'],
            processes=options['processes'], log=self.stdout.write,
        )
