# One-off/manual entry point over the whole library -- see
# filepopulator/cross_collection_dedup.py's own module docstring for the
# full design and validation this is based on. The recurring Celery task
# (filepopulator.merge_cross_collection_image_duplicates, same underlying
# find_unresolved_cross_collection_pairs()/resolve_pair()) covers this
# going forward as new collections/photos are added; this command is for
# the existing backlog (e.g. a newly-added collection's accumulated
# duplicate pairs) and for re-running by hand.
from django.core.management.base import BaseCommand

from filepopulator.cross_collection_dedup import find_unresolved_cross_collection_pairs, resolve_pair
from filepopulator.models import CrossCollectionImageDuplicate


class Command(BaseCommand):
    help = (
        "Resolve cross-collection exact-phash duplicate ImageFile pairs: "
        "auto-merge their Face rows where safe, flag the rest for manual review."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument(
            '--limit', type=int, default=None,
            help='Restrict to the first N unresolved pairs (for testing).',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']

        totals = {
            CrossCollectionImageDuplicate.STATUS_AUTO_MERGED: 0,
            CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW: 0,
        }
        faces_matched = faces_transferred = faces_deleted = 0
        n = 0

        for pair in find_unresolved_cross_collection_pairs():
            if limit is not None and n >= limit:
                break
            n += 1
            stats = resolve_pair(pair, dry_run=dry_run)
            totals[stats['status']] += 1
            faces_matched += stats['faces_matched']
            faces_transferred += stats['faces_transferred']
            faces_deleted += stats['faces_deleted']

        verb = "Would resolve" if dry_run else "Resolved"
        self.stdout.write(f"{verb} {n} cross-collection exact-phash pair(s).")
        self.stdout.write(f"  Auto-merged: {totals[CrossCollectionImageDuplicate.STATUS_AUTO_MERGED]}")
        self.stdout.write(f"  Needs review: {totals[CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW]}")
        self.stdout.write(
            f"  Faces matched: {faces_matched}, transferred: {faces_transferred}, deleted: {faces_deleted}"
        )
        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
        else:
            self.stdout.write(self.style.SUCCESS("Done."))
