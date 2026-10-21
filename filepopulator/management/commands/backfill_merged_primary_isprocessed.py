# One-time catch-up: cross_collection_dedup.py's resolve_pair() now
# marks a merge's primary isProcessed=True when secondary had already
# completed its own real detection pass (2026-10-21, see CLAUDE.md) --
# but that only applies to merges run AFTER the fix landed. The original
# backlog run (4,678 auto-merged pairs) predates it, leaving many
# primaries sitting at isProcessed=False despite already holding a real,
# complete face set via the merge (matched survivors + transferred
# faces) -- meaning they'd otherwise sit forever in the normal
# face_extraction backlog for a genuinely unnecessary redetection pass.
#
# secondary.isProcessed can't be used retroactively to tell "was it
# processed BEFORE the merge" -- resolve_pair() always sets it True
# unconditionally regardless of prior state, so by now every merged
# secondary reads True either way. Instead, uses a reliable proxy
# already stored on each CrossCollectionImageDuplicate row:
# faces_matched/faces_transferred are only nonzero if secondary actually
# had real Face rows at merge time, which only happens if a real
# detection pass had already run on it. A secondary that was processed
# but genuinely found zero faces is the one real gap this proxy can't
# distinguish from "never processed" -- a harmless false negative (that
# primary just gets a real, likely-also-zero-yielding redetection pass
# later), not a correctness risk.
from django.db.models import F

from django.core.management.base import BaseCommand

from filepopulator.models import CrossCollectionImageDuplicate, ImageFile


class Command(BaseCommand):
    help = (
        "Mark a merge's primary ImageFile isProcessed=True when its merge "
        "record shows secondary had real faces (matched/transferred) at merge "
        "time, for merges that predate that fix being added to resolve_pair()."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        # (faces_matched + faces_transferred) > 0 -- can't express a sum
        # of two columns compared to a constant directly in .filter()
        # kwargs, so annotate then filter.
        candidates = CrossCollectionImageDuplicate.objects.filter(
            status=CrossCollectionImageDuplicate.STATUS_AUTO_MERGED,
            primary_image__isnull=False,
            primary_image__isProcessed=False,
        ).annotate(
            faces_present=F('faces_matched') + F('faces_transferred'),
        ).filter(faces_present__gt=0)

        primary_ids = list(candidates.values_list('primary_image_id', flat=True).distinct())
        self.stdout.write(
            f"{'Would mark' if dry_run else 'Marking'} {len(primary_ids)} merged primary "
            f"ImageFile row(s) isProcessed=True (had real transferred/matched faces at merge time)."
        )
        if not dry_run and primary_ids:
            ImageFile.objects.filter(pk__in=primary_ids).update(isProcessed=True)
