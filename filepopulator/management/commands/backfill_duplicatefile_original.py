#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for DuplicateFile.original
# (see CLAUDE.md / filepopulator/models.py's own comment on that field):
# existing DuplicateFile rows predate it, so this retroactively resolves
# each one's primary ImageFile by content, the same way create_image_
# file() now does going forward.
#
# For each DuplicateFile with original IS NULL:
#   - if its own file no longer exists on disk: the record is moot
#     either way (nothing to protect, nothing left to re-ingest) --
#     delete it.
#   - if its file exists and decodes: match its pixel content against
#     ImageFile.pixel_hash. Exactly one match -> set original. No match
#     (the primary was already deleted before this field existed) ->
#     delete the record, freeing this now-sole-surviving copy to be
#     properly re-ingested as a real photo on the next scan.
#   - if its file exists but fails to decode (corrupted): leave it
#     alone rather than guess -- this is the one case that isn't safe
#     to resolve automatically.
#
# Safe to re-run: only ever touches rows where original is still NULL.

from django.core.management.base import BaseCommand

from filepopulator.models import DuplicateFile, ImageFile
import os


class Command(BaseCommand):
    help = "Backfill DuplicateFile.original by matching each duplicate's content to its primary."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only report what would happen; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        candidates = list(DuplicateFile.objects.filter(original__isnull=True))
        self.stdout.write(f"DuplicateFile rows with no original: {len(candidates)}")

        to_resolve = []  # (dup, primary)
        to_delete = []   # dup, either file gone or no primary found
        to_skip = []     # dup, file exists but failed to decode

        for dup in candidates:
            if not os.path.exists(dup.filename):
                to_delete.append(dup)
                continue

            probe = ImageFile(filename=dup.filename)
            try:
                probe.process_new_no_md5()
                probe._generate_md5_hash()
            except OSError:
                to_skip.append(dup)
                continue

            primaries = list(ImageFile.objects.filter(pixel_hash=probe.pixel_hash))
            if len(primaries) == 0:
                to_delete.append(dup)
            elif len(primaries) == 1:
                to_resolve.append((dup, primaries[0]))
            else:
                # More than one current ImageFile shares this content --
                # ambiguous, don't guess which is "the" primary.
                to_skip.append(dup)

        self.stdout.write(f"Resolvable (primary found): {len(to_resolve)}")
        self.stdout.write(f"To delete (file gone, or no primary exists anymore): {len(to_delete)}")
        self.stdout.write(f"Skipped (couldn't verify, left untouched): {len(to_skip)}")

        if not to_resolve and not to_delete:
            self.stdout.write(self.style.SUCCESS("Nothing to do."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(
                f"Set original on {len(to_resolve)} row(s) and delete {len(to_delete)} "
                f"stale row(s)? y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        for dup, primary in to_resolve:
            dup.original = primary
            dup.save()

        for dup in to_delete:
            dup.delete()

        self.stdout.write(self.style.SUCCESS(
            f"Set original on {len(to_resolve)} row(s), deleted {len(to_delete)} stale row(s)."
        ))
