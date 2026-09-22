# Resolves VideoFile rows that turned out to share the same real file
# content (VideoFile.content_hash, populated by backfill_video_content_hash
# for rows ingested before content-hash-based duplicate detection existed
# -- see CLAUDE.md's 2026-09-22 write-up). These are NOT the DuplicateVideoFile
# case (a genuinely-new path recognized as a duplicate AT INGESTION TIME,
# which never gets its own VideoFile row in the first place) -- this command
# is for videos that were already fully, independently ingested (and
# sometimes independently face-extracted) as separate rows before this
# feature existed, found by content_hash matching after the fact.
#
# Real motivating case (2026-09-22): 5 Face rows (1093652, 1119799, 1093914,
# 1109430, 1115468) all turned out to be the same real face from the same
# real video frame, because 5 differently-named files were confirmed
# byte-for-byte identical (same MD5) and had each been independently
# face-extracted.
#
# For each content_hash group with more than one row: if at most ONE row in
# the group has isProcessed=True (i.e. has already been through face
# extraction), it's safe to merge -- pick that one (or, if none, the
# best-available row) as the primary, transfer any Face rows from the
# others onto it, and delete the others (VideoFile.delete() cleans up
# their own thumbnails and cascades Face-thumbnail cleanup too). If MORE
# THAN ONE row in a group is already isProcessed=True, merging would risk
# silently duplicating that group's Face rows onto the primary (two
# independent extraction runs over identical content won't in general
# produce identical boxes/timestamps the way a single-frame image
# duplicate does) -- left alone, reported as unresolved for manual review,
# same discipline this project already uses for the equivalent ambiguous
# image-duplicate case (see merge_duplicate_imagefiles.py's own
# "unresolved" handling).
import os
from collections import defaultdict

from django.core.management.base import BaseCommand
from django.db.models import Count

from face_manager.models import Face
from filepopulator.models import VideoFile


class Command(BaseCommand):
    help = "Merge VideoFile rows sharing a content_hash (real duplicate video content) into one."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only report what would happen; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def _find_groups(self):
        dupe_hashes = (
            VideoFile.objects.exclude(content_hash__isnull=True)
            .values('content_hash')
            .annotate(n=Count('id'))
            .filter(n__gt=1)
            .values_list('content_hash', flat=True)
        )
        groups = defaultdict(list)
        for video in VideoFile.objects.filter(content_hash__in=list(dupe_hashes)).order_by('pk'):
            groups[video.content_hash].append(video)
        return list(groups.values())

    def _pick_primary(self, group):
        face_counts = {v.pk: Face.objects.filter(source_video_file=v).count() for v in group}

        def sort_key(v):
            return (
                0 if v.isProcessed else 1,
                0 if os.path.exists(v.filename) else 1,
                -face_counts[v.pk],
                v.pk,
            )
        return sorted(group, key=sort_key)[0]

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        groups = self._find_groups()
        self.stdout.write(f"Duplicate content_hash groups found: {len(groups)}")

        safe_plans = []
        unresolved = []
        for group in groups:
            n_processed = sum(1 for v in group if v.isProcessed)
            if n_processed > 1:
                unresolved.append(group)
                continue
            primary = self._pick_primary(group)
            others = [v for v in group if v.pk != primary.pk]
            safe_plans.append((primary, others))

        total_faces_to_transfer = sum(
            Face.objects.filter(source_video_file__in=others).count()
            for _, others in safe_plans
        )
        total_rows_to_delete = sum(len(others) for _, others in safe_plans)

        self.stdout.write(f"Safe to merge: {len(safe_plans)} group(s)")
        self.stdout.write(f"  -> {total_rows_to_delete} duplicate row(s) to delete")
        self.stdout.write(f"  -> {total_faces_to_transfer} face(s) to transfer onto a primary")
        self.stdout.write(
            f"Unresolved (>1 row already face-processed, left alone): {len(unresolved)} group(s)"
        )
        for group in unresolved:
            names = ', '.join(f"{v.pk}:{v.filename}" for v in group)
            self.stdout.write(self.style.WARNING(f"  {names}"))

        if not safe_plans:
            self.stdout.write(self.style.SUCCESS("Nothing safe to merge."))
            return

        if dry_run:
            for primary, others in safe_plans:
                self.stdout.write(
                    f"  primary={primary.pk}:{primary.filename} <- "
                    f"{[f'{o.pk}:{o.filename}' for o in others]}"
                )
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(
                f"Merge {len(safe_plans)} duplicate VideoFile group(s), deleting "
                f"{total_rows_to_delete} row(s)? y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        merged_faces = 0
        deleted_rows = 0
        for primary, others in safe_plans:
            for other in others:
                merged_faces += Face.objects.filter(source_video_file=other).update(
                    source_video_file=primary
                )
                other.delete()
                deleted_rows += 1

        self.stdout.write(self.style.SUCCESS(
            f"Merged {len(safe_plans)} group(s): transferred {merged_faces} face(s), "
            f"deleted {deleted_rows} duplicate VideoFile row(s)."
        ))
