#! /usr/bin/env python

# One-time (but safely re-runnable) cleanup for the create_image_file()
# missing-return bug (see CLAUDE.md): every genuine duplicate file used
# to get BOTH a correct DuplicateFile record AND a redundant ImageFile
# row of its own. That row can carry real, independent human work
# (labels, validations) if someone tagged faces on the duplicate without
# realizing it was a duplicate -- confirmed in production: 5,009 faces
# across 1,197 contaminated rows, 1,190 of them validated.
#
# For each contaminated row (an ImageFile whose own filename also has a
# DuplicateFile record), finds its "primary" (another ImageFile sharing
# the same pixel_hash, not itself flagged as a duplicate), reassigns
# every Face on the duplicate row over to the primary, collapses any
# resulting same-image duplicate face pairs on the primary (reusing
# dedupe_overlapping_faces' own grouping/survivor logic, since the two
# rows are pixel-identical and their faces will usually land at the same
# coordinates), and then deletes the now-empty duplicate ImageFile row
# (which also cleans up its own thumbnail files -- see ImageFile.
# delete()).
#
# Safe to re-run: once no ImageFile row has a matching DuplicateFile
# entry, this finds nothing.

from django.conf import settings
from django.core.management.base import BaseCommand

from face_manager.management.commands.dedupe_overlapping_faces import (
    _connected_groups, _pick_survivor,
)
from face_manager.models import Face, Person
from filepopulator.models import DuplicateFile, ImageFile


class Command(BaseCommand):
    help = "Merge contaminated duplicate ImageFile rows into their primary, preserving face data."

    IOU_THRESH = 0.9

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only report what would happen; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def _resolve_pairs(self):
        dup_filenames = set(DuplicateFile.objects.values_list('filename', flat=True))
        contaminated = list(ImageFile.objects.filter(filename__in=dup_filenames))

        pairs = []
        unresolved = []
        for dup in contaminated:
            primary_candidates = list(
                ImageFile.objects.filter(pixel_hash=dup.pixel_hash)
                .exclude(id=dup.id)
                .exclude(filename__in=dup_filenames)
                .order_by('id')
            )
            if not primary_candidates:
                unresolved.append(dup)
                continue
            pairs.append((dup, primary_candidates[0]))

        return pairs, unresolved

    def _collapse_duplicate_faces_on(self, image, affected_person_ids):
        """After transferring faces onto `image`, some may now be
        pixel-identical duplicates of faces already there (the two
        source rows were pixel-identical, so a real face detected on
        both will usually land at the same box). Collapse those the
        same way dedupe_overlapping_faces does. Returns the number of
        faces deleted."""
        faces = list(Face.objects.filter(source_image_file=image).values_list(
            'id', 'box_left', 'box_top', 'box_right', 'box_bottom',
            'validated', 'kps', 'declared_name__person_name',
        ))
        face_tuples = [
            (fid, (l, t, r, b), validated, kps is not None, person_name != settings.BLANK_FACE_NAME)
            for fid, l, t, r, b, validated, kps, person_name in faces
        ]

        deleted = 0
        for group in _connected_groups(face_tuples, self.IOU_THRESH):
            survivor = _pick_survivor(group)
            for face in group:
                if face[0] != survivor[0]:
                    f = Face.objects.get(pk=face[0])
                    if f.declared_name_id:
                        affected_person_ids.add(f.declared_name_id)
                    f.delete()
                    deleted += 1
        return deleted

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        pairs, unresolved = self._resolve_pairs()

        self.stdout.write(f"Contaminated duplicate ImageFile rows: {len(pairs) + len(unresolved)}")
        self.stdout.write(f"Resolvable pairs (duplicate -> primary): {len(pairs)}")
        self.stdout.write(f"Unresolved (no primary found, left alone): {len(unresolved)}")
        for dup in unresolved:
            self.stdout.write(self.style.WARNING(f"  no primary found for: {dup.filename}"))

        total_faces_to_transfer = sum(
            Face.objects.filter(source_image_file=dup).count() for dup, _ in pairs
        )
        self.stdout.write(f"Faces to transfer: {total_faces_to_transfer}")

        if not pairs:
            self.stdout.write(self.style.SUCCESS("Nothing to clean up."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(
                f"Merge {len(pairs)} duplicate ImageFile row(s) into their primaries? y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        total_deleted_dupes = 0
        total_collapsed_faces = 0
        affected_person_ids = set()

        for dup, primary in pairs:
            Face.objects.filter(source_image_file=dup).update(source_image_file=primary)
            total_collapsed_faces += self._collapse_duplicate_faces_on(primary, affected_person_ids)
            dup.delete()
            total_deleted_dupes += 1

        for person in Person.objects.filter(id__in=affected_person_ids):
            person.num_faces = person.face_declared.count()
            person.num_possibilities = person.face_poss1.count()
            person.num_unverified_faces = person.face_declared.filter(validated=False).count()
            person.save()

        self.stdout.write(self.style.SUCCESS(
            f"Merged and deleted {total_deleted_dupes} duplicate ImageFile row(s)."
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Collapsed {total_collapsed_faces} resulting duplicate face(s)."
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Recomputed face counts for {len(affected_person_ids)} affected person(s)."
        ))
