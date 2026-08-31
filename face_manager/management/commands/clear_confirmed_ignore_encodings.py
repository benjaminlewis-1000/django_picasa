#! /usr/bin/env python

# Storage cleanup: face_encoding_512 for confirmed .ignore/.realignore
# faces accounts for a large chunk of face_manager_face's storage (see
# CLAUDE.md's "consider nulling face_encoding_512 for .ignore/.realignore
# faces" note) but is never usefully compared again once a face is
# permanently ignored. This clears it to reclaim that space, keeping
# Face.kps (if present) so FaceExtractor.reencode_missing_faces() can
# still recover a usable embedding later if the face is ever reassigned
# away from .ignore/.realignore.
#
# Deliberately restricted to CONFIRMED faces only -- declared_name
# actually equal to .ignore/.realignore. classify_unassigned() only ever
# writes an ignore suggestion to poss_ident1, never to declared_name;
# declared_name only becomes .ignore/.realignore through a human bulk
# action (close_unassigned/close_ignored/confirm_proposed, all of which
# go through Face.associate_person()). A face merely SUGGESTED as ignore
# (poss_ident1 set, declared_name still unassigned) is left untouched --
# its embedding is exactly what a human still needs to review that
# suggestion against.
#
# Safe to re-run: matches nothing once already cleared.

from django.conf import settings
from django.core.management.base import BaseCommand

from face_manager.models import Face, clear_confirmed_ignore_face_encodings


class Command(BaseCommand):
    help = (
        "Clear face_encoding_512 for CONFIRMED .ignore/.realignore faces "
        "(declared_name only -- never merely-suggested faces) to reclaim storage. "
        "Face.kps is left untouched."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many faces would be affected; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        qs = Face.objects.filter(
            declared_name__person_name__in=[settings.SOFT_IGNORE_NAME, '.realignore'],
        ).exclude(face_encoding_512__isnull=True)

        total = qs.count()
        soft_count = qs.filter(declared_name__person_name=settings.SOFT_IGNORE_NAME).count()
        hard_count = qs.filter(declared_name__person_name='.realignore').count()
        with_kps = qs.exclude(kps__isnull=True).count()
        without_kps = total - with_kps

        self.stdout.write(f"Confirmed-ignore faces with an encoding to clear: {total}")
        self.stdout.write(f"  .{settings.SOFT_IGNORE_NAME.strip('.')} : {soft_count}")
        self.stdout.write(f"  .realignore : {hard_count}")
        self.stdout.write(
            f"  {with_kps} have kps stored (exact recovery later); "
            f"{without_kps} do not (approximate crop-based recovery only)."
        )

        if total == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to clear."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(
                f"Clear face_encoding_512 for {total} confirmed-ignore face(s)? "
                "This is only approximately recoverable for faces without kps. y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        updated = clear_confirmed_ignore_face_encodings()
        self.stdout.write(self.style.SUCCESS(f"Cleared face_encoding_512 for {updated} face(s)."))
