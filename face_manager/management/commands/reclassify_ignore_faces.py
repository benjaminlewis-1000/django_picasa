#! /usr/bin/env python

# Re-runs classify_unassigned() against every currently ".ignore"-proposed,
# still-unassigned face -- a one-off pass added 2026-11-16 alongside the
# next-best-guesses feature (poss_ident2-5 now get populated with below-
# threshold candidate guesses even when poss_ident1 stays .ignore, see
# assign_faces.py's _populate_next_best_guesses()).
#
# The regular scheduled assign_faces task (execute(redo_all=False)) never
# touches this population -- it only picks up faces whose poss_ident1 is
# still NULL (see execute()'s own `no_suggestions` filter) -- so without
# this one-off pass, the existing .ignore backlog would sit with blank
# poss_ident2-5 until each face happened to be reprocessed by a future
# full redo_all=True run.

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from tqdm import tqdm

from face_manager.assign_faces import faceAssigner
from face_manager.models import Face, Person


class Command(BaseCommand):
    help = (
        "Re-runs classify_unassigned() against every currently "
        "'.ignore'-proposed, still-unassigned face, so the poss_ident2-5 "
        "next-best-guess population reaches the existing backlog, not "
        "just future classifications."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true',
                             help='Report the target population size only; make no changes.')

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME).first()
        if ignore_person is None:
            raise CommandError(f'Sentinel person {settings.SOFT_IGNORE_NAME!r} not found')

        select_related_fields = ('declared_name', 'source_image_file', 'source_video_file')
        target = (
            Face.objects.filter(
                declared_name__person_name=settings.BLANK_FACE_NAME,
                poss_ident1=ignore_person,
            )
            .select_related(*select_related_fields)
            .order_by('?')
        )
        total = target.count()
        self.stdout.write(f"{total} .ignore-proposed face(s) to reclassify.")
        if dry_run or total == 0:
            return

        fa = faceAssigner()
        self.stdout.write("Loading encodings...")
        fa.load_encodings()

        batch = []
        progress = tqdm(total=total)
        for face in target.iterator():
            batch.append(face)
            if len(batch) >= fa.CLASSIFY_BATCH_SIZE:
                fa._classify_batch_safely(batch)
                progress.update(len(batch))
                batch = []
        if batch:
            fa._classify_batch_safely(batch)
            progress.update(len(batch))
        progress.close()

        fa._verify_classification_invariant()

        still_ignore = Face.objects.filter(
            declared_name__person_name=settings.BLANK_FACE_NAME, poss_ident1=ignore_person,
        )
        still_ignore_count = still_ignore.count()
        promoted = total - still_ignore_count
        with_next_best_guess = still_ignore.exclude(poss_ident2__isnull=True).count()

        self.stdout.write(self.style.SUCCESS(
            f"Done. {promoted} promoted to a real candidate (now above threshold); "
            f"{still_ignore_count} remain .ignore, of which {with_next_best_guess} "
            f"now carry a poss_ident2 next-best guess."
        ))
