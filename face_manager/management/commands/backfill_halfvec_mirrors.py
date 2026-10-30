#! /usr/bin/env python

# One-time (but safely re-runnable) backfill for the 4 halfvec mirror
# columns added 2026-10-30 (face_encoding_512_hv/face_encoding_lvface_
# {b,s,l}_hv -- see CLAUDE.md's pgvector investigation and Face.
# set_embedding()'s own docstring) -- every real WRITE call site has
# already been updated to dual-write both columns going forward (and
# the 3 long-running production backfill/remediation jobs active at
# the time were restarted specifically to pick that up), but every row
# written BEFORE that still only has the float32 field. This is a pure
# copy -- no decode, no model inference, no computation at all, so
# it's cheap even at the real ~420k-row scale (unlike the LVFace
# embedding backfills, which pay real per-face inference cost).
#
# Processes all 4 fields together per face (one combined query/update
# pass), not 4 separate full-table scans, since a face can need
# backfilling on more than one field at once.

from django.core.management.base import BaseCommand
from django.db.models import Q

from face_manager.models import Face

BATCH_SIZE = 5000
EMBEDDING_FIELDS = [
    'face_encoding_512', 'face_encoding_lvface_b', 'face_encoding_lvface_s', 'face_encoding_lvface_l',
]


class Command(BaseCommand):
    help = (
        "Backfill the halfvec mirror columns (face_encoding_512_hv/"
        "face_encoding_lvface_{b,s,l}_hv) for rows written before the "
        "dual-write migration -- a pure copy of the already-stored "
        "float32 value, no model inference."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None,
                             help='Only process the first N candidate faces (for a quick check).')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']

        needs_backfill = Q()
        for field in EMBEDDING_FIELDS:
            needs_backfill |= Q(**{f'{field}__isnull': False, f'{field}_hv__isnull': True})

        qs = Face.objects.filter(needs_backfill).only(
            'id', *EMBEDDING_FIELDS, *[f'{f}_hv' for f in EMBEDDING_FIELDS]
        ).order_by('id')
        if limit is not None:
            qs = qs[:limit]

        per_field_updated = {f: 0 for f in EMBEDDING_FIELDS}
        total = 0
        to_update = []
        update_fields_needed = set()

        def flush():
            if to_update and not dry_run:
                Face.objects.bulk_update(to_update, list(update_fields_needed), batch_size=BATCH_SIZE)
            to_update.clear()
            update_fields_needed.clear()

        for face in qs.iterator(chunk_size=BATCH_SIZE):
            total += 1
            touched = False
            for field in EMBEDDING_FIELDS:
                value = getattr(face, field)
                hv_field = f'{field}_hv'
                if value is not None and getattr(face, hv_field) is None:
                    setattr(face, hv_field, value)
                    update_fields_needed.add(hv_field)
                    per_field_updated[field] += 1
                    touched = True
            if touched:
                to_update.append(face)
            if len(to_update) >= BATCH_SIZE:
                flush()
            if total % 50000 == 0:
                self.stdout.write(f"  ...{total} candidate face(s) processed")

        flush()

        self.stdout.write(
            f"\n{'Would backfill' if dry_run else 'Backfilled'} {total} candidate face(s):"
        )
        for field in EMBEDDING_FIELDS:
            self.stdout.write(f"  {field}_hv: {per_field_updated[field]}")
