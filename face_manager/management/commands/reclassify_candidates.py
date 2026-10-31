#! /usr/bin/env python

# Re-evaluates currently-PROPOSED (poss_ident1, not yet human-confirmed)
# candidates for a specific set of people, under a chosen embedding field --
# either the real production insightface pipeline (classify_unassigned(),
# unmodified) or a direct LVFace-B cosine-similarity check against a single
# global threshold (2026-10-31, see CLAUDE.md's empirical validation --
# unlike insightface, LVFace-B did NOT need gallery-size-bucketed
# thresholds: TPR ~97%, FPR <0.4% held consistently from 30-face galleries
# up through 16,653-face ones).
#
# Built to replace ad hoc one-off scratch scripts (scratch_reclassify_
# magnets.py) with one reusable, parameterized tool: any person-ids, any
# supported encoding, any threshold, not hardcoded to a specific name list
# or a specific moment's calibration.
#
# A rejected candidate is reverted the exact same way classify_unassigned()
# itself rejects a candidate -- set_possible_person(ignore_person, 1,
# weight) -- not a different mechanism, so the resulting state is
# indistinguishable from what the real pipeline would have produced.

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from face_manager.assign_faces import faceAssigner
from face_manager.models import Face, Person

ENCODING_FIELDS = {
    'insightface': 'face_encoding_512',
    'lvface_b': 'face_encoding_lvface_b',
}


def _normed(arr):
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


class Command(BaseCommand):
    help = (
        "Re-evaluates currently-proposed candidates for --person-ids under "
        "either the real insightface pipeline or a direct LVFace-B "
        "similarity check, reverting anything that no longer clears the "
        "bar back to .ignore -- the same outcome classify_unassigned() "
        "itself would produce, not a separate mechanism."
    )

    def add_arguments(self, parser):
        parser.add_argument('--person-ids', type=str, required=True,
                             help='Comma-separated Person ids to re-evaluate proposed candidates for.')
        parser.add_argument('--encoding', choices=sorted(ENCODING_FIELDS), default='insightface',
                             help="'insightface' reuses the real classify_unassigned() pipeline "
                                  "unmodified (gallery-size-bucketed thresholds, margin/det_score "
                                  "gates, etc). 'lvface_b' does a direct sim_99th-vs-threshold "
                                  "check against the person's own validated LVFace-B gallery -- "
                                  "no bucketing, see this file's own header comment for why.")
        parser.add_argument('--threshold', type=float, default=None,
                             help='Required for --encoding=lvface_b -- a candidate is reverted '
                                  "if its sim_99th against the person's validated LVFace-B "
                                  'gallery falls below this value. Ignored for --encoding=insightface, '
                                  "which always uses the pipeline's own real thresholds.")
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        person_ids = [int(s.strip()) for s in options['person_ids'].split(',') if s.strip()]
        encoding = options['encoding']
        threshold = options['threshold']
        dry_run = options['dry_run']

        if encoding == 'lvface_b' and threshold is None:
            raise CommandError('--threshold is required for --encoding=lvface_b')

        people = list(Person.objects.filter(id__in=person_ids))
        found_ids = {p.id for p in people}
        missing = set(person_ids) - found_ids
        if missing:
            raise CommandError(f'Person id(s) not found: {sorted(missing)}')

        before_counts = {p.id: Face.objects.filter(poss_ident1=p).count() for p in people}

        if encoding == 'insightface':
            reverted, kept = self._reclassify_insightface(person_ids, dry_run)
        else:
            reverted, kept = self._reclassify_lvface_b(people, threshold, dry_run)

        self.stdout.write(
            f"\n{'Would revert' if dry_run else 'Reverted'}: {reverted}, "
            f"{'would keep' if dry_run else 'kept'}: {kept}.\n"
        )
        for p in people:
            after = before_counts[p.id] if dry_run else Face.objects.filter(poss_ident1=p).count()
            self.stdout.write(f"  {p.person_name!r}: before={before_counts[p.id]} after={after}")

    def _reclassify_insightface(self, person_ids, dry_run):
        sentinel = list(settings.NON_DETECTED_FACE_ENCODING)
        target_faces = list(
            Face.objects.filter(poss_ident1_id__in=person_ids)
            .exclude(face_encoding_512__isnull=True)
            .exclude(face_encoding_512=sentinel)
        )
        self.stdout.write(f"Loading encodings...")
        fa = faceAssigner()
        fa.load_encodings()
        self.stdout.write(f"Reclassifying {len(target_faces)} currently-proposed faces (insightface)...")

        reverted = kept = 0
        for i, f in enumerate(target_faces):
            old_person_id = f.poss_ident1_id
            if not dry_run:
                fa.classify_unassigned(f)
                f.refresh_from_db()
                changed = f.poss_ident1_id != old_person_id
            else:
                changed = None
            if changed:
                reverted += 1
            else:
                kept += 1
            if (i + 1) % 200 == 0:
                self.stdout.write(f"  ...{i + 1}/{len(target_faces)} reverted={reverted} kept={kept}")
        return reverted, kept

    def _reclassify_lvface_b(self, people, threshold, dry_run):
        field = ENCODING_FIELDS['lvface_b']
        sentinel = list(settings.NON_DETECTED_FACE_ENCODING)
        ignore_person = Person.objects.filter(person_name=settings.SOFT_IGNORE_NAME).first()
        if ignore_person is None:
            raise CommandError(f'Sentinel person {settings.SOFT_IGNORE_NAME!r} not found')

        reverted = kept = 0
        for p in people:
            gallery = list(
                Face.objects.filter(declared_name=p, validated=True, **{f'{field}__isnull': False})
                .exclude(**{field: sentinel})
                .values_list(field, flat=True)
            )
            if not gallery:
                self.stdout.write(f"  {p.person_name!r}: no validated LVFace-B gallery yet -- skipping")
                continue
            gallery_arr = _normed(np.array(gallery, dtype=np.float32))

            proposed = list(
                Face.objects.filter(poss_ident1=p, **{f'{field}__isnull': False})
                .exclude(**{field: sentinel})
            )
            self.stdout.write(
                f"  {p.person_name!r}: gallery={len(gallery_arr)}, "
                f"proposed_with_lvface_b={len(proposed)}"
            )
            for f in proposed:
                emb = _normed(np.array(getattr(f, field), dtype=np.float32).reshape(1, -1))[0]
                sims = gallery_arr @ emb
                p99 = float(np.percentile(sims, 99))
                if p99 < threshold:
                    reverted += 1
                    if not dry_run:
                        # set_possible_person requires weight >= 0; raw cosine
                        # similarity can go slightly negative for a genuinely
                        # dissimilar pair, so clamp rather than pass it through
                        # raw (classify_unassigned() itself always computes a
                        # normalized 0..1 weight for its own ignore case, never
                        # a raw score -- see its own margin-clamping comment).
                        f.set_possible_person(ignore_person, 1, max(0.0, p99))
                else:
                    kept += 1
        return reverted, kept
