#! /usr/bin/env python

# One-time (but safely re-runnable) data cleanup for a real bug, now
# fixed at the source (see common/advisory_lock.py and CLAUDE.md):
# find_and_encode_faces() had no locking of its own, so two concurrent
# invocations processing the same never-before-processed image could
# each independently detect and insert a Face row for the same real
# face -- pixel-identical box, same embedding. Confirmed against real
# production data via pairwise IOU: ~3,745 same-image Face-row pairs
# with IOU > 0.9 (nearly all exactly 1.0). Two flavors of the same root
# cause -- isolated single-image collisions (near-adjacent ids) and
# whole-bulk-import-batch collisions (many images in one contiguous
# ImageFile-id range, larger id deltas) -- both close from here on
# thanks to the advisory lock, but neither un-does the duplicate rows
# already sitting in the database.
#
# This command finds groups of mutually-overlapping (IOU > threshold)
# Face rows on the same image and collapses each group down to one
# survivor, deleting the rest via Face.delete() (not a bulk queryset
# delete) so thumbnail files on disk are cleaned up too.
#
# Survivor choice within a group, in order: prefer already-validated
# (never discard a completed human verification), then prefer a face
# that's actually been assigned a label (declared_name isn't the blank
# sentinel -- a common real scenario is a human tagging one copy of a
# duplicate pair without knowing the other copy existed, leaving it
# blank forever), then prefer a face with kps populated (lets a later
# reencode reproduce the exact embedding without re-detecting), then
# prefer the lowest id (earliest/canonical row) as a final, deterministic
# tiebreaker.
#
# Safe to re-run: once duplicates are gone, this finds nothing.

from collections import defaultdict

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count

from face_manager.models import Face, Person


def _iou(a, b):
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il = max(al, bl)
    it = max(at, bt)
    ir = min(ar, br)
    ib = min(ab, bb)
    if il >= ir or it >= ib:
        return 0.0
    inter = (ir - il) * (ib - it)
    area_a = (ar - al) * (ab - at)
    area_b = (br - bl) * (bb - bt)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _connected_groups(faces, iou_thresh):
    """faces: list of (face_id, box, validated, has_kps, has_label). Returns a list
    of groups (each a list of the same tuples) where every pair within a
    group has IOU > iou_thresh with at least one other member
    (single-linkage is fine here -- duplicate faces are pixel-identical
    or near enough that transitive chaining onto an unrelated face isn't
    a realistic risk, unlike the much noisier embedding-similarity
    clustering elsewhere in this codebase)."""
    n = len(faces)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(faces[i][1], faces[j][1]) > iou_thresh:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(faces[i])
    return [g for g in groups.values() if len(g) > 1]


def _pick_survivor(group):
    def sort_key(face):
        face_id, box, validated, has_kps, has_label = face
        return (0 if validated else 1, 0 if has_label else 1, 0 if has_kps else 1, face_id)
    return sorted(group, key=sort_key)[0]


class Command(BaseCommand):
    help = "Delete duplicate Face rows (near-pixel-identical boxes on the same image)."

    IOU_THRESH = 0.9

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only report how many groups/faces would be affected; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        multi_ids = list(
            Face.objects.values('source_image_file_id')
            .annotate(n=Count('id')).filter(n__gte=2)
            .values_list('source_image_file_id', flat=True)
        )

        by_image = defaultdict(list)
        qs = Face.objects.filter(source_image_file_id__in=multi_ids).values_list(
            'id', 'source_image_file_id', 'box_left', 'box_top', 'box_right', 'box_bottom',
            'validated', 'kps', 'declared_name__person_name',
        )
        for face_id, img_id, l, t, r, b, validated, kps, person_name in qs.iterator(chunk_size=5000):
            has_label = person_name != settings.BLANK_FACE_NAME
            by_image[img_id].append((face_id, (l, t, r, b), validated, kps is not None, has_label))

        to_delete = []
        affected_person_ids = set()
        groups_found = 0

        for img_id, faces in by_image.items():
            for group in _connected_groups(faces, self.IOU_THRESH):
                groups_found += 1
                survivor = _pick_survivor(group)
                for face in group:
                    if face[0] != survivor[0]:
                        to_delete.append(face[0])

        self.stdout.write(f"Duplicate groups found: {groups_found}")
        self.stdout.write(f"Face rows to delete: {len(to_delete)}")

        if not to_delete:
            self.stdout.write(self.style.SUCCESS("Nothing to clean up."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(f"Delete {len(to_delete)} duplicate face(s)? y/N: ")
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        deleted = 0
        for face_id in to_delete:
            try:
                face = Face.objects.get(pk=face_id)
                if face.declared_name_id:
                    affected_person_ids.add(face.declared_name_id)
                face.delete()
                deleted += 1
            except Face.DoesNotExist:
                pass

        for person in Person.objects.filter(id__in=affected_person_ids):
            person.num_faces = person.face_declared.count()
            person.num_possibilities = person.face_poss1.count()
            person.num_unverified_faces = person.face_declared.filter(validated=False).count()
            person.save()

        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} duplicate face(s)."))
        self.stdout.write(self.style.SUCCESS(
            f"Recomputed face counts for {len(affected_person_ids)} affected person(s)."
        ))
