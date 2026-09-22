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
# face-extracted. Checked directly against that real group: because the
# video bytes are identical, the (deterministic) extraction pipeline
# produces Face rows with video_first_timestamp_seconds/
# video_last_timestamp_seconds matching to many decimal places across
# copies -- timestamp closeness is a very strong, reliable signal for
# "this is the same real face", not a loose heuristic.
#
# Every content_hash group -- regardless of how many copies were already
# face-extracted -- is resolved the same way now: pick a primary, then for
# each other copy, match its Face rows against the primary's CURRENT Face
# rows via optimal bipartite assignment (scipy.linear_sum_assignment) on
# timestamp closeness, with a cosine-similarity floor on the embeddings as
# a safety veto (guards against two different real people who happen to
# overlap in time within the tolerance -- unlikely given this pipeline's
# own sampling stride, but not assumed impossible). A face is never
# force-matched: linear_sum_assignment finds the best OVERALL pairing, but
# any pair whose cost exceeds the tolerance (or fails the embedding floor)
# is rejected afterward, exactly the same "optimal assignment, then filter"
# pattern face_extract_encode.py already uses for reconciling existing vs.
# freshly-detected image faces.
#
# For each matched pair, the SURVIVOR (kept; the other is deleted) is
# chosen by the same preference this project already uses for the
# equivalent same-image duplicate-face case (dedupe_overlapping_faces.py):
# validated > has a real label > has kps > has det_score > lowest id --
# refined here to treat settings.IGNORED_NAMES (not just the blank
# sentinel) as "no real label", so a genuine person name always outranks
# a mere .ignore/.realignore. Any face NOT matched to anything on the
# other side -- a detection only one copy's extraction run happened to
# find -- is never discarded, only transferred onto the primary.
import os
from collections import defaultdict

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Count
from scipy.optimize import linear_sum_assignment

from face_manager.models import Face
from filepopulator.models import VideoFile

TIME_TOLERANCE_SECONDS = 0.5
EMBEDDING_COS_FLOOR = 0.5
UNREACHABLE_COST = 1e6


def _face_time_distance(a_first, a_last, b_first, b_last):
    if a_first is None or a_last is None or b_first is None or b_last is None:
        return None
    return abs(a_first - b_first) + abs(a_last - b_last)


def match_duplicate_video_faces(primary_faces, other_faces,
                                 time_tolerance_seconds=TIME_TOLERANCE_SECONDS,
                                 embedding_cos_floor=EMBEDDING_COS_FLOOR):
    """Match Face rows between two copies of the SAME (byte-identical)
    video's content. primary_faces/other_faces: list of dicts with keys
    'first'/'last' (float seconds or None) and 'embedding' (1-D
    array-like or None).

    Returns a list of (primary_index, other_index) pairs considered the
    same real face. Any index absent from every pair is unmatched and
    must be treated by the caller as a distinct face to keep -- this
    never forces a low-confidence match: linear_sum_assignment picks the
    single best overall pairing, but any pair whose timestamp distance
    exceeds the tolerance, or (when both sides have an embedding) whose
    cosine similarity falls under the floor, is rejected afterward."""
    n, m = len(primary_faces), len(other_faces)
    if n == 0 or m == 0:
        return []

    cost = np.full((n, m), UNREACHABLE_COST)
    for i, p in enumerate(primary_faces):
        for j, o in enumerate(other_faces):
            d = _face_time_distance(p['first'], p['last'], o['first'], o['last'])
            if d is not None:
                cost[i, j] = d

    rows, cols = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(rows, cols):
        if cost[i, j] > time_tolerance_seconds:
            continue
        p_emb, o_emb = primary_faces[i].get('embedding'), other_faces[j].get('embedding')
        if p_emb is not None and o_emb is not None:
            p_emb = np.asarray(p_emb, dtype=float)
            o_emb = np.asarray(o_emb, dtype=float)
            denom = np.linalg.norm(p_emb) * np.linalg.norm(o_emb)
            cos_sim = float(np.dot(p_emb, o_emb) / denom) if denom > 0 else 0.0
            if cos_sim < embedding_cos_floor:
                continue
        matches.append((i, j))
    return matches


def _has_real_label(person_name):
    return person_name not in settings.IGNORED_NAMES


def _survivor_key(face_info):
    """Lower sorts first (wins). Mirrors dedupe_overlapping_faces.py's
    _pick_survivor preference, refined to treat any IGNORED_NAMES entry
    (not just the blank sentinel) as "no real label"."""
    return (
        0 if face_info['validated'] else 1,
        0 if _has_real_label(face_info['person_name']) else 1,
        0 if face_info['has_kps'] else 1,
        0 if face_info['has_det_score'] else 1,
        face_info['id'],
    )


def _load_face_infos(video):
    infos = []
    for f in Face.objects.filter(source_video_file=video).select_related('declared_name'):
        infos.append({
            'id': f.pk,
            'first': f.video_first_timestamp_seconds,
            'last': f.video_last_timestamp_seconds,
            'embedding': f.face_encoding_512,
            'validated': f.validated,
            'person_name': f.declared_name.person_name if f.declared_name_id else settings.BLANK_FACE_NAME,
            'has_kps': f.kps is not None,
            'has_det_score': f.det_score is not None,
        })
    return infos


def resolve_group(primary, others, dry_run):
    """Merge every "other" VideoFile in a duplicate-content group onto
    primary. Returns stats; performs real writes only when dry_run is
    False. The same code path computes the plan either way, so dry-run
    reporting is guaranteed to match what a real run would do."""
    primary_faces = _load_face_infos(primary)
    stats = {'matched_pairs': 0, 'faces_deleted': 0, 'faces_transferred': 0, 'videos_deleted': 0}

    for other in others:
        other_faces = _load_face_infos(other)
        matches = match_duplicate_video_faces(primary_faces, other_faces)
        matched_other_idx = {j for _, j in matches}

        primary_delete_ids = []
        other_delete_ids = []
        other_transfer_ids = []

        for i, j in matches:
            p_info, o_info = primary_faces[i], other_faces[j]
            if _survivor_key(o_info) < _survivor_key(p_info):
                # The other copy's version of this same real face is the
                # better one (e.g. it's the copy a human actually
                # labeled) -- keep it, drop primary's.
                primary_delete_ids.append(p_info['id'])
                other_transfer_ids.append(o_info['id'])
            else:
                other_delete_ids.append(o_info['id'])

        for j, o_info in enumerate(other_faces):
            if j not in matched_other_idx:
                # Only this copy's extraction run found this face at all
                # -- never discard it.
                other_transfer_ids.append(o_info['id'])

        if not dry_run:
            for face in Face.objects.filter(pk__in=primary_delete_ids):
                face.delete()
            for face in Face.objects.filter(pk__in=other_delete_ids):
                face.delete()
            if other_transfer_ids:
                Face.objects.filter(pk__in=other_transfer_ids).update(source_video_file=primary)
            other.delete()

        stats['matched_pairs'] += len(matches)
        stats['faces_deleted'] += len(primary_delete_ids) + len(other_delete_ids)
        stats['faces_transferred'] += len(other_transfer_ids)
        stats['videos_deleted'] += 1

        # Keep the virtual primary face list in sync for the NEXT other
        # in this group (duplicate groups commonly have 3+ copies).
        primary_faces = (
            [info for info in primary_faces if info['id'] not in primary_delete_ids]
            + [o for o in other_faces if o['id'] in other_transfer_ids]
        )

    return stats


class Command(BaseCommand):
    help = (
        "Merge VideoFile rows sharing a content_hash (real duplicate video content) into "
        "one, matching and deduping their Face rows so nothing assigned/validated is lost."
    )

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
        if not groups:
            self.stdout.write(self.style.SUCCESS("Nothing to merge."))
            return

        total_rows_to_delete = sum(len(g) - 1 for g in groups)

        if not dry_run and not options['yes']:
            go_ahead = input(
                f"Merge {len(groups)} duplicate VideoFile group(s), deleting up to "
                f"{total_rows_to_delete} row(s)? Face rows are matched and deduped "
                f"(preferring validated/labeled copies), never blindly discarded. y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        totals = {'matched_pairs': 0, 'faces_deleted': 0, 'faces_transferred': 0, 'videos_deleted': 0}
        for group in groups:
            primary = self._pick_primary(group)
            others = [v for v in group if v.pk != primary.pk]
            if dry_run:
                stats = resolve_group(primary, others, dry_run=True)
            else:
                with transaction.atomic():
                    stats = resolve_group(primary, others, dry_run=False)
            for key in totals:
                totals[key] += stats[key]

        verb = "Would delete" if dry_run else "Deleted"
        self.stdout.write(f"Matched (same real face found on multiple copies): {totals['matched_pairs']}")
        self.stdout.write(f"{verb} {totals['faces_deleted']} duplicate Face row(s) (kept the better copy each time)")
        self.stdout.write(f"Transferred {totals['faces_transferred']} Face row(s) onto their group's primary")
        self.stdout.write(f"{verb} {totals['videos_deleted']} duplicate VideoFile row(s)")

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
        else:
            self.stdout.write(self.style.SUCCESS("Done."))
