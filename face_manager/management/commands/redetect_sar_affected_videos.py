#! /usr/bin/env python

# One-time (but safely re-runnable) redetect+reconcile backfill for
# already-processed videos whose decode geometry was wrong (non-square
# pixels -- see CLAUDE.md's 2026-09-23 aspect-ratio write-up: old
# digitized NTSC/VHS-era .mpg content is commonly coded at e.g. 352x480
# with sample_aspect_ratio=20:11, the TRUE display shape being 4:3, not
# the raw squished grid). Unlike the HDR tone-map backfill (a pure
# cosmetic fix, safe to just overwrite the stored thumbnail in place), a
# squished decode could have hidden faces from the detector entirely, or
# produced meaningfully worse embeddings/boxes for faces it DID find --
# per the user's own call, "I'm not confident we got all the faces if
# they were squished by 2x". So this runs a REAL full redetect on the
# now-correctly-scaled decode (VideoFaceExtractor._compute_candidate_
# groups(), the same detect->track->cluster->union-merge pipeline
# process_video() uses, factored out specifically so this command can
# reuse it without duplicating it) and reconciles the result against
# each video's existing Face rows via match_redetect_candidates_to_
# existing_faces() (optimal bipartite assignment on embedding cosine
# similarity, same "optimal assignment, then filter" pattern used
# throughout this project), rather than trusting a simple box/kps
# rescale to recover already-tagged faces.
#
# For each already-processed, SAR-affected video:
#   - matched (existing Face <-> fresh candidate): the existing row's
#     identity (declared_name/validated/poss_identN/rejected_fields) is
#     left completely untouched; its box/kps/thumbnail/det_score/
#     embedding/timestamps are overwritten with the fresh, correctly-
#     scaled values -- same fields process_video() itself sets for a
#     brand-new face, just applied in place to the existing row.
#   - unmatched candidate: a face the squished geometry hid from the
#     original pass -- created as a normal fresh, unclassified Face
#     (declared_name left blank, run through classify_unassigned() the
#     same way any new video detection is -- see VideoFaceExtractor.
#     _classify_and_pick_thumbnail()).
#   - unmatched existing Face: left alone, untouched -- never silently
#     mutated/deleted, in case it's a real identification the redetect
#     didn't happen to reproduce. Reported by name if it has a real
#     declared_name, so these are easy to find for manual review.
import time

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from face_manager.models import Face
from filepopulator.models import VideoFile
from video_face_pipeline import VideoFaceExtractor, match_redetect_candidates_to_existing_faces


class Command(BaseCommand):
    help = (
        "Redetect and reconcile already-processed videos whose decode geometry "
        "was wrong (non-square pixels) -- see CLAUDE.md's 2026-09-23 write-up."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--yes', action='store_true')

    def _existing_dicts(self, video):
        faces = list(Face.objects.filter(source_video_file=video).select_related('declared_name'))
        usable = []
        skipped = 0
        for f in faces:
            if f.face_encoding_512 is None or f.face_encoding_512 == settings.NON_DETECTED_FACE_ENCODING:
                skipped += 1
                continue
            usable.append({'face': f, 'embedding': f.face_encoding_512})
        return usable, skipped

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        videos = list(VideoFile.objects.filter(isProcessed=True, sar_scale_width__gt=0))
        self.stdout.write(f"{len(videos)} already-processed, non-square-pixel video(s) to redetect.")
        if not videos:
            self.stdout.write(self.style.SUCCESS("Nothing to do."))
            return

        if not dry_run and not options['yes']:
            go_ahead = input(f"Redetect+reconcile {len(videos)} video(s)? y/N: ")
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        extractor = VideoFaceExtractor()

        total_matched = total_new = total_unmatched_old = total_failed = 0
        for vi, video in enumerate(videos, start=1):
            video_start = time.time()
            existing, skipped = self._existing_dicts(video)

            try:
                fps, frame_pixels, candidates = extractor._compute_candidate_groups(video)
            except Exception as e:
                total_failed += 1
                self.stdout.write(f"  [{vi}/{len(videos)}] {video.filename}: redetect failed: {e}")
                continue

            cand_dicts = [{'embedding': c['centroid']} for c in candidates]
            matches = match_redetect_candidates_to_existing_faces(existing, cand_dicts)
            matched_existing_idx = {i for i, _ in matches}
            matched_cand_idx = {j for _, j in matches}

            n_matched = len(matches)
            n_new = len(candidates) - len(matched_cand_idx)
            unmatched_old = [existing[i] for i in range(len(existing)) if i not in matched_existing_idx]
            total_matched += n_matched
            total_new += n_new
            total_unmatched_old += len(unmatched_old)

            for entry in unmatched_old:
                face = entry['face']
                if face.declared_name_id and face.declared_name.person_name not in settings.IGNORED_NAMES:
                    self.stdout.write(
                        f"    unmatched-old face {face.id} declared_name="
                        f"'{face.declared_name.person_name}' -- left untouched, review manually"
                    )

            self.stdout.write(
                f"  [{vi}/{len(videos)}] {video.filename}: {len(existing)} existing "
                f"({skipped} no-embedding, skipped), {len(candidates)} candidate(s) "
                f"-> {n_matched} matched, {n_new} new, {len(unmatched_old)} unmatched-old, "
                f"{time.time() - video_start:.1f}s"
            )

            if dry_run:
                continue

            with transaction.atomic():
                for i, j in matches:
                    face = existing[i]['face']
                    cand = candidates[j]
                    provisional = cand['provisional']
                    provisional_frame = frame_pixels[provisional['frame_idx']]
                    face.face_encoding_512 = cand['centroid'].tolist()
                    face.video_first_timestamp_seconds = extractor._clamp_to_duration(
                        video, cand['first_sample'] / fps, 'video_first_timestamp_seconds'
                    )
                    face.video_last_timestamp_seconds = extractor._clamp_to_duration(
                        video, cand['last_sample'] / fps, 'video_last_timestamp_seconds'
                    )
                    extractor._set_face_box_and_thumbnail(
                        face, provisional_frame, provisional['box'], provisional['kps'],
                        provisional['frame_idx'] / fps, det_score=provisional['det_score'],
                    )
                    face.save()

                for j, cand in enumerate(candidates):
                    if j in matched_cand_idx:
                        continue
                    provisional = cand['provisional']
                    provisional_frame = frame_pixels[provisional['frame_idx']]

                    face = Face()
                    face.source_video_file = video
                    face.declared_name = extractor.blank_face_person
                    face.dateTakenUTC = video.dateTakenUTC
                    face.reencoded = True
                    face.written_to_photo_metadata = False
                    face.face_encoding_512 = cand['centroid'].tolist()
                    face.video_first_timestamp_seconds = extractor._clamp_to_duration(
                        video, cand['first_sample'] / fps, 'video_first_timestamp_seconds'
                    )
                    face.video_last_timestamp_seconds = extractor._clamp_to_duration(
                        video, cand['last_sample'] / fps, 'video_last_timestamp_seconds'
                    )
                    extractor._set_face_box_and_thumbnail(
                        face, provisional_frame, provisional['box'], provisional['kps'],
                        provisional['frame_idx'] / fps, det_score=provisional['det_score'],
                    )
                    face.save()
                    extractor._classify_and_pick_thumbnail(face, cand['pooled_reps'], frame_pixels, fps)

        verb = "Would match" if dry_run else "Matched"
        self.stdout.write(
            f"{verb} {total_matched}, new {total_new}, unmatched-old {total_unmatched_old}, "
            f"failed {total_failed} (out of {len(videos)} video(s))."
        )
