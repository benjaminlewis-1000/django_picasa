"""Rigorous, verified kps/det_score/LVFace backfill -- the real fix for
the kps/box translation bug (see pyramidal_detector.py's find_raw_faces(),
and CLAUDE.md for the full investigation). Unlike backfill_det_score.py
(which only has the matched detection's OWN box geometry to lean on),
this command cross-checks every match against something semantically
meaningful: the matched detection's freshly-computed insightface
embedding, compared to the face's own ALREADY-TRUSTED stored
face_encoding_512.

Deliberately oriented around whole IMAGES, not individual faces (per
explicit design direction) -- real efficiency and accuracy comes from
detecting once per image, matching ALL of that image's faces against the
detections in one optimal bipartite pass, then handling every face on
it together, rather than one wasted redetect per trigger face. An
incidental face that happens to share an already-triggered image (e.g.
an unlabeled/.ignore face with no stored encoding to verify against)
still gets its kps/det_score/LVFace embeddings saved -- just without
the identity-verification gate, since there's nothing to verify
against; evaluate those later rather than discard the opportunity.

Verification tiers, per face WITH a real stored face_encoding_512:
  - similarity >= HIGH_CONFIDENCE_SIM: accept outright.
  - GRAY_ZONE_SIM <= similarity < HIGH_CONFIDENCE_SIM: accept only if
    the match is a tight box match (IOU >= GRAY_ZONE_MIN_IOU) AND the
    face is small (relative_size <= GRAY_ZONE_MAX_RELATIVE_SIZE) --
    calibrated against 39 real faces with known-good stored embeddings
    2026-10-26: genuine same-photo matches scored 0.78-1.00 (mean
    0.975), with the one real outlier (0.78) visually confirmed as the
    same face, just small/blurry (naturally noisier embeddings). A
    wrong match would be expected to land far below this range
    (different-person similarity is typically well under 0.5-0.6 per
    this project's own established classification work).
  - Below GRAY_ZONE_SIM: rejected, logged, not written.

Faces WITHOUT a usable stored encoding (NULL or the
NON_DETECTED_FACE_ENCODING sentinel) skip the similarity gate entirely
-- the IOU match (and the DB's own face_kps_within_box_margin
constraint) is the only check; kps/det_score/LVFace embeddings are
saved either way, since there's no ground truth to verify against and
discarding the detection would lose real, if unverified, information.

--shard-index/--total-shards (same _shard_annotation() hash-based
partition this project already uses elsewhere) let several instances of
this command run concurrently against disjoint image sets.
"""
import signal
import time
from collections import defaultdict

import numpy as np
import torch
import torchvision.ops.boxes as bops
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import IntegrityError, transaction
from django.db.models import Q
from insightface.app import FaceAnalysis
from scipy.optimize import linear_sum_assignment

import common
import lvface
from face_extract_encode import FaceExtractor, _kps_matches_box
from face_manager.models import Face, Person
from django.db.models.expressions import RawSQL
from filepopulator.models import ImageFile, below_collection_floor
from pyramidal_detector import PyramidalDetector


def _image_shard_annotation(total_shards):
    """Same HASHTEXT-based hash-shard scheme as face_extract_encode.py's
    _shard_annotation(), but qualified to filepopulator_imagefile.id --
    that function's own RawSQL hardcodes face_manager_face.id, which is
    wrong here: this command's queryset is a queryset of ImageFile (the
    unit of sharding is the whole image, not any one of its faces), and
    using the Face-qualified version on a query that JOINS through Face
    sharded inconsistently -- an image with multiple matching Face rows
    could land in different shard buckets depending on which joined row
    got hashed, causing the same image to be (partially) processed by
    more than one shard. Found via a real test failure, not assumed."""
    return RawSQL('MOD(ABS(HASHTEXT(filepopulator_imagefile.id::text)), %s)', (total_shards,))


IOU_THRESH = 0.3
HIGH_CONFIDENCE_SIM = 0.9
GRAY_ZONE_SIM = 0.7
GRAY_ZONE_MIN_IOU = 0.8
GRAY_ZONE_MAX_RELATIVE_SIZE = 0.08

# Geometry-trust fallback tiers (2026-10-31, see CLAUDE.md) -- for a face
# whose STORED embedding doesn't match a fresh redetection well enough to
# clear GRAY_ZONE_SIM, a confident and UNAMBIGUOUS box match is still real
# evidence the box/kps themselves are correct, even though the identity
# question (is this really who it's declared/proposed as) is explicitly
# NOT this tier's job -- that stays gated separately by whichever real
# classification pass reads the resulting embedding later (e.g.
# reclassify_candidates --encoding=lvface_b). Validated empirically against
# real production data (301 real candidate faces): whole-image IOU>=0.5
# recovers 254/301 with zero ambiguous cases found; a second, cheap
# crop-based redetection fallback (same REDETECT_PAD_PCT/REDETECT_MIN_SCORE
# shape video_face_pipeline.py already uses) on the remainder recovers 15
# more, flags 3 as genuinely ambiguous (left untouched), and correctly
# finds nothing at all for the other ~30 (visually confirmed: some are
# real people who aren't a match, some are pure background/no-face-at-all
# -- never guessed at either way).
IOU_OVERRIDE_THRESH = 0.5
CROP_REDETECT_PAD_PCT = 0.6
CROP_REDETECT_MIN_SCORE = 0.5

PER_IMAGE_TIMEOUT_SECONDS = 60


class _ImageTimeout(Exception):
    pass


def _raise_image_timeout(signum, frame):
    raise _ImageTimeout(f"Image processing exceeded {PER_IMAGE_TIMEOUT_SECONDS}s")


def _l2norm(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _crop_redetect(app, img_numpy, box, pad_pct=CROP_REDETECT_PAD_PCT, min_score=CROP_REDETECT_MIN_SCORE):
    """Cheap fallback for an existing face whose stored box got no
    confident whole-image match at all -- crops a padded region around
    the box and redetects just within it, same shape as video_face_
    pipeline.py's own _redetect_in_crop(). Only searches the immediate
    neighborhood of the stored box, so (unlike a whole-image search) it
    structurally cannot accidentally match a detection far away in the
    image -- a real, confirmed failure mode of the whole-image IOU
    override (see IOU_OVERRIDE_THRESH's own comment).

    Returns (kps_in_full_image_coords, det_score, n_candidates_clearing_
    min_score) for the single best candidate, or None if nothing in the
    crop clears min_score at all. The caller decides what to do with
    n_candidates > 1 (ambiguous -- more than one real face nearby, don't
    guess which one is right)."""
    l, t, r, b = box
    w, h = r - l, b - t
    l2 = max(0, int(l - w * pad_pct))
    t2 = max(0, int(t - h * pad_pct))
    r2 = min(img_numpy.shape[1], int(r + w * pad_pct))
    b2 = min(img_numpy.shape[0], int(b + h * pad_pct))
    crop = img_numpy[t2:b2, l2:r2]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
    dets = app.get(crop)
    if not dets:
        return None
    n_candidates = sum(1 for d in dets if d.det_score >= min_score)
    if n_candidates == 0:
        return None
    best = max(dets, key=lambda d: d.det_score)
    kps_out = (np.asarray(best['kps'], dtype=float) + np.array([l2, t2])).reshape(-1).tolist()
    return kps_out, float(best.det_score), n_candidates


def _unresolved_faces_q(lvface_sizes, sentinel):
    """The exact negation of _face_is_done(), expressed as a Q object so
    an already-done face is never loaded out of the DB at all -- not
    fetched and then filtered in Python. 2026-10-31's mixed-image fix
    still pulled every face on an image (including fully-settled ones,
    with their full embedding arrays) just to immediately discard most
    of them; this pushes the same "done" check into the query itself."""
    if not lvface_sizes:
        return Q(kps__isnull=True)
    done_with_kps = Q(kps__isnull=False)
    done_without_kps = Q(kps__isnull=True)
    for size in lvface_sizes:
        field = f'face_encoding_lvface_{size.lower()}'
        done_with_kps &= Q(**{f'{field}__isnull': False})
        done_without_kps &= Q(**{field: sentinel})
    return ~(done_with_kps | done_without_kps)


def _face_is_done(face, lvface_sizes, sentinel):
    """A face is done either the normal way (real kps + every requested
    LVFace size populated) or via _mark_no_face_found()'s permanent
    "confirmed nothing here" marker (kps stays NULL forever by design,
    but every requested LVFace size is the sentinel). Without the second
    branch, a marked face would have no kps and get re-examined by every
    future run forever -- exactly the wasted-recompute problem marking
    it was meant to solve."""
    if not lvface_sizes:
        return face.kps is not None
    if face.kps is not None:
        return all(
            getattr(face, f'face_encoding_lvface_{size.lower()}') is not None
            for size in lvface_sizes
        )
    return all(
        list(getattr(face, f'face_encoding_lvface_{size.lower()}') or []) == sentinel
        for size in lvface_sizes
    )


def _build_full_detector():
    """Unlike backfill_det_score's detection-only detector, this one also
    needs the recognition module (to verify a match against the face's
    own stored embedding). It deliberately does NOT need genderage or
    either landmark module (landmark_3d_68/landmark_2d_106) -- nothing
    in this command reads detected_age/detected_gender or the 68/106-
    point landmark sets, only bbox/kps/det_score (from detection) and
    the embedding (from recognition). The plain default
    FaceAnalysis(name='buffalo_l') loads all 5 modules unconditionally;
    restricting allowed_modules avoids paying for the 3 unused ones on
    every single detection call, same real optimization already found
    and validated for backfill_det_score.py (see CLAUDE.md)."""
    face_analysis = FaceAnalysis(
        name='buffalo_l', providers=['CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition'],
    )
    face_analysis.prepare(ctx_id=-1)
    return PyramidalDetector(detector=face_analysis, iou_thresh=IOU_THRESH)


class Command(BaseCommand):
    help = (
        "Rigorous recovery backfill for the kps/box translation bug (see "
        "pyramidal_detector.py, CLAUDE.md) -- redetects each image with the "
        "now-fixed, production-identical PyramidalDetector, bipartite-matches "
        "every face on it against the fresh detections (scipy.optimize."
        "linear_sum_assignment, same pattern find_and_encode_faces() already "
        "uses), and only trusts a match once its fresh embedding is verified "
        "against the face's own stored face_encoding_512 (when one exists). "
        "Phase 1 (this command's default scope): faces with a real, already-"
        "trustworthy stored encoding -- the only population we have ground "
        "truth to verify against. A later pass can widen scope to the "
        "unlabeled/.ignore population once this is proven out."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None,
                             help='Restrict to the first N source images (for testing).')
        parser.add_argument('--shard-index', type=int, default=None)
        parser.add_argument('--total-shards', type=int, default=None)
        parser.add_argument('--lvface-sizes', type=str, default='S,B,L',
                             help="Comma-separated LVFace sizes to backfill for each verified face.")
        parser.add_argument('--person-ids', type=str, default=None,
                             help="Comma-separated Person ids. When given, restricts to source "
                                  "images that have at least one face declared to one of these "
                                  "people (in ADDITION to the usual trigger-population "
                                  "requirement) -- lets a run be redirected at specific "
                                  "under-represented people/negative-pool sentinels instead of "
                                  "working through the full trigger population in random order.")
        parser.add_argument('--per-person-target', type=int, default=None,
                             help="With --person-ids: once a (non-.ignore/.realignore) targeted "
                                  "person already has this many faces with face_encoding_lvface_b "
                                  "set, skip any image whose ONLY targeted faces are that "
                                  "already-satisfied person -- moves on to other under-"
                                  "represented people instead of exhaustively processing every "
                                  "image for whoever happens to come up first. Checked live "
                                  "against the DB (not an in-process cache), so it stays correct "
                                  "across multiple concurrently-running shards. .ignore/"
                                  "/.realignore are exempt from this cap (there's no such thing "
                                  "as 'enough' negative-pool faces at this investigation's scale) "
                                  "-- only --max-new-faces bounds those.")
        parser.add_argument('--max-new-faces', type=int, default=None,
                             help="Stop this invocation once it has newly backfilled this many "
                                  "faces (saved at least one LVFace size) -- a hard per-shard "
                                  "budget, for a bounded investigation sample rather than an "
                                  "exhaustive run.")
        parser.add_argument('--max-faces-per-image', type=int, default=None,
                             help="Skip any image with more than this many EXISTING faces "
                                  "outright, before even decoding -- avoids paying the real "
                                  "dominant cost (the matching/LVFace loop, which scales with "
                                  "face count) on crowd photos that contribute little per-"
                                  "targeted-person value anyway.")
        parser.add_argument('--max-image-pixels', type=int, default=None,
                             help="Skip any image whose width*height exceeds this, before even "
                                  "decoding -- catches giant/panoramic images that can hang "
                                  "app.get() ITSELF (not the trailing matching/LVFace loop), "
                                  "which no timeout can bound since that's a single "
                                  "uninterruptible C call. Independent of --max-faces-per-image, "
                                  "since such images can have very few cataloged faces.")
        parser.add_argument('--max-aspect-ratio', type=float, default=None,
                             help="Skip any image whose max(width,height)/min(width,height) "
                                  "exceeds this, before even decoding -- a panorama's extreme "
                                  "aspect ratio is a cleaner, more reliable signal than raw pixel "
                                  "count alone (which can't distinguish a genuine panorama from "
                                  "an ordinary large photo in the same megapixel range).")

    def _save_kps_and_lvface(self, face, kps, det_score, img_numpy, img_obj, lvface_sizes, stats, tier, dry_run):
        """Shared write path for every verification tier (high_confidence/
        gray_zone/iou_override/crop_redetect/unverified) -- saves kps/
        det_score (the DB's own face_kps_within_box_margin constraint is
        the real source of truth, never fatal if it rejects), then LVFace
        for each configured size if the face clears its collection's
        detection floor. face_encoding_512 is deliberately never touched
        here -- only this command's dedicated verification tiers decide
        when a fresh embedding is trustworthy enough to overwrite it, and
        none of them currently do.

        Returns True if at least one LVFace size was saved (the caller's
        own new_faces_this_run counter), False otherwise (including dry
        run, where nothing is written at all)."""
        if dry_run:
            stats[f'saved_{tier}'] += 1
            return False

        # _kps_matches_box() (checked by callers before reaching here) is
        # deliberately looser than the DB's own face_kps_within_box_margin
        # constraint (every one of the 5 points individually within a 25%
        # margin) -- a real match can clear the looser centroid check while
        # one individual landmark still fails the DB's tighter, per-point
        # requirement. transaction.atomic() here creates a real savepoint,
        # not just a Python-level try/except -- without it, a caught
        # IntegrityError still leaves Postgres considering the connection's
        # current transaction aborted until an actual ROLLBACK happens.
        try:
            with transaction.atomic():
                face.kps = FaceExtractor._flatten_kps(kps)
                face.det_score = det_score
                face.save(update_fields=['kps', 'det_score'])
        except IntegrityError as e:
            stats['db_constraint_rejected'] += 1
            self.stdout.write(f"  face {face.id}: DB rejected the recovered kps ({e}) -- skipping")
            return False

        stats[f'saved_{tier}'] += 1

        if below_collection_floor(img_obj.directory.collection, face.det_score, face.relative_size):
            stats['below_collection_floor_skip'] += 1
            return False

        any_lvface_saved = False
        for size in lvface_sizes:
            try:
                emb = lvface.embed_from_image_array(
                    img_numpy, face.kps, input_is_bgr=False, size=size,
                )
            except Exception as e:
                self.stdout.write(f"  LVFace-{size} failed for face {face.id}: {e}")
                continue
            if emb is None:
                continue
            field = f'face_encoding_lvface_{size.lower()}'
            face.set_embedding(field, emb.tolist())
            face.save(update_fields=[field, Face.EMBEDDING_FIELD_HV_MIRRORS[field]])
            any_lvface_saved = True
        return any_lvface_saved

    def _mark_no_face_found(self, face, lvface_sizes, stats, dry_run):
        """2026-10-31 (see CLAUDE.md): for a face where BOTH detection
        attempts came back genuinely empty -- the whole-image pass never
        matched it at all, and the cheap crop-based fallback found
        nothing either -- leaving its LVFace fields NULL forever means
        every future run re-examines it from scratch, paying the same
        whole-image + crop detection cost indefinitely for a face that's
        already been confirmed to have no findable geometry. Marking the
        sentinel (settings.NON_DETECTED_FACE_ENCODING, the same one
        face_encoding_512 already uses for this exact 'no match' concept)
        records "checked, nothing here" as a real, permanent fact --
        consistent with how every reader of this field already treats
        the sentinel (reclassify_candidates, the gallery-building code in
        this very file) as "not a usable embedding," not as "unknown."

        Deliberately does NOT touch kps -- there's no equivalent "no kps"
        sentinel in this codebase, and NULL already correctly means that.
        Only applied to crop_redetect_empty (not crop_redetect_ambiguous,
        where a real face likely IS there, just unclear which candidate
        -- marking that as 'confirmed nothing' would be actively wrong)."""
        if dry_run:
            stats['marked_no_face_found'] += 1
            return
        sentinel = list(settings.NON_DETECTED_FACE_ENCODING)
        update_fields = []
        for size in lvface_sizes:
            field = f'face_encoding_lvface_{size.lower()}'
            face.set_embedding(field, sentinel)
            update_fields += [field, Face.EMBEDDING_FIELD_HV_MIRRORS[field]]
        if update_fields:
            face.save(update_fields=update_fields)
            stats['marked_no_face_found'] += 1

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        shard_index = options['shard_index']
        total_shards = options['total_shards']
        lvface_sizes = [s.strip().upper() for s in options['lvface_sizes'].split(',') if s.strip()]
        person_ids = None
        if options['person_ids']:
            person_ids = [int(s.strip()) for s in options['person_ids'].split(',') if s.strip()]
        per_person_target = options['per_person_target']
        max_new_faces = options['max_new_faces']
        max_faces_per_image = options['max_faces_per_image']
        max_image_pixels = options['max_image_pixels']
        max_aspect_ratio = options['max_aspect_ratio']

        sentinel = list(settings.NON_DETECTED_FACE_ENCODING)
        unlimited_person_ids = set(
            Person.objects.filter(person_name__in=['.ignore', '.realignore']).values_list('id', flat=True)
        )

        # Deliberately queried at the Face level first, not
        # ImageFile.objects.filter(face__...).exclude(face__...) -- that
        # shape is the classic Django "spanning multi-valued
        # relationships" trap: filter() and exclude() across a reverse
        # FK don't combine on the SAME related row, so an image with one
        # real-encoded face and one sentinel-encoded face would have the
        # whole image excluded (the exclude() matches the OTHER face),
        # even though the real-encoded one should still trigger it.
        # Found via a real test failure, not assumed.
        base_trigger_filter = dict(face_encoding_512__isnull=False, source_image_file__isnull=False)
        if person_ids is not None:
            # Same "query at the Face level, not via a cross-relation
            # exclude()" discipline as above -- this is an additional
            # AND-restriction (a trigger image that ALSO has a face
            # declared OR PROPOSED to one of these people), not a
            # replacement for the base trigger requirement. Proposed
            # (poss_ident1) is included alongside declared_name so a
            # gallery-targeted run also covers the images a person's
            # own proposed-but-unconfirmed candidates sit on -- those
            # are exactly the faces a gallery-focused classification
            # pass cares about reclassifying with the new LVFace-B
            # encodings, not just the already-confirmed population.
            target_image_ids = Face.objects.filter(
                Q(declared_name_id__in=person_ids) | Q(poss_ident1_id__in=person_ids),
                source_image_file__isnull=False,
            ).values_list('source_image_file_id', flat=True).distinct()
            base_trigger_filter['source_image_file_id__in'] = target_image_ids

        trigger_image_ids = Face.objects.filter(**base_trigger_filter).exclude(
            face_encoding_512=sentinel
        ).values_list('source_image_file_id', flat=True).distinct()

        trigger_images = ImageFile.objects.filter(id__in=trigger_image_ids)

        if shard_index is not None and total_shards is not None:
            trigger_images = trigger_images.annotate(
                _shard=_image_shard_annotation(total_shards)
            ).filter(_shard=shard_index)

        image_ids = list(trigger_images.order_by('?').values_list('id', flat=True))
        if limit is not None:
            image_ids = image_ids[:limit]

        self.stdout.write(f"{len(image_ids)} source images to process {'(dry run)' if dry_run else ''}")

        app = _build_full_detector()

        stats = defaultdict(int)
        t0 = time.time()
        new_faces_this_run = 0

        for idx, img_id in enumerate(image_ids):
            if max_new_faces is not None and new_faces_this_run >= max_new_faces:
                self.stdout.write(f"  --max-new-faces={max_new_faces} reached -- stopping")
                break

            img_obj = ImageFile.objects.select_related('directory__collection').get(id=img_id)

            # Makes the command safely resumable/relaunchable: without
            # this, restarting would re-run detection on every already-
            # finished image, wasting all prior progress instead of
            # building on it. An image is "done" only once every face on
            # it has kps AND every CURRENTLY-requested LVFace size --
            # checking kps alone is a real bug if --lvface-sizes changes
            # between runs (found 2026-10-27: faces backfilled during an
            # earlier kps-only phase, --lvface-sizes '', would otherwise
            # be silently skipped forever once LVFace is re-enabled,
            # since they already have kps but never got the LVFace
            # fields this run actually wants). An empty lvface_sizes
            # list makes the query match every kps-null face, preserving
            # the original kps-only behavior unchanged for that mode.
            #
            # A face _mark_no_face_found() marked (kps NULL forever, but
            # LVFace set to the sentinel) also counts as done -- it has
            # no kps to wait for, by design, since nothing was ever found
            # to write kps FROM. Without this, every such face would be
            # re-examined (whole-image + crop redetect, both already
            # confirmed empty) on every single future run forever,
            # defeating the entire point of marking it.
            #
            # 2026-10-31: pushed the "done" check into the query itself
            # (_unresolved_faces_q) instead of loading every face on the
            # image and filtering in Python -- an already-done face is
            # never fetched at all now, not even its lightweight columns,
            # let alone the full embedding arrays the old `image_faces`
            # list pulled for every face on a mixed image regardless of
            # whether it still needed anything.
            unresolved_faces = list(
                Face.objects.filter(source_image_file=img_obj)
                .filter(_unresolved_faces_q(lvface_sizes, sentinel))
            )
            if not unresolved_faces:
                total_face_count = Face.objects.filter(source_image_file=img_obj).count()
                stats['already_done'] += total_face_count
                continue

            total_face_count = Face.objects.filter(source_image_file=img_obj).count()
            stats['already_done'] += total_face_count - len(unresolved_faces)

            # Skip crowd photos outright, before even decoding -- real
            # hang found 2026-10-27: a 710-face group photo makes the
            # matching/verification/LVFace loop the dominant cost (up to
            # faces*len(lvface_sizes) separate embedding calls), and such
            # images contribute little per-targeted-person value anyway
            # (mostly strangers/background people) relative to their
            # cost. Checked against the EXISTING face count (already
            # fetched, no extra query) rather than a fresh detection
            # count -- cheaper, and a strong proxy for "crowded" since
            # this project's own earlier backfills already found roughly
            # how many faces a given image has.
            if max_faces_per_image is not None and len(unresolved_faces) > max_faces_per_image:
                stats['too_many_faces_skip'] += len(unresolved_faces)
                continue

            # Skip giant panoramas outright, before even decoding -- a
            # SEPARATE real hang found 2026-10-27, distinct from the
            # crowd-photo one above: several images in this library are
            # extreme-aspect-ratio stitched panoramas (e.g. 22096x1088,
            # ~24MP) that hang app.get() itself (not the trailing
            # matching/LVFace loop) -- almost certainly the tiling
            # detector exploding on the extreme aspect ratio. Confirmed
            # via a stuck process showing 149 OS threads, no open image
            # file (already past decode), state R (still running) --
            # this is NOT caught by --max-faces-per-image, since these
            # panoramas have very few EXISTING cataloged faces (2-4),
            # not many. Checked against width*height (already stored, no
            # extra query) against a real empirical cutoff: 1,377 of
            # 79,077 images in this run's targeted scope exceed 25MP vs.
            # 9,326 exceeding 20MP -- 25MP cleanly separates the rare
            # panorama outliers from the much larger population of
            # ordinary 20-24MP photos (a common legitimate size for
            # modern phone/DSLR shots) that would be wasteful to exclude
            # too. Crucially, unlike the crowd-photo cap, this can't be
            # bounded by a timeout at all -- app.get() is a single C call
            # that SIGALRM cannot interrupt once inside it, confirmed by
            # three real, identical hangs in a row with zero "TIMEOUT on"
            # log lines despite PER_IMAGE_TIMEOUT_SECONDS firing.
            if max_image_pixels is not None and (img_obj.width or 0) * (img_obj.height or 0) > max_image_pixels:
                stats['too_many_pixels_skip'] += len(unresolved_faces)
                continue

            # Same panorama hang as above, but via ASPECT RATIO instead
            # of raw pixel count -- found necessary the SAME day:
            # --max-image-pixels=25000000 let the known 22096x1088
            # (~24.04MP) offender through a SECOND time, since its total
            # pixel count sits just under that threshold despite its
            # extreme 20.3:1 aspect ratio. Raw pixel count conflates a
            # genuine panorama with an ordinary large 20-24MP DSLR/phone
            # photo (both can land in the same MP range); aspect ratio
            # is the real distinguishing signal and separates MUCH more
            # cleanly in this run's targeted population: only 67 of
            # 79,077 images (0.08%) exceed a 2.5:1 ratio, vs. 883 at a
            # merely-wide 2:1 -- ordinary photos (even wide phone shots)
            # essentially never exceed 2:1, so 2.5 is a safe margin
            # above that with no observed false-exclusion risk. Kept
            # alongside --max-image-pixels (not a replacement) since a
            # genuinely enormous but normally-proportioned image could
            # still be slow for sheer size reasons, a different concern.
            if max_aspect_ratio is not None and img_obj.width and img_obj.height:
                aspect_ratio = max(img_obj.width, img_obj.height) / min(img_obj.width, img_obj.height)
                if aspect_ratio > max_aspect_ratio:
                    stats['too_wide_skip'] += len(unresolved_faces)
                    continue

            # Per-person quota: once every (non-sentinel) targeted person
            # on this image already has enough lvface-backfilled faces,
            # skip the whole image WITHOUT decoding/detecting -- the real
            # point of this check is to stop paying detection cost on
            # whoever happens to come up first in random order, and move
            # on to other under-represented people instead. Checked live
            # against the DB (not a cached in-process count) so this
            # stays correct even with several shards running at once.
            if per_person_target is not None and person_ids is not None:
                # declared_name_id over EVERY face on the image (not just
                # unresolved_faces) -- identity presence is unaffected by
                # per-face resolution state. A plain values_list query,
                # not a Python set comprehension over a loaded list, now
                # that image_faces no longer exists.
                relevant_pids = set(
                    Face.objects.filter(source_image_file=img_obj, declared_name_id__in=person_ids)
                    .exclude(declared_name_id__in=unlimited_person_ids)
                    .values_list('declared_name_id', flat=True)
                )
                if relevant_pids and all(
                    Face.objects.filter(
                        declared_name_id=pid, face_encoding_lvface_b__isnull=False,
                    ).count() >= per_person_target
                    for pid in relevant_pids
                ):
                    stats['quota_satisfied_skip'] += len(unresolved_faces)
                    continue

            # The timeout wraps the WHOLE per-image body, not just
            # decode+detect -- found via a real hang, not assumed. A
            # crowded photo (e.g. a 710-face group shot) makes the
            # trailing matching/verification/LVFace-embedding loop below
            # the real dominant cost (up to faces*len(lvface_sizes)
            # separate embedding calls), and that loop used to run
            # completely unguarded once the original, narrower alarm was
            # canceled right after app.get() returned -- a shard could
            # hang on one such image indefinitely. Re-raising _ImageTimeout
            # mid-loop is safe: every face whose kps/det_score/LVFace was
            # already saved before the alarm fired stays saved (each
            # write already committed via its own face.save() call), the
            # remaining faces on this image are simply left for a later
            # run to pick up (the already_done resumability check only
            # treats an image as finished once every one of its faces has
            # kps, so a partially-processed image is correctly retried).
            try:
                signal.signal(signal.SIGALRM, _raise_image_timeout)
                signal.alarm(PER_IMAGE_TIMEOUT_SECONDS)
                try:
                    img_numpy = common.open_img_oriented(img_obj.filename, as_numpy=True)
                    if img_numpy is None:
                        stats['decode_failed'] += len(unresolved_faces)
                        continue
                    detections = app.get(img_numpy)

                    if not detections:
                        stats['no_detections'] += len(unresolved_faces)
                        continue

                    existing_boxes = torch.tensor(
                        [[f.box_left, f.box_top, f.box_right, f.box_bottom] for f in unresolved_faces],
                        dtype=torch.float32,
                    )
                    detect_boxes = torch.tensor([d.bbox.tolist() for d in detections], dtype=torch.float32)
                    iou = bops.distance_box_iou(existing_boxes, detect_boxes).numpy()
                    rows, cols = linear_sum_assignment(-iou)

                    matched_rows = set()
                    for r, c in zip(rows, cols):
                        if float(iou[r, c]) < IOU_THRESH:
                            continue
                        matched_rows.add(int(r))
                        face = unresolved_faces[int(r)]
                        det = detections[int(c)]
                        this_iou = float(iou[r, c])

                        if not _kps_matches_box(det['kps'], face.box_left, face.box_top, face.box_right, face.box_bottom):
                            stats['geometry_rejected'] += 1
                            continue

                        has_trusted_encoding = (
                            face.face_encoding_512 is not None
                            and list(face.face_encoding_512) != sentinel
                        )

                        verified = True
                        tier = 'unverified'
                        if has_trusted_encoding:
                            fresh_emb = _l2norm(det['embedding'])
                            stored_emb = _l2norm(np.array(face.face_encoding_512, dtype=np.float32))
                            sim = float(np.dot(fresh_emb, stored_emb))
                            if sim >= HIGH_CONFIDENCE_SIM:
                                tier = 'high_confidence'
                            elif sim >= GRAY_ZONE_SIM and this_iou >= GRAY_ZONE_MIN_IOU and (
                                face.relative_size is not None and face.relative_size <= GRAY_ZONE_MAX_RELATIVE_SIZE
                            ):
                                tier = 'gray_zone'
                            elif this_iou >= IOU_OVERRIDE_THRESH and (
                                int((iou[r, :] >= IOU_OVERRIDE_THRESH).sum()) == 1
                            ):
                                # The stored embedding doesn't match well enough to
                                # trust via similarity, but the box match is
                                # confident AND unambiguous (no other detection in
                                # this image also clears the bar against this same
                                # existing box) -- real evidence the GEOMETRY is
                                # right even though identity isn't being judged
                                # here at all (see IOU_OVERRIDE_THRESH's own
                                # comment). face_encoding_512 is deliberately left
                                # untouched, same as every other tier -- only
                                # kps/det_score/LVFace get written.
                                tier = 'iou_override'
                            else:
                                verified = False
                                stats['similarity_rejected'] += 1

                        if not verified:
                            continue

                        if self._save_kps_and_lvface(
                            face, det['kps'], float(det.det_score), img_numpy, img_obj,
                            lvface_sizes, stats, tier, dry_run,
                        ):
                            new_faces_this_run += 1

                    unmatched_rows = [r for r in range(len(unresolved_faces)) if r not in matched_rows]
                    for r in unmatched_rows:
                        stats['no_match'] += 1
                        face = unresolved_faces[r]
                        has_trusted_encoding = (
                            face.face_encoding_512 is not None
                            and list(face.face_encoding_512) != sentinel
                        )
                        if not has_trusted_encoding:
                            continue  # nothing to recover a geometry match FOR
                        cr = _crop_redetect(
                            app, img_numpy,
                            (face.box_left, face.box_top, face.box_right, face.box_bottom),
                        )
                        if cr is None:
                            stats['crop_redetect_empty'] += 1
                            self._mark_no_face_found(face, lvface_sizes, stats, dry_run)
                            continue
                        kps_out, det_score_out, n_candidates = cr
                        if n_candidates > 1:
                            stats['crop_redetect_ambiguous'] += 1
                            continue
                        if not _kps_matches_box(
                            kps_out, face.box_left, face.box_top, face.box_right, face.box_bottom,
                        ):
                            stats['geometry_rejected'] += 1
                            continue
                        if self._save_kps_and_lvface(
                            face, kps_out, det_score_out, img_numpy, img_obj,
                            lvface_sizes, stats, 'crop_redetect', dry_run,
                        ):
                            new_faces_this_run += 1
                finally:
                    signal.alarm(0)
            except _ImageTimeout:
                self.stdout.write(f"  TIMEOUT on {img_obj.filename} ({len(unresolved_faces)} faces) -- skipping remainder")
                stats['timed_out'] += len(unresolved_faces)
                continue
            except Exception as e:
                self.stdout.write(f"  decode/detect error on {img_obj.filename}: {e}")
                stats['decode_failed'] += len(unresolved_faces)
                continue

            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = len(image_ids) - (idx + 1)
                eta_min = remaining / rate / 60 if rate else float('inf')
                self.stdout.write(
                    f"  ...{idx+1}/{len(image_ids)} images ({rate:.2f} img/s, ETA {eta_min:.0f}min, "
                    f"new_faces_this_run={new_faces_this_run}) -- "
                    + ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
                )

        elapsed = time.time() - t0
        self.stdout.write(
            f"DONE: {len(image_ids)} images, {elapsed:.0f}s, new_faces_this_run={new_faces_this_run}. "
            + ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        )
