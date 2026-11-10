# 2026-11-10, follow-up to scratch_validate_lvface_agreement*.py: builds
# and simulates an explicit THREE-TIER scheme (the user's own design):
#   HIGH   -- both insightface and LVFace-B clear their own threshold
#   MEDIUM -- one clears its threshold, the other is "close" (within
#             some close_shortfall of also clearing)
#   REJECT -- everything else
#
# Reuses the same paired-leave-one-out positive construction as
# scratch_validate_lvface_agreement.py, and the same kps-replay
# real-re-encode negative construction as
# scratch_validate_lvface_agreement_negatives.py, but with a LARGER
# negative sample this time (the user flagged the earlier 500-face
# negative estimate as "a bit muddy" -- true, ~1.2% standard error at
# n=500 isn't enough to cleanly separate tiers that differ by only a
# few points). Reports Wilson 95% CIs, not just point estimates, so
# "muddy" has an actual number attached to it.
#
# The expensive part (gallery loading + the real kps-replay re-encode
# of NEG_SAMPLE_SIZE negatives) is cached to CACHE_PATH after the first
# run -- every later invocation just reloads the raw per-face scores
# and re-sweeps CLOSE_SHORTFALLS in pure numpy, so finding the right
# "close" cutoff doesn't mean re-paying the real inference cost each
# time. Delete CACHE_PATH to force a fresh pull (e.g. after more of the
# LVFace-B backfill has landed).
#
# Nothing is ever written to the DB -- read-only throughout.
#
# Run via: cat scratch_validate_lvface_tiered.py | docker exec -i
# picasa_api python manage.py shell

import os
import pickle
import random
import numpy as np
import insightface
from django.conf import settings
from face_manager.models import Face, Person
from face_manager.face_extract_encode import FaceExtractor, _kps_matches_box
import common

random.seed(42)
np.random.seed(42)

LV_FIELD = 'face_encoding_lvface_b'
IF_FIELD = 'face_encoding_512'
IGNORE_NAMES = [settings.SOFT_IGNORE_NAME, '.realignore']
MIN_GALLERY = 10
SAMPLE_CAP = 300
POS_HOLDOUTS_PER_PERSON = 30
NEG_SAMPLE_SIZE = 2000   # up from 500 -- the earlier sample was "muddy", this halves the CI width

LV_THRESHOLD = 0.30
IF_THRESHOLD = 0.40
CLOSE_SHORTFALLS = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]   # swept, not a single fixed value
CACHE_PATH = '/tmp/lvface_tiered_scores.pkl'


def normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1
    return arr / n


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion -- better
    behaved than the normal approximation near 0 or 1, which matters
    here since some tiers are rare."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def shortfall(score, threshold):
    return np.maximum(0.0, threshold - score)


def tier_masks(a_pass, a_shortfall, b_pass, b_shortfall, close_shortfall):
    """a/b are symmetric -- call with (lv_pass, lv_shortfall, if_pass,
    if_shortfall, close_shortfall) and it returns boolean HIGH/MEDIUM/
    REJECT masks, where 'shortfall' is 0 if passing, else
    (threshold - score) i.e. how far short of passing a failing score
    is."""
    high = a_pass & b_pass
    medium = (~high) & (
        (a_pass & (b_shortfall <= close_shortfall)) |
        (b_pass & (a_shortfall <= close_shortfall))
    )
    reject = ~high & ~medium
    return high, medium, reject


if os.path.exists(CACHE_PATH):
    print(f"Loading cached scores from {CACHE_PATH} (skipping the expensive gallery/re-encode "
          f"steps -- delete this file to force a fresh pull)...")
    with open(CACHE_PATH, 'rb') as fh:
        cache = pickle.load(fh)
    pos_lv, pos_if = cache['pos_lv'], cache['pos_if']
    neg_lv_top1, neg_if_top1 = cache['neg_lv_top1'], cache['neg_if_top1']
    n_pos, n_neg = len(pos_lv), len(neg_lv_top1)
    print(f"Cached: {n_pos} positive queries, {n_neg} negative queries.")
else:
    print("Loading galleries (both embeddings, people with >= MIN_GALLERY on both)...")
    people = (
        Person.objects.filter(face_declared__validated=True, face_declared__source_image_file__isnull=False)
        .exclude(person_name__in=IGNORE_NAMES)
        .exclude(person_name__startswith='_')
        .distinct()
    )
    lv_sentinel = list(settings.NON_DETECTED_FACE_ENCODING)
    if_sentinel = list(settings.NON_DETECTED_FACE_ENCODING)

    gallery_lv, gallery_if = {}, {}
    for p in people:
        lv_rows = list(
            Face.objects.filter(declared_name=p, validated=True, source_image_file__isnull=False)
            .exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel})
            .values_list(LV_FIELD, flat=True)
        )
        if_rows = list(
            Face.objects.filter(declared_name=p, validated=True, source_image_file__isnull=False)
            .exclude(**{f'{IF_FIELD}__isnull': True}).exclude(**{IF_FIELD: if_sentinel})
            .values_list(IF_FIELD, flat=True)
        )
        if len(lv_rows) < MIN_GALLERY or len(if_rows) < MIN_GALLERY:
            continue
        if len(lv_rows) > SAMPLE_CAP:
            lv_rows = random.sample(lv_rows, SAMPLE_CAP)
        if len(if_rows) > SAMPLE_CAP:
            if_rows = random.sample(if_rows, SAMPLE_CAP)
        gallery_lv[p.id] = normed(np.array(lv_rows, dtype=np.float32))
        gallery_if[p.id] = normed(np.array(if_rows, dtype=np.float32))

    person_ids = list(gallery_lv.keys())
    print(f"Galleries loaded: {len(person_ids)} people")

    # -------------------------------------------------------------------
    # POSITIVES: paired leave-one-out, same face's own-person p99 under
    # both embeddings.
    # -------------------------------------------------------------------
    print("\nBuilding paired positive leave-one-out queries...")
    pos_lv, pos_if = [], []
    for pid in person_ids:
        rows = list(
            Face.objects.filter(declared_name_id=pid, validated=True, source_image_file__isnull=False)
            .exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel})
            .exclude(**{f'{IF_FIELD}__isnull': True}).exclude(**{IF_FIELD: if_sentinel})
            .values_list(LV_FIELD, IF_FIELD)
        )
        if len(rows) < MIN_GALLERY + 1:
            continue
        lv_arr = normed(np.array([r[0] for r in rows], dtype=np.float32))
        if_arr = normed(np.array([r[1] for r in rows], dtype=np.float32))
        n = lv_arr.shape[0]
        idxs = list(range(n))
        random.shuffle(idxs)
        idxs = idxs[:POS_HOLDOUTS_PER_PERSON]
        for i in idxs:
            lv_rest = np.delete(lv_arr, i, axis=0)
            if_rest = np.delete(if_arr, i, axis=0)
            pos_lv.append(float(np.percentile(lv_rest @ lv_arr[i], 99)))
            pos_if.append(float(np.percentile(if_rest @ if_arr[i], 99)))

    pos_lv = np.array(pos_lv)
    pos_if = np.array(pos_if)
    print(f"Paired positive queries: {len(pos_lv)}")

    # -------------------------------------------------------------------
    # NEGATIVES: real kps-replay re-encode, larger sample this time.
    # -------------------------------------------------------------------
    print(f"\nSampling up to {NEG_SAMPLE_SIZE} confirmed-ignore faces with usable kps + LVFace-B...")
    candidates = list(
        Face.objects.filter(
            declared_name__person_name__in=IGNORE_NAMES, source_image_file__isnull=False,
            kps__isnull=False,
        )
        .exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel})
        .select_related('source_image_file')
        .order_by('?')[:int(NEG_SAMPLE_SIZE * 1.3)]
    )
    print(f"Candidate pool pulled: {len(candidates)}")

    print("\nBuilding detector (recognition model only, for kps-replay)...")
    extractor = FaceExtractor()
    app = extractor._build_detector()
    rec_model = app.app.models['recognition']

    neg_lv_raw, neg_if_raw = [], []
    decode_failed = kps_mismatch = 0
    for i, face in enumerate(candidates):
        if len(neg_lv_raw) >= NEG_SAMPLE_SIZE:
            break
        if not _kps_matches_box(face.kps, face.box_left, face.box_top, face.box_right, face.box_bottom):
            kps_mismatch += 1
            continue
        img_obj = face.source_image_file
        try:
            img = common.open_img_oriented(
                img_obj.filename, as_numpy=True,
                extra_rotation_degrees=getattr(img_obj, 'confirmed_rotation_override_degrees', 0),
            )
        except Exception:
            decode_failed += 1
            continue
        if img is None or len(img.shape) != 3:
            decode_failed += 1
            continue
        kps_arr = np.array(face.kps, dtype=np.float32).reshape(5, 2)
        insight_face_obj = insightface.app.common.Face(kps=kps_arr)
        embedding = rec_model.get(img, insight_face_obj)
        neg_lv_raw.append(np.array(face.face_encoding_lvface_b, dtype=np.float32))
        neg_if_raw.append(np.array(embedding, dtype=np.float32))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(candidates)} candidates tried, {len(neg_lv_raw)} usable so far "
                  f"(decode_failed={decode_failed}, kps_mismatch={kps_mismatch})")

    print(f"\nUsable negative pairs: {len(neg_lv_raw)} (decode_failed={decode_failed}, "
          f"kps_mismatch={kps_mismatch})")
    neg_lv = normed(np.array(neg_lv_raw, dtype=np.float32))
    neg_if = normed(np.array(neg_if_raw, dtype=np.float32))

    neg_lv_top1 = np.full(neg_lv.shape[0], -1.0, dtype=np.float32)
    neg_if_top1 = np.full(neg_if.shape[0], -1.0, dtype=np.float32)
    for pid in person_ids:
        neg_lv_top1 = np.maximum(neg_lv_top1, np.percentile(neg_lv @ gallery_lv[pid].T, 99, axis=1))
        neg_if_top1 = np.maximum(neg_if_top1, np.percentile(neg_if @ gallery_if[pid].T, 99, axis=1))

    n_pos, n_neg = len(pos_lv), len(neg_lv_top1)

    with open(CACHE_PATH, 'wb') as fh:
        pickle.dump({
            'pos_lv': pos_lv, 'pos_if': pos_if,
            'neg_lv_top1': neg_lv_top1, 'neg_if_top1': neg_if_top1,
        }, fh)
    print(f"Cached raw scores to {CACHE_PATH} for future shortfall sweeps.")

# ---------------------------------------------------------------------
# Tier assignment, swept across CLOSE_SHORTFALLS -- this part is cheap
# pure numpy, re-run freely without touching the cache.
# ---------------------------------------------------------------------
pos_lv_pass = pos_lv >= LV_THRESHOLD
pos_if_pass = pos_if >= IF_THRESHOLD
pos_lv_short = shortfall(pos_lv, LV_THRESHOLD)
pos_if_short = shortfall(pos_if, IF_THRESHOLD)

neg_lv_pass = neg_lv_top1 >= LV_THRESHOLD
neg_if_pass = neg_if_top1 >= IF_THRESHOLD
neg_lv_short = shortfall(neg_lv_top1, LV_THRESHOLD)
neg_if_short = shortfall(neg_if_top1, IF_THRESHOLD)

print(f"\n=== Three-tier sweep (LV_thresh={LV_THRESHOLD}, IF_thresh={IF_THRESHOLD}) ===")
print(f"Positives: n={n_pos}   Negatives: n={n_neg} (real kps-replay re-encode)")

and_tpr = (pos_lv_pass & pos_if_pass).sum() / n_pos
and_fpr = (neg_lv_pass & neg_if_pass).sum() / n_neg
print(f"\nBaseline AND-gate (close_shortfall=0, i.e. no medium tier): TPR={and_tpr:.4f} FPR={and_fpr:.4f}")

print(f"\n{'shortfall':>10} {'HIGH+MED TPR':>13} {'CI':>18} {'HIGH+MED FPR':>13} {'CI':>18} "
      f"{'dTPR':>7} {'dFPR':>7} {'ratio':>7}")
for cs in CLOSE_SHORTFALLS:
    pos_high, pos_medium, pos_reject = tier_masks(pos_lv_pass, pos_lv_short, pos_if_pass, pos_if_short, cs)
    neg_high, neg_medium, neg_reject = tier_masks(neg_lv_pass, neg_lv_short, neg_if_pass, neg_if_short, cs)

    combined_p = pos_high | pos_medium
    combined_n = neg_high | neg_medium
    tpr = combined_p.sum() / n_pos
    fpr = combined_n.sum() / n_neg
    tpr_ci = wilson_ci(int(combined_p.sum()), n_pos)
    fpr_ci = wilson_ci(int(combined_n.sum()), n_neg)
    d_tpr = tpr - and_tpr
    d_fpr = fpr - and_fpr
    ratio = (d_tpr / d_fpr) if d_fpr > 1e-9 else float('inf')
    print(f"{cs:10.2f} {tpr:13.4f} [{tpr_ci[0]:.4f},{tpr_ci[1]:.4f}] {fpr:13.4f} "
          f"[{fpr_ci[0]:.4f},{fpr_ci[1]:.4f}] {d_tpr:+7.4f} {d_fpr:+7.4f} {ratio:7.2f}")

print("\n(ratio = TPR recovered per unit of FPR added, relative to the plain AND-gate baseline -- "
      "ratio > 1 means the medium tier is buying TPR more cheaply than it costs in FPR, "
      "ratio < 1 means it's a bad trade.)")

print("\n=== For reference: plain AND-gate / OR-gate / LVFace-B alone ===")
for name, pmask, nmask in [
    ('AND (both pass)', pos_lv_pass & pos_if_pass, neg_lv_pass & neg_if_pass),
    ('OR (either pass)', pos_lv_pass | pos_if_pass, neg_lv_pass | neg_if_pass),
    ('LV alone', pos_lv_pass, neg_lv_pass),
]:
    tpr = pmask.sum() / n_pos
    fpr = nmask.sum() / n_neg
    tpr_ci = wilson_ci(int(pmask.sum()), n_pos)
    fpr_ci = wilson_ci(int(nmask.sum()), n_neg)
    print(f"{name:>16} TPR={tpr:.4f} [{tpr_ci[0]:.4f},{tpr_ci[1]:.4f}]  "
          f"FPR={fpr:.4f} [{fpr_ci[0]:.4f},{fpr_ci[1]:.4f}]")

print("\nDone.")
