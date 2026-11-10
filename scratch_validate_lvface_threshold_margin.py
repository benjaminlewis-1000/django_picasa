# 2026-11-10, follow-up to scratch_validate_lvface_threshold.py: tests
# whether a margin-above-best-competitor gate (and a det_score floor)
# can recover the FPR that a bare cosine threshold alone can't reach
# (10.5% FPR at threshold=0.30, see CLAUDE.md) -- the same combined-
# signal approach that fixed insightface's own magnet-gallery false-
# positive problem on 2026-09-21 (DET_SCORE_FLOOR + CANDIDATE_MARGIN_
# FLOOR), re-derived here for LVFace-B from scratch rather than assumed
# to transfer.
#
# Margin definition: for a positive (a真 true leave-one-out holdout of
# person P), margin = own_person_p99 - best_OTHER_person_p99 (how much
# more this face looks like its own person than its best alternative
# candidate). For a negative (confirmed .ignore), margin = top1_p99 -
# top2_p99 across all galleries (how confidently it latches onto ONE
# candidate vs. being an ambiguous near-tie across several) -- the
# natural equivalent when there's no "true person" to exclude.
#
# Run via: cat scratch_validate_lvface_threshold_margin.py | docker exec
# -i picasa_api python manage.py shell

import random
import numpy as np
from django.conf import settings
from face_manager.models import Face, Person

random.seed(42)
np.random.seed(42)

FIELD = 'face_encoding_lvface_b'
IGNORE_NAMES = [settings.SOFT_IGNORE_NAME, '.realignore']
MIN_GALLERY = 10
SAMPLE_CAP = 300
POS_HOLDOUTS_PER_PERSON = 30
NEG_SAMPLE_SIZE = 1500
BASE_THRESHOLDS = [0.26, 0.28, 0.30]
MARGIN_FLOORS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18]
DET_SCORE_FLOORS = [0.0, 0.5, 0.6, 0.7]


def normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1
    return arr / n


print("Loading galleries (embedding + det_score)...")
people = (
    Person.objects.filter(face_declared__validated=True, face_declared__source_image_file__isnull=False)
    .exclude(person_name__in=IGNORE_NAMES)
    .exclude(person_name__startswith='_')
    .distinct()
)
sentinel = list(settings.NON_DETECTED_FACE_ENCODING)

gallery_arr = {}      # person_id -> normed embedding array (capped sample)
gallery_det = {}      # person_id -> det_score array, same order
gallery_sizes = {}
for p in people:
    rows = list(
        Face.objects.filter(declared_name=p, validated=True, source_image_file__isnull=False)
        .exclude(**{f'{FIELD}__isnull': True})
        .exclude(**{FIELD: sentinel})
        .values_list(FIELD, 'det_score')
    )
    if len(rows) < MIN_GALLERY:
        continue
    gallery_sizes[p.id] = len(rows)
    if len(rows) > SAMPLE_CAP:
        rows = random.sample(rows, SAMPLE_CAP)
    embs, dets = zip(*rows)
    gallery_arr[p.id] = normed(np.array(embs, dtype=np.float32))
    gallery_det[p.id] = np.array([d if d is not None else 0.0 for d in dets], dtype=np.float32)

person_ids = list(gallery_arr.keys())
print(f"Galleries loaded: {len(person_ids)} people")

print("\nBuilding positive holdout queries (leave-one-out own_p99, det_score)...")
pos_queries = []  # dict: person_id, emb, det_score, own_p99
for pid in person_ids:
    arr = gallery_arr[pid]
    dets = gallery_det[pid]
    n = arr.shape[0]
    if n < 2:
        continue
    idxs = list(range(n))
    random.shuffle(idxs)
    idxs = idxs[:POS_HOLDOUTS_PER_PERSON]
    for i in idxs:
        rest = np.delete(arr, i, axis=0)
        if rest.shape[0] == 0:
            continue
        sims = rest @ arr[i]
        own_p99 = float(np.percentile(sims, 99))
        pos_queries.append({'person_id': pid, 'emb': arr[i], 'det_score': float(dets[i]), 'own_p99': own_p99})

print(f"Positive queries: {len(pos_queries)}")

print("\nSampling negative pool (confirmed .ignore/.realignore faces, with det_score)...")
neg_rows = list(
    Face.objects.filter(declared_name__person_name__in=IGNORE_NAMES, source_image_file__isnull=False)
    .exclude(**{f'{FIELD}__isnull': True})
    .exclude(**{FIELD: sentinel})
    .values_list(FIELD, 'det_score')
)
random.shuffle(neg_rows)
neg_rows = neg_rows[:NEG_SAMPLE_SIZE]
neg_embs_raw, neg_dets_raw = zip(*neg_rows)
neg_embs = normed(np.array(neg_embs_raw, dtype=np.float32))
neg_dets = np.array([d if d is not None else 0.0 for d in neg_dets_raw], dtype=np.float32)
print(f"Negative sample: {neg_embs.shape[0]}")

# ---- Positive pass: best_other_p99 (max over OTHER people's galleries) ----
print("\nComputing best-OTHER-person p99 for every positive query (excluding own person)...")
pos_embs = np.array([q['emb'] for q in pos_queries], dtype=np.float32)
pos_person = np.array([q['person_id'] for q in pos_queries])
pos_best_other = np.full(pos_embs.shape[0], -1.0, dtype=np.float32)
for idx, pid in enumerate(person_ids):
    arr = gallery_arr[pid]
    sims = pos_embs @ arr.T
    p99 = np.percentile(sims, 99, axis=1)
    mask = pos_person != pid   # only let OTHER people's galleries count toward best_other
    better = mask & (p99 > pos_best_other)
    pos_best_other[better] = p99[better]
    if (idx + 1) % 100 == 0:
        print(f"  ...{idx+1}/{len(person_ids)} galleries (positives pass)")

pos_own = np.array([q['own_p99'] for q in pos_queries], dtype=np.float32)
pos_det = np.array([q['det_score'] for q in pos_queries], dtype=np.float32)
pos_margin = pos_own - pos_best_other

# ---- Negative pass: top1/top2 p99 across all galleries ----
print("\nComputing top1/top2 p99 for every negative query...")
neg_top1 = np.full(neg_embs.shape[0], -1.0, dtype=np.float32)
neg_top2 = np.full(neg_embs.shape[0], -1.0, dtype=np.float32)
for idx, pid in enumerate(person_ids):
    arr = gallery_arr[pid]
    sims = neg_embs @ arr.T
    p99 = np.percentile(sims, 99, axis=1)
    new_top1 = np.maximum(neg_top1, p99)
    new_top2 = np.where(p99 > neg_top1, neg_top1, np.maximum(neg_top2, p99))
    neg_top1, neg_top2 = new_top1, new_top2
    if (idx + 1) % 100 == 0:
        print(f"  ...{idx+1}/{len(person_ids)} galleries (negatives pass)")

neg_margin = neg_top1 - np.where(neg_top2 < 0, 0.0, neg_top2)

print("\n=== Grid: base_threshold x margin_floor x det_score_floor ===")
print(f"{'thresh':>7} {'margin':>7} {'det':>5} {'TPR':>8} {'FPR':>8} {'cut/cost':>9}")
for base_t in BASE_THRESHOLDS:
    for mf in MARGIN_FLOORS:
        for df in DET_SCORE_FLOORS:
            pos_pass = (pos_own >= base_t) & (pos_margin >= mf) & (pos_det >= df)
            neg_pass = (neg_top1 >= base_t) & (neg_margin >= mf) & (neg_dets >= df)
            tpr = float(pos_pass.mean())
            fpr = float(neg_pass.mean())
            # baseline (margin=0, det=0) at this same base_t, for cut/cost
            baseline_fpr = float(((neg_top1 >= base_t)).mean())
            baseline_tpr = float(((pos_own >= base_t)).mean())
            fpr_cut = baseline_fpr - fpr
            tpr_cost = baseline_tpr - tpr
            ratio = (fpr_cut / tpr_cost) if tpr_cost > 1e-6 else float('inf')
            print(f"{base_t:7.2f} {mf:7.2f} {df:5.2f} {tpr:8.4f} {fpr:8.4f} {ratio:9.2f}")

print("\nDone.")
