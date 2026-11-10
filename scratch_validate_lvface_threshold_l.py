# 2026-11-10: LVFace-L threshold validation, same leave-one-out
# methodology as scratch_validate_lvface_threshold.py (which validated
# LVFace-B: TPR 97.3%/FPR 10.5% @ threshold=0.30, 461 galleries), run
# against a real sample of LVFace-L encodings specifically written to
# the DB for this purpose (backfill_lvface_targeted --lvface-sizes=L
# --shard-index=0 --total-shards=22, 25,437 faces, plus ~40k already
# present from the 2026-10-26 one-off investigation -- NOT a throwaway
# computation, these are real stored embeddings).
#
# Per the user's explicit framing this time: (1) a lower TPR is
# acceptable in exchange for a much lower FPR -- the real goal is
# cutting down "magnet gallery" junk (e.g. Leslie Williams), not
# maximizing recall from the .ignore bucket; (2) false positives are
# much more tolerable if they're LOW confidence (tail of a review list)
# than if they're HIGH confidence (indistinguishable from a real
# match). So this script reports, for each candidate threshold, not
# just TPR/FPR but the score distribution of the negatives that pass
# -- do they cluster right at the threshold (low-confidence, tail-end,
# acceptable) or extend up toward high scores (dangerous)?
#
# Run via: cat scratch_validate_lvface_threshold_l.py | docker exec -i
# picasa_api python manage.py shell

import random
import numpy as np
from django.conf import settings
from face_manager.models import Face, Person

random.seed(42)
np.random.seed(42)

FIELD = 'face_encoding_lvface_l'
IGNORE_NAMES = [settings.SOFT_IGNORE_NAME, '.realignore']
MIN_GALLERY = 10
SAMPLE_CAP = 300
POS_HOLDOUTS_PER_PERSON = 50
NEG_SAMPLE_SIZE = 1500
THRESHOLDS = [0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60]


def normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1
    return arr / n


print("Loading LVFace-L galleries...")
people = (
    Person.objects.filter(face_declared__validated=True, face_declared__source_image_file__isnull=False)
    .exclude(person_name__in=IGNORE_NAMES)
    .exclude(person_name__startswith='_')
    .distinct()
)
sentinel = list(settings.NON_DETECTED_FACE_ENCODING)

galleries = {}
gallery_sizes = {}
for p in people:
    embs = list(
        Face.objects.filter(declared_name=p, validated=True, source_image_file__isnull=False)
        .exclude(**{f'{FIELD}__isnull': True})
        .exclude(**{FIELD: sentinel})
        .values_list(FIELD, flat=True)
    )
    if len(embs) < MIN_GALLERY:
        continue
    gallery_sizes[p.id] = len(embs)
    if len(embs) > SAMPLE_CAP:
        embs = random.sample(embs, SAMPLE_CAP)
    galleries[p.id] = normed(np.array(embs, dtype=np.float32))

print(f"Galleries loaded: {len(galleries)} people, size range "
      f"{min(gallery_sizes.values())}-{max(gallery_sizes.values())}" if galleries else "Galleries loaded: 0")

print("\nComputing leave-one-out positive p99 scores...")
pos_p99s = []
for pid, arr in galleries.items():
    n = arr.shape[0]
    if n < 2:
        continue
    holdout_idx = list(range(n))
    random.shuffle(holdout_idx)
    holdout_idx = holdout_idx[:POS_HOLDOUTS_PER_PERSON]
    for i in holdout_idx:
        rest = np.delete(arr, i, axis=0)
        if rest.shape[0] == 0:
            continue
        sims = rest @ arr[i]
        p99 = float(np.percentile(sims, 99))
        pos_p99s.append((pid, gallery_sizes[pid], p99))

print(f"Positive trials: {len(pos_p99s)}")

print("\nSampling negative pool (confirmed .ignore/.realignore faces with LVFace-L)...")
neg_ids = list(
    Face.objects.filter(declared_name__person_name__in=IGNORE_NAMES, source_image_file__isnull=False)
    .exclude(**{f'{FIELD}__isnull': True})
    .exclude(**{FIELD: sentinel})
    .values_list('id', flat=True)
)
random.shuffle(neg_ids)
neg_ids = neg_ids[:NEG_SAMPLE_SIZE]
neg_embs = normed(np.array(
    list(Face.objects.filter(id__in=neg_ids).values_list(FIELD, flat=True)), dtype=np.float32
))
print(f"Negative sample: {neg_embs.shape[0]}")

print("\nComputing per-negative top1/top2-over-all-galleries p99 "
      "(top1 = the real production exposure; top2 lets us compute margin too)...")
neg_top1 = np.full(neg_embs.shape[0], -1.0, dtype=np.float32)
neg_top2 = np.full(neg_embs.shape[0], -1.0, dtype=np.float32)
neg_best_person = [None] * neg_embs.shape[0]
for idx, (pid, arr) in enumerate(galleries.items()):
    sims = neg_embs @ arr.T
    p99 = np.percentile(sims, 99, axis=1)
    new_top1 = np.maximum(neg_top1, p99)
    new_top2 = np.where(p99 > neg_top1, neg_top1, np.maximum(neg_top2, p99))
    better = p99 > neg_top1
    for j in np.where(better)[0]:
        neg_best_person[j] = pid
    neg_top1, neg_top2 = new_top1, new_top2
    if (idx + 1) % 50 == 0:
        print(f"  ...{idx+1}/{len(galleries)} galleries checked")

neg_margin = neg_top1 - np.where(neg_top2 < 0, 0.0, neg_top2)

print("\n=== Threshold sweep ===")
pos_arr = np.array([p for (_, _, p) in pos_p99s])
print(f"{'thresh':>8} {'TPR':>8} {'FPR':>8} {'pos_n':>7} {'neg_n':>7}")
for t in THRESHOLDS:
    tpr = float((pos_arr >= t).mean())
    fpr = float((neg_top1 >= t).mean())
    print(f"{t:8.2f} {tpr:8.4f} {fpr:8.4f} {len(pos_arr):7d} {len(neg_top1):7d}")

print("\n=== TPR by gallery-size bucket at threshold=0.30 (checking bucketing need) ===")
buckets = [(10, 50), (50, 200), (200, 1000), (1000, 100000)]
for lo, hi in buckets:
    bucket_p99s = [p for (pid, size, p) in pos_p99s if lo <= size < hi]
    if not bucket_p99s:
        continue
    arr = np.array(bucket_p99s)
    tpr_030 = float((arr >= 0.30).mean())
    print(f"  size [{lo},{hi}): n={len(arr)}, TPR@0.30={tpr_030:.4f}, mean_p99={arr.mean():.4f}")

print("\n=== False-positive score/margin distribution (per the user's framing: "
      "low-confidence tail-end FPs are tolerable, high-confidence ones are not) ===")
# Computed once, used below -- the real-match reference distribution the
# user actually cares about comparing against (not an arbitrary
# threshold+0.10 offset): does a false positive's score look like a
# TYPICAL real match (dangerous, would sort near the top of a review
# list alongside genuine matches) or like a weak one (sits near the
# bottom, easy to skip)?
_ref_pos_scores = []
for pid, arr in galleries.items():
    n = arr.shape[0]
    if n < 2:
        continue
    holdout_idx = list(range(n))
    random.shuffle(holdout_idx)
    holdout_idx = holdout_idx[:10]
    for i in holdout_idx:
        rest = np.delete(arr, i, axis=0)
        if rest.shape[0] == 0:
            continue
        _ref_pos_scores.append(float(np.percentile(rest @ arr[i], 99)))
_ref_pos_scores = np.array(_ref_pos_scores)
_ref_median = float(np.median(_ref_pos_scores))
_ref_p25 = float(np.percentile(_ref_pos_scores, 25))
print(f"(Real-match reference: median={_ref_median:.3f}, p25={_ref_p25:.3f}, n={len(_ref_pos_scores)})")

for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    passing = neg_top1 >= t
    n_pass = int(passing.sum())
    if n_pass == 0:
        print(f"\nthreshold={t}: 0 negatives pass -- nothing to characterize.")
        continue
    scores = neg_top1[passing]
    margins = neg_margin[passing]
    pct = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
    margin_pct = np.percentile(margins, [0, 10, 25, 50, 75, 90, 100])
    # How many of the passing FPs are "high confidence" -- defined two
    # ways: (a) comfortably above threshold (score >= t + 0.10, not just
    # barely clearing the bar), and (b) a large margin over the runner-up
    # (>= 0.10, meaning it confidently locked onto ONE specific wrong
    # person rather than being an ambiguous near-tie across several).
    high_conf_score = float((scores >= t + 0.10).mean())
    high_conf_margin = float((margins >= 0.10).mean())
    above_ref_median = float((scores >= _ref_median).mean())
    above_ref_p25 = float((scores >= _ref_p25).mean())
    print(f"\nthreshold={t}: {n_pass} negatives pass (FPR={n_pass/len(neg_top1):.4f})")
    print(f"  score distribution   [min,p10,p25,med,p75,p90,max]: "
          + ", ".join(f"{v:.3f}" for v in pct))
    print(f"  margin distribution  [min,p10,p25,med,p75,p90,max]: "
          + ", ".join(f"{v:.3f}" for v in margin_pct))
    print(f"  fraction with score >= threshold+0.10 (comfortably clear, not tail-end): "
          f"{high_conf_score:.3f}")
    print(f"  fraction with margin >= 0.10 (confidently locked onto ONE wrong person): "
          f"{high_conf_margin:.3f}")
    print(f"  fraction scoring >= real-match MEDIAN ({_ref_median:.3f}) -- looks like a "
          f"typical genuine match, the dangerous case: {above_ref_median:.3f}")
    print(f"  fraction scoring >= real-match P25 ({_ref_p25:.3f}) -- would NOT sort to the "
          f"bottom of a list alongside real matches: {above_ref_p25:.3f}")

print("\n=== For comparison: same score/margin breakdown on the POSITIVE (real match) side, "
      "so we know what a 'real high-confidence match' looks like ===")
pos_margin_list = []
pos_embs_by_idx = []
for pid, arr in galleries.items():
    n = arr.shape[0]
    if n < 2:
        continue
    holdout_idx = list(range(n))
    random.shuffle(holdout_idx)
    holdout_idx = holdout_idx[:10]  # smaller, just for a representative margin sample
    for i in holdout_idx:
        rest = np.delete(arr, i, axis=0)
        if rest.shape[0] == 0:
            continue
        own_p99 = float(np.percentile(rest @ arr[i], 99))
        pos_margin_list.append((pid, own_p99))

print(f"(Positive-side own-person p99 reference distribution, n={len(pos_margin_list)}): "
      + ", ".join(f"{v:.3f}" for v in np.percentile([p for _, p in pos_margin_list], [0, 10, 25, 50, 75, 90, 100])))

print("\nDone.")
