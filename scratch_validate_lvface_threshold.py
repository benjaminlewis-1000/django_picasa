# 2026-11-10: real, read-only (no writes) LVFace-B threshold validation,
# re-run at current scale (461 validated galleries vs. the 152 the
# 2026-10-31 investigation used) per the user's request to "do one more
# run of finding the right parameters" before migrating classify_
# unassigned() to LVFace-B. See CLAUDE.md's "LVFace-B threshold
# re-validated at scale" entry for the real numbers and what they mean.
#
# Key methodological correction from the 2026-10-31 investigation: FPR
# here is measured as "does this negative face clear the bar against ANY
# of the 461 galleries" (max over all competing galleries), not against
# one gallery in isolation -- that's the real exposure a single global
# threshold faces in production, where classify_unassigned() checks a
# face against everyone, not one person. This is why the FPR found here
# (10.5% at threshold=0.30) is far worse than the earlier "<0.4%" figure.
#
# Run via: cat scratch_validate_lvface_threshold.py | docker exec -i
# picasa_api python manage.py shell
#
# To re-run against the halfvec mirror instead of the float32 field
# (confirms precision loss doesn't matter -- it didn't, 2026-11-10):
#   sed 's/face_encoding_lvface_b/face_encoding_lvface_b_hv/' \
#     scratch_validate_lvface_threshold.py | docker exec -i picasa_api \
#     python manage.py shell

import random
import numpy as np
from django.conf import settings
from face_manager.models import Face, Person

random.seed(42)
np.random.seed(42)

FIELD = 'face_encoding_lvface_b'
IGNORE_NAMES = [settings.SOFT_IGNORE_NAME, '.realignore']
MIN_GALLERY = 10
SAMPLE_CAP = 300          # cap per-gallery embeddings (cost bound, same pattern as COHESION_SAMPLE_CAP)
POS_HOLDOUTS_PER_PERSON = 50
NEG_SAMPLE_SIZE = 1500
THRESHOLDS = [0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45]


def normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1
    return arr / n


print("Loading galleries...")
people = (
    Person.objects.filter(face_declared__validated=True, face_declared__source_image_file__isnull=False)
    .exclude(person_name__in=IGNORE_NAMES)
    .exclude(person_name__startswith='_')
    .distinct()
)
sentinel = list(settings.NON_DETECTED_FACE_ENCODING)

galleries = {}  # person_id -> normed embedding array (capped sample)
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
      f"{min(gallery_sizes.values())}-{max(gallery_sizes.values())}")

print("\nComputing leave-one-out positive p99 scores...")
pos_p99s = []  # (person_id, gallery_size_bucket_source_size, p99)
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

print("\nSampling negative pool (confirmed .ignore/.realignore faces with LVFace-B)...")
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

print("\nComputing per-negative max-over-all-galleries p99 (the real production exposure -- "
      "could this junk face get PROPOSED to ANY person)...")
neg_max_p99 = np.zeros(neg_embs.shape[0], dtype=np.float32)
neg_best_person = [None] * neg_embs.shape[0]
for idx, (pid, arr) in enumerate(galleries.items()):
    sims = neg_embs @ arr.T  # (N_neg, gallery_size)
    p99 = np.percentile(sims, 99, axis=1)
    better = p99 > neg_max_p99
    neg_max_p99[better] = p99[better]
    for j in np.where(better)[0]:
        neg_best_person[j] = pid
    if (idx + 1) % 100 == 0:
        print(f"  ...{idx+1}/{len(galleries)} galleries checked")

print("\n=== Threshold sweep ===")
pos_arr = np.array([p for (_, _, p) in pos_p99s])
print(f"{'thresh':>8} {'TPR':>8} {'FPR':>8} {'pos_n':>7} {'neg_n':>7}")
for t in THRESHOLDS:
    tpr = float((pos_arr >= t).mean())
    fpr = float((neg_max_p99 >= t).mean())
    print(f"{t:8.2f} {tpr:8.4f} {fpr:8.4f} {len(pos_arr):7d} {len(neg_max_p99):7d}")

print("\n=== TPR by gallery-size bucket at threshold=0.30 (checking bucketing still unneeded) ===")
buckets = [(10, 50), (50, 200), (200, 1000), (1000, 100000)]
for lo, hi in buckets:
    bucket_p99s = [p for (pid, size, p) in pos_p99s if lo <= size < hi]
    if not bucket_p99s:
        continue
    arr = np.array(bucket_p99s)
    tpr_030 = float((arr >= 0.30).mean())
    print(f"  size [{lo},{hi}): n={len(arr)}, TPR@0.30={tpr_030:.4f}, mean_p99={arr.mean():.4f}")

print("\nDone.")
