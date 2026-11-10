# 2026-11-10, follow-up to scratch_validate_lvface_threshold(_margin).py:
# tests whether requiring insightface AND LVFace-B to agree cuts FPR
# further than either embedding space alone (and whether disagreement
# is informative enough to support a tiered confidence score, per the
# user's own design idea: both agree -> high confidence; disagree but
# the second embedding is "close" -> medium; otherwise low/reject).
#
# Methodology mirrors scratch_validate_lvface_threshold.py exactly (same
# people, same leave-one-out positive construction, same confirmed-
# .ignore negative pool) but computes BOTH embeddings' own-person p99
# for every query, not just LVFace-B's, so the two signals can be
# cross-tabulated directly on the same real faces.
#
# Run via: cat scratch_validate_lvface_agreement.py | docker exec -i
# picasa_api python manage.py shell

import random
import numpy as np
from django.conf import settings
from face_manager.models import Face, Person

random.seed(42)
np.random.seed(42)

IF_FIELD = 'face_encoding_512'
LV_FIELD = 'face_encoding_lvface_b'
IGNORE_NAMES = [settings.SOFT_IGNORE_NAME, '.realignore']
MIN_GALLERY = 10
SAMPLE_CAP = 300
POS_HOLDOUTS_PER_PERSON = 30
NEG_SAMPLE_SIZE = 1500

IF_THRESHOLD = 0.40   # a representative insightface bucket threshold (see assign_faces.py)
LV_THRESHOLDS = [0.22, 0.26, 0.30, 0.35]


def normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1
    return arr / n


print("Loading galleries (both embeddings)...")
people = (
    Person.objects.filter(face_declared__validated=True, face_declared__source_image_file__isnull=False)
    .exclude(person_name__in=IGNORE_NAMES)
    .exclude(person_name__startswith='_')
    .distinct()
)
# NOTE: insightface encodings are cleared for confirmed-ignore faces
# (clear_confirmed_ignore_face_encodings()), so the negative pool can
# only ever supply LVFace-B for the "negative" side of this test --
# that's fine, it's exactly the real-world asymmetry this investigation
# needs to confront: a currently-proposed junk face has NO insightface
# signal left to cross-check against once a human confirms it ignore.
# What we CAN test with both signals is the POSITIVE side (does
# requiring insightface to also agree reduce false NEGATIVES among real
# matches), and separately, whether LVFace-B's OWN disagreement
# structure (margin) is informative on its own -- already covered in
# scratch_validate_lvface_threshold_margin.py. This script focuses on:
# for genuinely-confirmed validated faces (galleries), does the
# insightface AND LVFace-B achieve similar effective per-face
# confidence, i.e. do the two spaces actually disagree much on the same
# real faces at all -- which is the real precondition for an agreement-
# gated or tiered design to add anything beyond LVFace-B alone.

lv_sentinel = list(settings.NON_DETECTED_FACE_ENCODING)

gallery_lv = {}
gallery_if = {}   # only populated where insightface wasn't cleared (rare on confirmed faces -- see below)
gallery_sizes = {}
for p in people:
    rows = list(
        Face.objects.filter(declared_name=p, validated=True, source_image_file__isnull=False)
        .exclude(**{f'{LV_FIELD}__isnull': True})
        .exclude(**{LV_FIELD: lv_sentinel})
        .values_list(LV_FIELD, flat=True)
    )
    if len(rows) < MIN_GALLERY:
        continue
    gallery_sizes[p.id] = len(rows)
    if len(rows) > SAMPLE_CAP:
        rows = random.sample(rows, SAMPLE_CAP)
    gallery_lv[p.id] = normed(np.array(rows, dtype=np.float32))

person_ids = list(gallery_lv.keys())
print(f"LVFace-B galleries loaded: {len(person_ids)} people")

# How many of these SAME validated faces still have a real insightface
# encoding at all? (Validated/confirmed-to-a-real-person faces are NOT
# cleared -- only confirmed .ignore/.realignore faces are. So this
# should be close to 100%, confirming the two signals really do coexist
# on most real, confirmed faces.)
if_sentinel = list(settings.NON_DETECTED_FACE_ENCODING)
total_validated = Face.objects.filter(
    declared_name__in=people, validated=True, source_image_file__isnull=False,
).exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel}).count()
has_if_too = Face.objects.filter(
    declared_name__in=people, validated=True, source_image_file__isnull=False,
).exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel}) \
 .exclude(**{f'{IF_FIELD}__isnull': True}).exclude(**{IF_FIELD: if_sentinel}).count()
print(f"Of {total_validated} validated LVFace-B faces, {has_if_too} ({100*has_if_too/total_validated:.1f}%) "
      f"ALSO still have a real insightface encoding -- confirms both signals coexist on confirmed faces.")

print("\nBuilding insightface galleries for the SAME people (for agreement cross-check)...")
for p in people:
    rows = list(
        Face.objects.filter(declared_name=p, validated=True, source_image_file__isnull=False)
        .exclude(**{f'{IF_FIELD}__isnull': True}).exclude(**{IF_FIELD: if_sentinel})
        .values_list(IF_FIELD, flat=True)
    )
    if len(rows) < MIN_GALLERY:
        continue
    if len(rows) > SAMPLE_CAP:
        rows = random.sample(rows, SAMPLE_CAP)
    gallery_if[p.id] = normed(np.array(rows, dtype=np.float32))

both_ids = [pid for pid in person_ids if pid in gallery_if]
print(f"People with BOTH a usable LVFace-B and insightface gallery: {len(both_ids)}")

print("\nBuilding positive leave-one-out queries for BOTH embeddings (paired per face)...")
# Pull paired (lvface_b, insightface) for the SAME face rows so the two
# scores are for literally the same photo, not independently resampled.
pos_pairs = []  # person_id, lv_own_p99, if_own_p99
for pid in both_ids:
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
        lv_p99 = float(np.percentile(lv_rest @ lv_arr[i], 99))
        if_p99 = float(np.percentile(if_rest @ if_arr[i], 99))
        pos_pairs.append((pid, lv_p99, if_p99))

print(f"Paired positive queries: {len(pos_pairs)}")

lv_pos = np.array([x[1] for x in pos_pairs])
if_pos = np.array([x[2] for x in pos_pairs])

print("\n=== Agreement structure on real (paired) positive queries ===")
corr = float(np.corrcoef(lv_pos, if_pos)[0, 1])
print(f"Correlation between LVFace-B p99 and insightface p99 (same face, same gallery): {corr:.3f}")

for lv_t in LV_THRESHOLDS:
    lv_pass = lv_pos >= lv_t
    if_pass = if_pos >= IF_THRESHOLD
    both = lv_pass & if_pass
    either = lv_pass | if_pass
    lv_only = lv_pass & ~if_pass
    if_only = if_pass & ~lv_pass
    neither = ~lv_pass & ~if_pass
    print(f"\nLVFace-B threshold={lv_t}, insightface threshold={IF_THRESHOLD}:")
    print(f"  both pass:    {both.mean():.4f}  (this is TPR under AND-gating)")
    print(f"  either pass:  {either.mean():.4f}  (this is TPR under OR-gating)")
    print(f"  LV only:      {lv_only.mean():.4f}")
    print(f"  IF only:      {if_only.mean():.4f}")
    print(f"  neither:      {neither.mean():.4f}  (both reject a REAL match -- a cost of AND-gating)")

print("\n=== Negative side: LVFace-B alone (insightface cleared for confirmed-ignore, "
      "so no paired cross-check is possible there -- see header comment) ===")
neg_rows = list(
    Face.objects.filter(declared_name__person_name__in=IGNORE_NAMES, source_image_file__isnull=False)
    .exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel})
    .values_list(LV_FIELD, flat=True)
)
random.shuffle(neg_rows)
neg_rows = neg_rows[:NEG_SAMPLE_SIZE]
neg_embs = normed(np.array(neg_rows, dtype=np.float32))
neg_top1 = np.full(neg_embs.shape[0], -1.0, dtype=np.float32)
for idx, pid in enumerate(person_ids):
    sims = neg_embs @ gallery_lv[pid].T
    p99 = np.percentile(sims, 99, axis=1)
    neg_top1 = np.maximum(neg_top1, p99)
for lv_t in LV_THRESHOLDS:
    print(f"  LVFace-B threshold={lv_t}: FPR={float((neg_top1 >= lv_t).mean()):.4f}")

print("\n=== Tiered-score design check (user's idea): does 'close on the second signal' "
      "carve out a useful medium tier among REAL matches that a strict AND-gate would reject? ===")
lv_t, if_t = 0.26, IF_THRESHOLD
both_pass = (lv_pos >= lv_t) & (if_pos >= if_t)
lv_only_fail_if = (lv_pos >= lv_t) & (if_pos < if_t)
if np.any(lv_only_fail_if):
    gap = if_t - if_pos[lv_only_fail_if]
    print(f"Real matches passing LVFace-B but failing insightface (n={lv_only_fail_if.sum()}): "
          f"insightface-shortfall mean={gap.mean():.3f}, median={np.median(gap):.3f}, "
          f"fraction within 0.05 of passing anyway={float((gap <= 0.05).mean()):.3f}")
print("(A real medium-confidence tier is viable if a meaningful fraction of these near-misses "
      "are 'close' -- i.e. the second signal's shortfall is small, not a wild disagreement.)")

print("\nDone.")
