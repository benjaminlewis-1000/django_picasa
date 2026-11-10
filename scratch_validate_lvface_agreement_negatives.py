# 2026-11-10, closes the one real gap in scratch_validate_lvface_
# agreement.py: that script could only test agreement-gating's effect
# on TPR (real matches), because insightface's own face_encoding_512
# is already cleared for every confirmed-.ignore/.realignore face
# (clear_confirmed_ignore_face_encodings()) -- there's no stored
# insightface signal left to cross-check the negative side against.
#
# Face.kps is deliberately left untouched for exactly this situation
# (see clear_confirmed_ignore_face_encodings()'s own docstring) -- this
# script uses the SAME kps-replay mechanism FaceExtractor.
# reencode_missing_faces() already uses in production
# (rec_model.get(img, insightface.app.common.Face(kps=kps_arr))) to
# reconstruct a REAL insightface embedding for a bounded sample of
# confirmed-ignore faces, entirely in memory -- nothing is written back
# to the DB. This lets agreement-gating's FPR impact actually be
# measured on real negative data for the first time.
#
# Deliberately bounded (NEG_SAMPLE_SIZE below) to avoid competing hard
# with the overnight backfill_verified_kps / train_rotation_classifier
# jobs for CPU -- this does real image decode + real recognition
# inference per face, unlike the pure-arithmetic scripts before it.
#
# Run via: cat scratch_validate_lvface_agreement_negatives.py | docker
# exec -i picasa_api python manage.py shell

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
IGNORE_NAMES = [settings.SOFT_IGNORE_NAME, '.realignore']
MIN_GALLERY = 10
SAMPLE_CAP = 300
NEG_SAMPLE_SIZE = 500   # bounded -- real image decode + real inference per face, not just arithmetic
LV_THRESHOLDS = [0.22, 0.26, 0.30, 0.35]
IF_THRESHOLD = 0.40


def normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1
    return arr / n


print("Loading LVFace-B galleries...")
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
        .exclude(face_encoding_512__isnull=True).exclude(face_encoding_512=if_sentinel)
        .values_list('face_encoding_512', flat=True)
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
print(f"Galleries loaded (both signals): {len(person_ids)} people")

print(f"\nSampling up to {NEG_SAMPLE_SIZE} confirmed-ignore faces with usable kps + LVFace-B + a real image...")
candidates = list(
    Face.objects.filter(
        declared_name__person_name__in=IGNORE_NAMES, source_image_file__isnull=False,
        kps__isnull=False,
    )
    .exclude(**{f'{LV_FIELD}__isnull': True}).exclude(**{LV_FIELD: lv_sentinel})
    .select_related('source_image_file')
    .order_by('?')[:NEG_SAMPLE_SIZE * 2]  # oversample; some will fail decode/kps-mismatch
)
print(f"Candidate pool pulled: {len(candidates)}")

print("\nBuilding detector (recognition model only, for kps-replay)...")
extractor = FaceExtractor()
app = extractor._build_detector()
rec_model = app.app.models['recognition']

neg_lv, neg_if = [], []
decode_failed = kps_mismatch = 0
for i, face in enumerate(candidates):
    if len(neg_lv) >= NEG_SAMPLE_SIZE:
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
    neg_lv.append(np.array(face.face_encoding_lvface_b, dtype=np.float32))
    neg_if.append(np.array(embedding, dtype=np.float32))
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}/{len(candidates)} candidates tried, {len(neg_lv)} usable so far "
              f"(decode_failed={decode_failed}, kps_mismatch={kps_mismatch})")

print(f"\nUsable negative pairs: {len(neg_lv)} (decode_failed={decode_failed}, kps_mismatch={kps_mismatch})")

neg_lv = normed(np.array(neg_lv, dtype=np.float32))
neg_if = normed(np.array(neg_if, dtype=np.float32))

neg_lv_top1 = np.full(neg_lv.shape[0], -1.0, dtype=np.float32)
neg_if_top1 = np.full(neg_if.shape[0], -1.0, dtype=np.float32)
for pid in person_ids:
    sims_lv = neg_lv @ gallery_lv[pid].T
    neg_lv_top1 = np.maximum(neg_lv_top1, np.percentile(sims_lv, 99, axis=1))
    sims_if = neg_if @ gallery_if[pid].T
    neg_if_top1 = np.maximum(neg_if_top1, np.percentile(sims_if, 99, axis=1))

print("\n=== Real negative-side agreement test (freshly re-encoded insightface via kps-replay) ===")
for lv_t in LV_THRESHOLDS:
    lv_pass = neg_lv_top1 >= lv_t
    if_pass = neg_if_top1 >= IF_THRESHOLD
    both = lv_pass & if_pass
    either = lv_pass | if_pass
    print(f"\nLVFace-B threshold={lv_t}, insightface threshold={IF_THRESHOLD}:")
    print(f"  LVFace-B alone FPR:      {lv_pass.mean():.4f}")
    print(f"  insightface alone FPR:  {if_pass.mean():.4f}")
    print(f"  AND-gate (both) FPR:    {both.mean():.4f}")
    print(f"  OR-gate (either) FPR:   {either.mean():.4f}")

corr = float(np.corrcoef(neg_lv_top1, neg_if_top1)[0, 1]) if len(neg_lv_top1) > 1 else float('nan')
print(f"\nCorrelation between LVFace-B and insightface top1 scores on these SAME negative faces: {corr:.3f}")

print("\nDone.")
