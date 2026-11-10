# 2026-11-10: targeted LVFace-L backfill for David Redd (1411) and Luca
# Wilson (2431) -- same shape as scratch_backfill_lvface_l_leslie.py,
# generalized to a list of person ids, to check whether Leslie's dramatic
# ~98% magnet-gallery cut generalizes or was a one-off. Emma (2034) and
# Elder Tim Call (1754) are skipped -- both currently have ZERO proposed
# candidates (the insightface-side cohesion/span nudge already cleared
# their queues), so there's nothing to test right now.
#
# Writes face_encoding_lvface_l (+ halfvec mirror) for real via
# Face.set_embedding() -- not a throwaway computation.
#
# Run via: cat scratch_backfill_lvface_l_magnets.py | docker exec -i
# picasa_api python manage.py shell

import time
from collections import defaultdict

from django.db.models import Q

import common
import lvface
from face_manager.models import Face
from filepopulator.models import ImageFile

PERSON_IDS = [1411, 2431]  # David Redd, Luca Wilson
FIELD = 'face_encoding_lvface_l'

q = Q()
for pid in PERSON_IDS:
    q |= Q(declared_name_id=pid, validated=True) | Q(poss_ident1_id=pid)

target_faces = Face.objects.filter(
    kps__isnull=False, source_image_file__isnull=False,
    **{f'{FIELD}__isnull': True},
).filter(q)

image_ids = list(target_faces.values_list('source_image_file_id', flat=True).distinct())
print(f"{len(image_ids)} source images, {target_faces.count()} faces to embed with LVFace-L")

stats = defaultdict(int)
t0 = time.time()
for idx, img_id in enumerate(image_ids):
    img_obj = ImageFile.objects.select_related('directory__collection').get(id=img_id)
    image_faces = list(target_faces.filter(source_image_file=img_obj))
    if not image_faces:
        continue
    try:
        img_numpy = common.open_img_oriented(
            img_obj.filename, as_numpy=True,
            extra_rotation_degrees=img_obj.confirmed_rotation_override_degrees,
        )
    except Exception as e:
        print(f"  decode error on {img_obj.filename}: {e}")
        stats['decode_failed'] += len(image_faces)
        continue
    if img_numpy is None:
        stats['decode_failed'] += len(image_faces)
        continue

    for face in image_faces:
        try:
            emb = lvface.embed_from_image_array(img_numpy, face.kps, input_is_bgr=False, size='L')
        except Exception as e:
            print(f"  LVFace-L failed for face {face.id}: {e}")
            stats['embed_failed'] += 1
            continue
        if emb is None:
            stats['embed_none'] += 1
            continue
        face.set_embedding(FIELD, emb.tolist())
        face.save(update_fields=[FIELD, Face.EMBEDDING_FIELD_HV_MIRRORS[FIELD]])
        stats['embedded'] += 1

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(image_ids) - (idx + 1)) / rate / 60 if rate else 0
        print(f"  ...{idx+1}/{len(image_ids)} images ({rate:.2f} img/s, ETA {eta:.0f}min) -- "
              + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))

print(f"\nDONE: {len(image_ids)} images, {time.time()-t0:.0f}s. "
      + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
