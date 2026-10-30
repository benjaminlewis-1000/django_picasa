"""Targeted LVFace-S/B/L backfill for the LVFace-vs-insightface investigation,
scoped to faces that already have VERIFIED kps (from backfill_verified_kps)
and belong to either a confirmed real person or a confirmed .ignore/
.realignore face -- never the blank/unverified sentinel population, which
isn't useful for a leave-one-out gallery/negative-pool comparison.

Deliberately skips detection/IOU-matching entirely -- kps was already
verified when it was saved (either against the face's own stored
insightface embedding, or geometrically via the DB's own
face_kps_within_box_margin constraint), so this command just replays that
trusted kps directly through LVFace's recognition-only alignment+embed,
the same kps-replay trick reencode_missing_faces() already uses elsewhere
in this codebase. No PyramidalDetector/FaceAnalysis detection pass at all
-- just an image decode plus 1-3 small LVFace ONNX calls per face, which
should be far cheaper than the full verified-kps pipeline.

Oriented around whole images (not individual faces), same reasoning as
backfill_verified_kps: one decode serves every qualifying face on that
image.
"""
import signal
import time
from collections import defaultdict

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models.expressions import RawSQL

import common
import lvface
from face_manager.models import Face
from filepopulator.models import ImageFile

PER_IMAGE_TIMEOUT_SECONDS = 30


class _ImageTimeout(Exception):
    pass


def _raise_image_timeout(signum, frame):
    raise _ImageTimeout(f"Image processing exceeded {PER_IMAGE_TIMEOUT_SECONDS}s")


def _image_shard_annotation(total_shards):
    return RawSQL('MOD(ABS(HASHTEXT(filepopulator_imagefile.id::text)), %s)', (total_shards,))


def _qualifying_faces_queryset(lvface_sizes):
    """Faces with verified kps, not yet backfilled for the FIRST requested
    size (so a re-run naturally skips already-done faces), restricted to
    confirmed real people or confirmed .ignore/.realignore -- the blank
    sentinel (_NO_FACE_ASSIGNED_, unverified) is deliberately excluded."""
    first_field = f'face_encoding_lvface_{lvface_sizes[0].lower()}'
    return Face.objects.filter(
        kps__isnull=False,
        source_image_file__isnull=False,
        **{f'{first_field}__isnull': True},
    ).exclude(
        declared_name__person_name='_NO_FACE_ASSIGNED_',
    )


class Command(BaseCommand):
    help = __doc__

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None,
                             help='Restrict to the first N source images (for testing).')
        parser.add_argument('--shard-index', type=int, default=None)
        parser.add_argument('--total-shards', type=int, default=None)
        parser.add_argument('--lvface-sizes', type=str, default='S,B,L')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        shard_index = options['shard_index']
        total_shards = options['total_shards']
        lvface_sizes = [s.strip().upper() for s in options['lvface_sizes'].split(',') if s.strip()]
        if not lvface_sizes:
            self.stdout.write("No --lvface-sizes given -- nothing to do.")
            return

        qualifying = _qualifying_faces_queryset(lvface_sizes)
        trigger_image_ids = qualifying.values_list('source_image_file_id', flat=True).distinct()
        trigger_images = ImageFile.objects.filter(id__in=trigger_image_ids)

        if shard_index is not None and total_shards is not None:
            trigger_images = trigger_images.annotate(
                _shard=_image_shard_annotation(total_shards)
            ).filter(_shard=shard_index)

        image_ids = list(trigger_images.order_by('?').values_list('id', flat=True))
        if limit is not None:
            image_ids = image_ids[:limit]

        self.stdout.write(f"{len(image_ids)} source images to process {'(dry run)' if dry_run else ''}")

        stats = defaultdict(int)
        t0 = time.time()

        for idx, img_id in enumerate(image_ids):
            img_obj = ImageFile.objects.select_related('directory__collection').get(id=img_id)
            image_faces = list(qualifying.filter(source_image_file=img_obj))
            if not image_faces:
                continue

            try:
                signal.signal(signal.SIGALRM, _raise_image_timeout)
                signal.alarm(PER_IMAGE_TIMEOUT_SECONDS)
                try:
                    img_numpy = common.open_img_oriented(img_obj.filename, as_numpy=True)
                finally:
                    signal.alarm(0)
            except _ImageTimeout:
                self.stdout.write(f"  TIMEOUT on {img_obj.filename} -- skipping")
                stats['timed_out'] += len(image_faces)
                continue
            except Exception as e:
                self.stdout.write(f"  decode error on {img_obj.filename}: {e}")
                stats['decode_failed'] += len(image_faces)
                continue

            if img_numpy is None:
                stats['decode_failed'] += len(image_faces)
                continue

            for face in image_faces:
                if dry_run:
                    stats['would_embed'] += 1
                    continue
                any_saved = False
                for size in lvface_sizes:
                    field = f'face_encoding_lvface_{size.lower()}'
                    if getattr(face, field) is not None:
                        continue
                    try:
                        emb = lvface.embed_from_image_array(
                            img_numpy, face.kps, input_is_bgr=False, size=size,
                        )
                    except Exception as e:
                        self.stdout.write(f"  LVFace-{size} failed for face {face.id}: {e}")
                        continue
                    if emb is None:
                        continue
                    face.set_embedding(field, emb.tolist())
                    face.save(update_fields=[field, Face.EMBEDDING_FIELD_HV_MIRRORS[field]])
                    any_saved = True
                stats['embedded' if any_saved else 'no_sizes_needed'] += 1

            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = len(image_ids) - (idx + 1)
                eta_min = remaining / rate / 60 if rate else float('inf')
                self.stdout.write(
                    f"  ...{idx+1}/{len(image_ids)} images ({rate:.2f} img/s, ETA {eta_min:.0f}min) -- "
                    + ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
                )

        elapsed = time.time() - t0
        self.stdout.write(
            f"DONE: {len(image_ids)} images, {elapsed:.0f}s. "
            + ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
        )
