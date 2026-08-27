"""Near-duplicate detection via perceptual hash (phash) Hamming distance.

Every ImageFile gets its phash computed synchronously at ingestion time
(filepopulator/models.py's _generate_md5_hash(), alongside the existing
pixel MD5 hash) -- there's no separate "encode" task. What lives here is
the comparison step: finding which images are close to which others.

Comparison is deliberately kept out of the database entirely except for
its inputs and outputs. The O(n^2) candidate-pair work (bounded, per a
real benchmark against production-sized data: ~200k images is ~2*10^10
unique pairs, comparable via vectorized numpy popcount(XOR) in well under
a minute) all happens against in-memory numpy arrays loaded with a single
query; only the rare pairs that actually clear the similarity threshold
ever get written back, one row each, to SimilarImagePair.
"""

import numpy as np
from django.conf import settings


def _load_hash_arrays():
    """Returns (ids, hashes) as aligned numpy arrays for every ImageFile
    that has a phash -- one query, no further DB access needed for the
    comparison itself."""
    from filepopulator.models import ImageFile

    rows = list(ImageFile.objects.exclude(phash=None).values_list('id', 'phash'))
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    hashes = np.array([r[1] for r in rows], dtype=np.int64)
    return ids, hashes


def _hamming_distances(one_hash, all_hashes):
    """Vectorized Hamming distance from one 64-bit hash to an array of
    them -- popcount(XOR), reinterpreting the signed bigint storage as
    unsigned 64-bit for the bit-count itself (the bit pattern is
    identical either way, bitwise_count just requires an unsigned dtype)."""
    xored = np.bitwise_xor(one_hash, all_hashes)
    return np.bitwise_count(xored.view(np.uint64))


def run_similarity_check(limit=None, dry_run=False, log=print):
    """Compares every ImageFile with similarity_checked=False against the
    full population of hashed images, records any pair within
    settings.PHASH_SIMILARITY_THRESHOLD as a SimilarImagePair, then marks
    them checked. Shared by the one-time backfill command and the small
    recurring Celery task, differing only in how large a batch (`limit`)
    makes sense to run at once.

    Safe to interrupt and re-run: an image is only ever compared once it
    has similarity_checked=False, so a partial run just picks up where it
    left off. Comparing an already-checked pair again (e.g. two images
    that are both unchecked in the same run, so each compares against the
    other) is harmless -- SimilarImagePair.record() canonicalizes the
    pair order and upserts, so it can't create a duplicate row.

    Returns a dict of counts: total_hashed, already_checked, remaining,
    checked, matches_found.
    """
    from filepopulator.models import ImageFile, SimilarImagePair

    ids, hashes = _load_hash_arrays()
    id_to_index = {int(image_id): i for i, image_id in enumerate(ids)}

    to_check_ids = list(
        ImageFile.objects.filter(similarity_checked=False)
        .exclude(phash=None)
        .order_by('id')
        .values_list('id', flat=True)
    )

    result = {
        'total_hashed': len(ids),
        'already_checked': len(ids) - len(to_check_ids),
        'remaining': len(to_check_ids),
        'checked': 0,
        'matches_found': 0,
    }

    log(f"Images with a phash: {result['total_hashed']}")
    log(f"Already checked: {result['already_checked']}")
    log(f"Remaining to check: {result['remaining']}")

    if dry_run:
        log("Dry run -- no changes written.")
        return result

    if limit is not None:
        to_check_ids = to_check_ids[:limit]

    threshold = settings.PHASH_SIMILARITY_THRESHOLD

    for image_id in to_check_ids:
        idx = id_to_index[image_id]
        this_hash = hashes[idx]

        distances = _hamming_distances(this_hash, hashes)
        match_indices = np.where(distances <= threshold)[0]

        for match_idx in match_indices:
            other_id = int(ids[match_idx])
            if other_id == image_id:
                continue
            SimilarImagePair.record(image_id, other_id, int(distances[match_idx]))
            result['matches_found'] += 1

        result['checked'] += 1

    ImageFile.objects.filter(id__in=to_check_ids).update(similarity_checked=True)

    log(f"Done. {result['checked']} checked, {result['matches_found']} similar pairs found.")
    return result
