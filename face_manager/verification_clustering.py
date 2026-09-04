"""Nightly clustering of confirmed-but-unverified faces.

Groups visually similar faces within the SAME person's own gallery so a
human reviewing a run can spot-check a whole coherent group at once,
rather than one face at a time. Populates Face.verification_cluster_group.

Complete-linkage agglomerative clustering was the only method (of many
tried against real production embeddings -- HDBSCAN eom/leaf, DBSCAN,
kNN+Louvain, kNN+connected-components, average linkage) that didn't
collapse into one dominant "blob" cluster: every other method tried is in
the single-linkage family, which merges A and D transitively as long as
some chain A-B-C-D each falls under threshold, even if A and D aren't
alike. Complete linkage requires the WORST pairwise distance within a
candidate cluster to still clear threshold, which structurally blocks
that chaining. See CLAUDE.md for the full investigation, including why
this only works on a single person's own gallery (mixing multiple
people's embeddings, e.g. the .ignore population, reproduced the same
blob failure regardless of algorithm).
"""
from django.conf import settings
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from face_manager.models import Face

DEFAULT_COS_THRESHOLD = 0.7


def _cos_to_euclidean_distance(cos_threshold):
    # AgglomerativeClustering's distance_threshold is Euclidean; on
    # L2-normalized vectors that's exactly sqrt(2 - 2*cos_sim).
    return np.sqrt(max(0.0, 2 - 2 * cos_threshold))


def eligible_faces_queryset():
    """Faces eligible for verification-cluster grouping: confirmed to a
    real person (not a blank/ignore sentinel), not yet human-verified,
    and carrying a real embedding -- excludes NULL and the
    NON_DETECTED_FACE_ENCODING sentinel, the same exclusion
    reencode_missing_faces() uses for "needs a real embedding"."""
    return (
        Face.objects.filter(validated=False)
        .exclude(declared_name__person_name__in=settings.IGNORED_NAMES)
        .exclude(face_encoding_512__isnull=True)
        .exclude(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)
    )


def _cluster_person_faces(face_ids, embeddings, cos_threshold):
    """Returns a list of face-id groups (each len >= 2) for one person's
    faces. Singletons are simply omitted -- caller leaves them NULL."""
    face_ids = np.array(face_ids)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    distance_threshold = _cos_to_euclidean_distance(cos_threshold)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        linkage='complete', metric='euclidean',
    ).fit_predict(normed.astype(np.float32))

    groups = []
    for label in np.unique(labels):
        member_ids = face_ids[labels == label].tolist()
        if len(member_ids) >= 2:
            groups.append(member_ids)
    return groups


def cluster_all_unverified_faces(cos_threshold=DEFAULT_COS_THRESHOLD):
    """Full nightly rebuild: clears every Face.verification_cluster_group
    value db-wide, then re-groups each eligible person's unverified faces
    from scratch, independently per person (group ids are 0-indexed per
    person, not globally unique). No attempt is made to preserve group
    identity night-to-night. Returns (num_people_clustered,
    num_faces_grouped)."""
    Face.objects.exclude(verification_cluster_group__isnull=True).update(
        verification_cluster_group=None
    )

    by_person = {}
    for face_id, person_id, encoding in eligible_faces_queryset().values_list(
        'id', 'declared_name_id', 'face_encoding_512'
    ):
        face_ids, encodings = by_person.setdefault(person_id, ([], []))
        face_ids.append(face_id)
        encodings.append(encoding)

    num_people_clustered = 0
    num_faces_grouped = 0
    for person_id, (face_ids, encodings) in by_person.items():
        if len(face_ids) < 2:
            continue
        embeddings = np.array(encodings, dtype=np.float32)
        groups = _cluster_person_faces(face_ids, embeddings, cos_threshold)
        if not groups:
            continue
        for group_id, member_ids in enumerate(groups):
            Face.objects.filter(pk__in=member_ids).update(
                verification_cluster_group=group_id
            )
        num_people_clustered += 1
        num_faces_grouped += sum(len(g) for g in groups)

    return num_people_clustered, num_faces_grouped
