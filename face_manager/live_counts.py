"""Live (non-cached) computation of per-Person face counts.

Replaces the old Person.num_faces / num_possibilities / num_unverified_faces
cached IntegerField columns, which were removed because they could silently
drift stale -- any code path that mutated Face.declared_name/poss_identN
without going through the old increment_*/decrement_* model methods (most
notably a bulk .update()) left the cached numbers wrong until the next
scheduled recompute. See CLAUDE.md for the investigation/benchmarking that
motivated this: a naive per-person-loop live query took ~4.5s across the
full person roster; a single query using correlated subqueries (avoiding
the join fan-out a naive multi-Count(distinct=True) annotation produces --
that version hung for 10+ minutes on real data) got that to ~1s; two
covering indexes (face_manager_face_declared_name_id_covering,
face_manager_face_poss_ident1_id_covering) got it to ~0.2-0.4s
single-threaded; splitting the person set across a few threads (each a
genuinely separate Postgres backend process, so real multi-core
parallelism, not fighting Python's GIL) gets the full-roster case to
~0.2s.
"""
from concurrent.futures import ThreadPoolExecutor

from django.db import connection
from django.db.models import Count, IntegerField, OuterRef, Q, Subquery, Value
from django.db.models.functions import Coalesce

from face_manager.models import Face, Person


def annotate_live_face_counts(queryset):
    """Annotate a Person queryset with live num_faces/num_possibilities/
    num_unverified_faces attributes, computed via correlated subqueries.

    Deliberately NOT implemented as joined Count(distinct=True)
    annotations -- joining face_declared and face_poss1 into the same
    query before aggregating produces a Cartesian product per person
    (confirmed on real data: hung for 10+ minutes on a table with some
    galleries in the tens of thousands of faces). Subqueries avoid the
    join entirely, at the cost of one SubPlan execution per person per
    subquery -- fast in practice given the two covering indexes below.

    Relies on two covering indexes for index-only scans (no heap
    fetches): face_manager_face_declared_name_id_covering
    (declared_name_id) INCLUDE (id, validated), and
    face_manager_face_poss_ident1_id_covering (poss_ident1_id)
    INCLUDE (id). Still correct without them, just slower.
    """
    faces_sq = (
        Face.objects.filter(declared_name=OuterRef('pk'))
        .order_by().values('declared_name')
        .annotate(c=Count('id'), u=Count('id', filter=Q(validated=False)))
        .values('c', 'u')
    )
    poss1_sq = (
        Face.objects.filter(poss_ident1=OuterRef('pk'))
        .order_by().values('poss_ident1')
        .annotate(c=Count('id')).values('c')
    )
    return queryset.annotate(
        num_faces=Coalesce(Subquery(faces_sq.values('c')[:1], output_field=IntegerField()), Value(0)),
        num_unverified_faces=Coalesce(Subquery(faces_sq.values('u')[:1], output_field=IntegerField()), Value(0)),
        num_possibilities=Coalesce(Subquery(poss1_sq, output_field=IntegerField()), Value(0)),
    )


def _chunk_live_counts(id_chunk):
    # Postgres chooses a JIT-compiled plan for this query by default
    # (its estimated cost is high enough to cross the JIT threshold),
    # but the compile overhead (measured ~650ms on real data) exceeds
    # what it saves for a query that only runs once per chunk.
    with connection.cursor() as c:
        c.execute("SET jit = off;")
    people = annotate_live_face_counts(Person.objects.filter(pk__in=id_chunk))
    return {
        p.id: {
            'num_faces': p.num_faces,
            'num_possibilities': p.num_possibilities,
            'num_unverified_faces': p.num_unverified_faces,
        }
        for p in people
    }


def _threaded_chunk_live_counts(id_chunk):
    try:
        return _chunk_live_counts(id_chunk)
    finally:
        # This chunk ran on a thread-local connection opened just for
        # this call -- close it explicitly rather than leaving it to
        # accumulate across repeated calls (e.g. many PersonListView
        # requests hitting a small worker pool).
        connection.close()


def compute_live_face_counts(person_ids, n_threads=4):
    """Bulk live-count computation for a set of Person ids, splitting the
    work across up to n_threads separate DB connections. Each Postgres
    connection is its own backend process, so this is genuine multi-core
    parallelism rather than Python threads contending over the GIL --
    appropriate here since the work is DB-round-trip-bound, not CPU-bound
    in Python.

    Returns {person_id: {'num_faces': ..., 'num_possibilities': ...,
    'num_unverified_faces': ...}}.
    """
    person_ids = list(person_ids)
    if not person_ids:
        return {}

    n_threads = max(1, min(n_threads, len(person_ids)))
    chunk_size = (len(person_ids) + n_threads - 1) // n_threads
    chunks = [person_ids[i:i + chunk_size] for i in range(0, len(person_ids), chunk_size)]

    if len(chunks) == 1:
        return _chunk_live_counts(chunks[0])

    results = {}
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        for chunk_result in ex.map(_threaded_chunk_live_counts, chunks):
            results.update(chunk_result)
    return results
