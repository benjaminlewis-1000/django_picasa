"""Detects a long-running process whose in-memory Django models are
stale relative to the live DB schema -- e.g. a celery worker still
holding the Face model from before a just-applied migration added a new
NOT-NULL column. See CLAUDE.md (2026-11-09): this exact situation made
every single face-match write fail with a raw "violates not-null
constraint" error for roughly an hour before anyone noticed, because
the many existing per-row try/except blocks throughout the pipeline
just log and move on to the next row -- there's no single choke point
that sees every such error, so retrofitting detection into each one
individually would be fragile and easy to miss a future call site. A
periodic structural self-check, independent of any specific error
message or call site, catches this general class of problem instead.
"""
from django.apps import apps
from django.db import connection


def find_stale_models():
    """Compares every installed Django model's known field columns
    against its live DB table's actual columns. Returns a dict of
    {"app_label.ModelName": {missing column names}} for any model whose
    table has one or more real DB columns the model doesn't know about
    -- the structural signature of a column having been added by a
    migration this process hasn't picked up (a model missing a column
    the DB *doesn't* have is a different, Django-level "unapplied
    migration" state, not the "stale in-memory process" problem this is
    for, and isn't checked here).

    Cheap: one introspection query per table (no row scanning), safe to
    run frequently.
    """
    stale = {}
    with connection.cursor() as cursor:
        for model in apps.get_models():
            if not model._meta.managed:
                continue
            table_name = model._meta.db_table
            try:
                db_columns = {
                    col.name for col in connection.introspection.get_table_description(cursor, table_name)
                }
            except Exception:
                # Table doesn't exist (yet), or introspection failed for
                # some unrelated reason -- not this check's job to
                # diagnose, just skip it.
                continue
            model_columns = {f.column for f in model._meta.fields}
            missing = db_columns - model_columns
            if missing:
                stale[f"{model._meta.app_label}.{model.__name__}"] = missing
    return stale
