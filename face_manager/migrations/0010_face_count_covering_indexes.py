from django.db import migrations


class Migration(migrations.Migration):

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    atomic = False

    dependencies = [
        ('face_manager', '0009_remove_person_face_counts'),
    ]

    operations = [
        # Covering indexes backing face_manager.live_counts's live
        # per-person face-count queries (see CLAUDE.md for the
        # benchmarking: these turn a bitmap-heap-scan-plus-heap-fetch
        # into a pure index-only scan, ~4-5x faster on real data).
        # IF NOT EXISTS / IF EXISTS: these were already created directly
        # against production (via CREATE INDEX CONCURRENTLY, matching
        # these exact names) before this migration existed, so this must
        # be a safe no-op there while still creating them for a fresh
        # install/CI/dev database.
        migrations.RunSQL(
            sql=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "face_manager_face_declared_name_id_covering "
                "ON face_manager_face (declared_name_id) INCLUDE (id, validated);"
            ),
            reverse_sql=(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "face_manager_face_declared_name_id_covering;"
            ),
        ),
        migrations.RunSQL(
            sql=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "face_manager_face_poss_ident1_id_covering "
                "ON face_manager_face (poss_ident1_id) INCLUDE (id);"
            ),
            reverse_sql=(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "face_manager_face_poss_ident1_id_covering;"
            ),
        ),
    ]
