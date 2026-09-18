from django.db import migrations


class Migration(migrations.Migration):

    # CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction.
    atomic = False

    dependencies = [
        ('face_manager', '0013_face_det_score'),
    ]

    operations = [
        # Backs live_counts.py's new num_possibilities_video subquery
        # (poss_ident1=X AND source_video_file IS NOT NULL) - without
        # source_video_file_id in the INCLUDE list, that filter would fall
        # back to a heap fetch per candidate row instead of an index-only
        # scan, on top of whatever face_manager_face_poss_ident1_id_covering
        # (0010_face_count_covering_indexes) already provides for the
        # plain poss_ident1 count. Same CONCURRENTLY/IF NOT EXISTS
        # reasoning as that migration.
        migrations.RunSQL(
            sql=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "face_manager_face_poss_ident1_video_covering "
                "ON face_manager_face (poss_ident1_id) INCLUDE (id, source_video_file_id);"
            ),
            reverse_sql=(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "face_manager_face_poss_ident1_video_covering;"
            ),
        ),
    ]
