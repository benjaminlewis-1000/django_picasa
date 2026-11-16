from django.db import migrations


class Migration(migrations.Migration):

    # CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction.
    atomic = False

    dependencies = [
        ('face_manager', '0024_remove_face_face_kps_within_box_margin_and_more'),
    ]

    operations = [
        # Backs the new ".ignore flagged-review's next-best-guess" subquery
        # (live_counts.py's num_ignore_review, and PersonParamView's
        # face_ignore_review listing) - poss_ident2 has no index at all
        # today, unlike poss_ident1 (0010_face_count_covering_indexes), so
        # without this the subquery would be a sequential scan of the whole
        # Face table per person. INCLUDE covers every column that subquery
        # filters/splits on (mobile_review_hidden, poss_ident1_id to scope
        # to actual .ignore candidates, source_video_file_id for the
        # existing media-filter split) so it stays an index-only scan. Same
        # CONCURRENTLY/IF NOT EXISTS reasoning as the migrations above.
        migrations.RunSQL(
            sql=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "face_manager_face_poss_ident2_covering "
                "ON face_manager_face (poss_ident2_id) "
                "INCLUDE (id, mobile_review_hidden, poss_ident1_id, source_video_file_id);"
            ),
            reverse_sql=(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "face_manager_face_poss_ident2_covering;"
            ),
        ),
    ]
