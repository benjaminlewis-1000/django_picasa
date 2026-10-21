# Backfills VideoFile.collections (added in 0030) for every pre-existing
# row, from its own directory.collection -- mirrors 0026's equivalent
# ImageFile.collections backfill. Per-row (not a single blanket "Default"
# assignment like 0026's simpler case) since by the time this runs a
# second real collection (Lewis Parents) may already have videos too.
from django.db import migrations


def backfill(apps, schema_editor):
    VideoFile = apps.get_model('filepopulator', 'VideoFile')

    through = VideoFile.collections.through
    rows = VideoFile.objects.exclude(directory__collection__isnull=True).values_list(
        'id', 'directory__collection_id',
    )
    through.objects.bulk_create(
        [
            through(videofile_id=video_id, photocollection_id=collection_id)
            for video_id, collection_id in rows
        ],
        batch_size=5000, ignore_conflicts=True,
    )


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('filepopulator', '0030_duplicatevideofile_collection_videofile_collections'),
    ]

    operations = [
        migrations.RunPython(backfill, noop_reverse),
    ]
