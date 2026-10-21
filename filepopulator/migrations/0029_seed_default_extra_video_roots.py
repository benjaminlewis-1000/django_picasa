# Seeds the Default PhotoCollection's extra_video_roots with the real
# dedicated video-only folders that used to be hardcoded as the
# VIDEO_ROOTS Django setting (see filepopulator/models.py's
# PhotoCollection and CLAUDE.md, 2026-10-21) -- these aren't under any
# collection's own root_directory, so they need an explicit entry here
# or the scheduled video scan would stop walking them the moment the
# code switches from reading settings.VIDEO_ROOTS to reading this field.
from django.conf import settings
from django.db import migrations
import os


def seed_extra_video_roots(apps, schema_editor):
    PhotoCollection = apps.get_model('filepopulator', 'PhotoCollection')
    default_collection = PhotoCollection.objects.filter(is_default=True).first()
    if default_collection is None:
        return

    legacy_video_roots = [
        os.path.join(settings.VIDEO_ROOT, 'Our_Home_Videos'),
        os.path.join(settings.VIDEO_ROOT, 'Lewis_family_VHS_video_clips'),
    ]
    existing = set(default_collection.extra_video_roots or [])
    merged = sorted(existing | set(legacy_video_roots))
    if merged != sorted(existing):
        default_collection.extra_video_roots = merged
        default_collection.save(update_fields=['extra_video_roots'])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('filepopulator', '0028_photocollection_extra_video_roots'),
    ]

    operations = [
        migrations.RunPython(seed_extra_video_roots, noop_reverse),
    ]
