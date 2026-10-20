# Seeds CameraGpsBackfillConfig with the cameras that were hardcoded,
# enabled, in owntracks_gps.py's old SUPPORTED_CAMERA_MODELS list before
# that list moved into this DB-backed table -- keeps the real production
# backfill (already run against these cameras) from silently losing its
# enabled scope the moment the code switches to reading from the DB.

from django.db import migrations

# Historical allowlist, frozen here rather than imported from
# owntracks_gps.py -- a migration should never depend on application
# code that might itself change later.
SEED_ENABLED_CAMERA_MODELS = ['Galaxy S23+', 'SM-S916U', 'NIKON D3200']


def seed_enabled_cameras(apps, schema_editor):
    CameraGpsBackfillConfig = apps.get_model('filepopulator', 'CameraGpsBackfillConfig')
    for camera_model in SEED_ENABLED_CAMERA_MODELS:
        CameraGpsBackfillConfig.objects.get_or_create(
            camera_model=camera_model, defaults={'enabled': True},
        )


def unseed_enabled_cameras(apps, schema_editor):
    CameraGpsBackfillConfig = apps.get_model('filepopulator', 'CameraGpsBackfillConfig')
    CameraGpsBackfillConfig.objects.filter(camera_model__in=SEED_ENABLED_CAMERA_MODELS).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('filepopulator', '0023_cameragpsbackfillconfig'),
    ]

    operations = [
        migrations.RunPython(seed_enabled_cameras, unseed_enabled_cameras),
    ]
