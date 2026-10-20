# Seeds the "default" PhotoCollection (root_directory=settings.PHOTO_ROOT,
# is_default=True, no face-detection floors -- preserves today's behavior
# exactly) and backfills every pre-existing Directory/ImageFile row onto
# it, so the new collection-scoping doesn't silently orphan the whole
# existing library the moment it goes live. See filepopulator/models.py's
# PhotoCollection for the full design.

from django.conf import settings
from django.db import migrations


def seed_and_backfill(apps, schema_editor):
    PhotoCollection = apps.get_model('filepopulator', 'PhotoCollection')
    Directory = apps.get_model('filepopulator', 'Directory')
    ImageFile = apps.get_model('filepopulator', 'ImageFile')

    default_collection, _ = PhotoCollection.objects.get_or_create(
        root_directory=settings.PHOTO_ROOT,
        defaults={'name': 'Default', 'is_default': True},
    )

    Directory.objects.filter(collection__isnull=True).update(collection=default_collection)

    # Bulk through-table insert, not a per-row .collections.add() loop --
    # real scale here is 200k+ ImageFile rows. ignore_conflicts=True
    # instead of checking for existing links first: the through table's
    # own (imagefile, photocollection) unique constraint makes a
    # re-run of this migration safely idempotent either way.
    through = ImageFile.collections.through
    through.objects.bulk_create(
        [
            through(imagefile_id=img_id, photocollection_id=default_collection.id)
            for img_id in ImageFile.objects.values_list('id', flat=True)
        ],
        batch_size=5000, ignore_conflicts=True,
    )


def unseed(apps, schema_editor):
    # Reversible in the trivial sense (removes the seeded row and its
    # backfilled links) -- not meant to be run against a database that's
    # since gained a second real collection.
    PhotoCollection = apps.get_model('filepopulator', 'PhotoCollection')
    PhotoCollection.objects.filter(root_directory=settings.PHOTO_ROOT, is_default=True).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('filepopulator', '0025_photocollection_directory_collection_and_more'),
    ]

    operations = [
        migrations.RunPython(seed_and_backfill, unseed),
    ]
