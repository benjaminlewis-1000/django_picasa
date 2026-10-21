from django.core.management.base import BaseCommand

from filepopulator.models import Directory, PhotoCollection


class Command(BaseCommand):
    help = (
        "One-time backfill of Directory.collection for directories created "
        "before create_video_file() started resolving it (2026-10-21). Only "
        "touches directories that actually have at least one VideoFile -- a "
        "directory with images too would already have a correctly-resolved "
        "collection from the image side, via ImageFile._get_dir(). Safe to "
        "re-run: only matches rows still NULL."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        qs = Directory.objects.filter(
            collection__isnull=True, video_set__isnull=False,
        ).distinct()
        total = qs.count()

        updated = 0
        for directory in qs.iterator():
            collection = PhotoCollection.resolve_for_path(directory.dir_path)
            if not dry_run:
                Directory.objects.filter(pk=directory.pk).update(collection=collection)
            updated += 1

        self.stdout.write(
            f"{'Would update' if dry_run else 'Updated'} {updated}/{total} "
            f"video-only directory row(s)."
        )
