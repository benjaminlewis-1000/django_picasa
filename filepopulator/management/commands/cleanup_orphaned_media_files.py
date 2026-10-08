from django.core.management.base import BaseCommand

from filepopulator.scripts import cleanup_orphaned_media_files


class Command(BaseCommand):
    help = (
        "Deletes orphaned thumbnail/highlight files -- ones sitting on "
        "disk that no current row references anymore. See "
        "cleanup_orphaned_media_files()'s own docstring (filepopulator/"
        "scripts.py) for the full rationale -- this command and the "
        "weekly scheduled task (filepopulator.cleanup_orphaned_media) "
        "share the same underlying function."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument(
            '--min-age-hours', type=float, default=24.0,
            help='Never touch a file modified more recently than this (default 24h) -- '
                 'guards against the race between a file landing on disk and its row committing.',
        )

    def handle(self, *args, **options):
        cleanup_orphaned_media_files(
            dry_run=options['dry_run'],
            min_age_hours=options['min_age_hours'],
            log=self.stdout.write,
        )
