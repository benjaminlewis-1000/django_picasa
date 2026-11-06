"""Manual-trigger wrapper around face_manager.rotation_audit.run_rotation_audit
-- the same function the scheduled face_manager.audit_image_rotation
Celery task calls. See CLAUDE.md's 2026-11-03/11-05 orientation-
detection write-up for the design and validation this is built on.
"""
from django.core.management.base import BaseCommand

from face_manager.rotation_audit import run_rotation_audit


class Command(BaseCommand):
    help = (
        "Scan ImageFile rows for likely rotation using the face-landmark "
        "eye-angle signal. Read-only by default; pass --yes to write."
    )

    def add_arguments(self, parser):
        parser.add_argument('--yes', action='store_true',
                             help="Actually write rotation_scan_state/proposed_rotation_degrees/"
                                  "rotation_signal_source/rotation_scanned_at_version. Without "
                                  "this, reports counts only -- no DB writes.")

    def handle(self, *args, **options):
        write = options['yes']
        result = run_rotation_audit(write=write)

        self.stdout.write(f"Candidates examined: {result['examined']}")
        for key, value in result['counts'].items():
            self.stdout.write(f"  {key}: {value}")

        if write:
            self.stdout.write(self.style.SUCCESS(f"Wrote updates for {result['updated']} image(s)."))
        else:
            self.stdout.write(self.style.WARNING(
                f"Dry run -- {result['updated']} image(s) would be updated. Pass --yes to write."))
