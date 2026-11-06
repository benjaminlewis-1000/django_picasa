"""Manual-trigger wrapper around face_manager.rotation_audit -- the same
functions the scheduled face_manager.audit_image_rotation Celery task
calls. See CLAUDE.md's 2026-11-03/11-05/11-06 orientation-detection
write-up for the design and validation this is built on.
"""
from django.core.management.base import BaseCommand

from face_manager.rotation_audit import (
    DEFAULT_MAX_IMAGES_PER_RUN,
    DEFAULT_MAX_RUNTIME_SECONDS,
    run_rotation_audit,
    run_rotation_audit_until_done,
)


class Command(BaseCommand):
    help = (
        "Scan ImageFile rows for likely rotation using the face-landmark "
        "eye-angle signal. Read-only, single-batch by default; pass --yes "
        "to write, which loops bounded batches until the backlog is "
        "exhausted or --max-runtime-seconds is reached (same as the "
        "scheduled task -- see run_rotation_audit_until_done()'s own "
        "comment for why looping only makes sense once writing)."
    )

    def add_arguments(self, parser):
        parser.add_argument('--yes', action='store_true',
                             help="Actually write rotation_scan_state/proposed_rotation_degrees/"
                                  "rotation_signal_source/rotation_scan_logic_hash, looping "
                                  "batches until done or the time limit. Without this, reports "
                                  "counts for a single batch only -- no DB writes.")
        parser.add_argument('--max-images', type=int, default=DEFAULT_MAX_IMAGES_PER_RUN,
                             help=f"Per-batch cap (default {DEFAULT_MAX_IMAGES_PER_RUN}) -- see "
                                  "run_rotation_audit()'s own comment on why this isn't unbounded. "
                                  "Pass 0 to remove the cap entirely for a dry-run preview only "
                                  "(ignored with --yes, which always loops in bounded batches).")
        parser.add_argument('--max-runtime-seconds', type=int, default=DEFAULT_MAX_RUNTIME_SECONDS,
                             help=f"With --yes only (default {DEFAULT_MAX_RUNTIME_SECONDS}): stop "
                                  "looping after this long even if the backlog isn't exhausted.")

    def handle(self, *args, **options):
        if options['yes']:
            result = run_rotation_audit_until_done(
                batch_size=options['max_images'], max_runtime_seconds=options['max_runtime_seconds'],
            )
            self.stdout.write(f"Candidates examined: {result['examined']} across {result['batches']} batch(es)")
            for key, value in result['counts'].items():
                self.stdout.write(f"  {key}: {value}")
            self.stdout.write(self.style.SUCCESS(f"Wrote updates for {result['updated']} image(s)."))
            if result['hit_time_limit']:
                self.stdout.write(self.style.WARNING(
                    "Stopped at the time limit with backlog likely remaining -- re-run to continue "
                    "(or wait for the next scheduled tick)."))
        else:
            max_images = options['max_images'] or None
            result = run_rotation_audit(write=False, max_images=max_images)
            self.stdout.write(f"Candidates examined: {result['examined']}")
            for key, value in result['counts'].items():
                self.stdout.write(f"  {key}: {value}")
            self.stdout.write(self.style.WARNING(
                f"Dry run (single batch) -- {result['updated']} image(s) would be updated. "
                "Pass --yes to write."))
