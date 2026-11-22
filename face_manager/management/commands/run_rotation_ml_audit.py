"""Manual-trigger wrapper around face_manager.rotation_ml_audit -- the
ML-classifier counterpart to audit_image_rotation.py (the landmark
signal's own command). See face_manager/rotation_ml_audit.py's own
docstring: this is infrastructure built ahead of being scheduled --
nothing here runs on its own, only this command (or a direct function
call) ever examines/writes anything.
"""
from django.core.management.base import BaseCommand, CommandError

from face_manager.rotation_ml_audit import (
    DEFAULT_MAX_IMAGES_PER_RUN,
    DEFAULT_MAX_RUNTIME_SECONDS,
    run_rotation_ml_audit,
    run_rotation_ml_audit_until_done,
)


class Command(BaseCommand):
    help = (
        "Scan ImageFile rows for likely rotation using the trained ML orientation "
        "classifier -- unlike audit_image_rotation (the face-landmark signal), this "
        "needs no detected faces/kps and considers every NOT_SCANNED image. Read-only, "
        "single-batch by default; pass --yes to write, which loops bounded batches "
        "until the backlog is exhausted or --max-runtime-seconds is reached. Requires "
        "a trained, exported ONNX model at the path "
        "face_manager.rotation_ml_audit.ONNX_MODEL_PATH names -- if none exists yet, "
        "fails with a clear error rather than crashing."
    )

    def add_arguments(self, parser):
        parser.add_argument('--yes', action='store_true',
                             help="Actually write rotation_scan_state/proposed_rotation_degrees/"
                                  "rotation_signal_source/rotation_ml_scan_hash, looping batches "
                                  "until done or the time limit. Without this, reports counts for "
                                  "a single batch only -- no DB writes.")
        parser.add_argument('--max-images', type=int, default=DEFAULT_MAX_IMAGES_PER_RUN,
                             help=f"Per-batch cap (default {DEFAULT_MAX_IMAGES_PER_RUN}). Pass 0 "
                                  "to remove the cap entirely for a dry-run preview only (ignored "
                                  "with --yes, which always loops in bounded batches).")
        parser.add_argument('--max-runtime-seconds', type=int, default=DEFAULT_MAX_RUNTIME_SECONDS,
                             help=f"With --yes only (default {DEFAULT_MAX_RUNTIME_SECONDS}): stop "
                                  "looping after this long even if the backlog isn't exhausted.")

    def handle(self, *args, **options):
        if options['yes']:
            result = run_rotation_ml_audit_until_done(
                batch_size=options['max_images'], max_runtime_seconds=options['max_runtime_seconds'],
            )
            if result.get('skipped_no_checkpoint'):
                raise CommandError(
                    "No trained checkpoint found -- run train_rotation_classifier first.")
            self.stdout.write(f"Candidates examined: {result['examined']} across {result['batches']} batch(es)")
            for key, value in result['counts'].items():
                self.stdout.write(f"  {key}: {value}")
            self.stdout.write(self.style.SUCCESS(f"Wrote updates for {result['updated']} image(s)."))
            if result['hit_time_limit']:
                self.stdout.write(self.style.WARNING(
                    "Stopped at the time limit with backlog likely remaining -- re-run to continue."))
        else:
            max_images = options['max_images'] or None
            try:
                result = run_rotation_ml_audit(write=False, max_images=max_images)
            except FileNotFoundError:
                raise CommandError(
                    "No trained checkpoint found -- run train_rotation_classifier first.")
            self.stdout.write(f"Candidates examined: {result['examined']}")
            for key, value in result['counts'].items():
                self.stdout.write(f"  {key}: {value}")
            self.stdout.write(self.style.WARNING(
                f"Dry run (single batch) -- {result['updated']} image(s) would be updated. "
                "Pass --yes to write."))
