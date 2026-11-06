"""Thin wrapper around face_manager.rotation_correction.
run_rotation_correction() -- see that module and CLAUDE.md's
2026-11-03/11-05/11-06 orientation-detection write-up for the design.
"""
from django.core.management.base import BaseCommand

from face_manager.rotation_correction import run_rotation_correction
from filepopulator.models import ImageFile


class Command(BaseCommand):
    help = (
        "Apply the pixel/geometry/encoding rotation correction to every "
        "ImageFile with rotation_scan_state=CONFIRMED and "
        "rotation_correction_applied=False."
    )

    def add_arguments(self, parser):
        parser.add_argument('--image-ids', type=str, default=None,
                             help="Comma-separated ImageFile ids to restrict to (still must be "
                                  "CONFIRMED). Default: every eligible row.")
        parser.add_argument('--dry-run', action='store_true',
                             help="Report what would be corrected without writing anything.")

    def handle(self, *args, **options):
        image_ids = None
        if options['image_ids']:
            image_ids = [int(x) for x in options['image_ids'].split(',') if x.strip()]

        if options['dry_run']:
            ScanState = ImageFile.RotationScanState
            queryset = ImageFile.objects.filter(
                rotation_scan_state=ScanState.CONFIRMED, rotation_correction_applied=False,
            )
            if image_ids is not None:
                queryset = queryset.filter(id__in=image_ids)
            for image in queryset:
                self.stdout.write(f"  Would correct ImageFile {image.id} ({image.filename}), "
                                   f"proposed_rotation_degrees={image.proposed_rotation_degrees}")
            self.stdout.write(f"{queryset.count()} CONFIRMED-but-uncorrected image(s) to process.")
            return

        result = run_rotation_correction(image_ids=image_ids)
        for image_id, error in result['failures']:
            self.stderr.write(self.style.ERROR(f"ImageFile {image_id}: correction failed -- {error}"))
        self.stdout.write(self.style.SUCCESS(
            f"Corrected {result['corrected']}, failed {result['failed']}."
        ))
