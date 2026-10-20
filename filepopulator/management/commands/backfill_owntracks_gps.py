#! /usr/bin/env python

# Backfill: estimates GPS for photos from an explicitly opted-in list of
# cameras (CameraGpsBackfillConfig, toggled via a frontend page -- see
# that model's own docstring), from OwnTracks location history. Safely
# re-runnable -- only ever touches rows still missing GPS, so this is
# meant to end up as a recurring cron/Celery task, not a true one-off.
# See owntracks_gps.py's own docstring for the real per-camera offset
# detail.

import datetime

from django.core.management.base import BaseCommand, CommandError

from filepopulator.owntracks_gps import (
    OwnTracksConfigError, backfill_camera_gps, get_enabled_camera_models,
)


def _parse_date(value):
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        raise CommandError(f"Not a valid YYYY-MM-DD date: {value!r}")


class Command(BaseCommand):
    help = "Estimate GPS for allowlisted cameras' photos missing it, from OwnTracks location history."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print what would be matched; don't write or query-write anything.",
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Only process this many photos (useful for a quick test run).",
        )
        parser.add_argument(
            '--camera-model', action='append', default=None,
            help="Scope to just this camera_model (repeatable). Must already be enabled in "
                 "CameraGpsBackfillConfig -- this narrows a run for testing, it does not "
                 "enable a new camera (use the toggle page/API for that).",
        )
        parser.add_argument(
            '--start-date', type=_parse_date, default=None,
            help="YYYY-MM-DD -- only photos on/after this date (by dateTaken). Real motivating "
                 "case: a camera with years of history predating OwnTracks tracking entirely "
                 "would otherwise spend a real exiftool+HTTP round trip per photo just to "
                 "learn that, over and over.",
        )
        parser.add_argument(
            '--end-date', type=_parse_date, default=None,
            help="YYYY-MM-DD -- only photos on/before this date (by dateTaken).",
        )
        parser.add_argument(
            '--retry-failed', action='store_true',
            help="Also re-attempt photos already marked 'no_nearby_point' from a previous run "
                 "(e.g. after the OwnTracks service's own location history grew). Never "
                 "re-attempts 'no_offset_tag' -- that can't change. By default, any photo with "
                 "an existing attempt record of either kind is skipped entirely.",
        )

    def handle(self, *args, **options):
        camera_models = options['camera_model']
        if camera_models:
            unknown = set(camera_models) - set(get_enabled_camera_models())
            if unknown:
                raise CommandError(
                    f"Not enabled in CameraGpsBackfillConfig: {sorted(unknown)}. Enable it "
                    "there first before scoping a run to it."
                )

        try:
            counts = backfill_camera_gps(
                camera_models=camera_models,
                start_date=options['start_date'], end_date=options['end_date'],
                retry_failed=options['retry_failed'],
                dry_run=options['dry_run'], limit=options['limit'], log=self.stdout.write,
            )
        except OwnTracksConfigError as e:
            raise CommandError(str(e))

        self.stdout.write(
            f"{'Would match' if options['dry_run'] else 'Matched'} {counts['matched']} photo(s). "
            f"{counts['no_offset_tag']} skipped (no offset source), "
            f"{counts['no_nearby_point']} skipped (no nearby OwnTracks point)."
        )
