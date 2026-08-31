#!/usr/bin/env python3
"""Tests for prune_backups.py. Plain unittest, no Django -- matches the
script's own "no Django dependency" design so it can run anywhere Python
3 is available, including inside the minimal db_picasa image.

Run with: python3 -m unittest dockerize.test_prune_backups -v
      or: python3 test_prune_backups.py -v   (from inside dockerize/)
"""
import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prune_backups as pb


def d(s):
    return datetime.date.fromisoformat(s)


class ParseBackupsTests(unittest.TestCase):
    def test_parses_dated_backups_and_ignores_other_files(self):
        tmp = tempfile.mkdtemp()
        try:
            for name in ["picasa_db_2026-08-01.tar.zst", "picasa_db_2026-08-15.tar.zst",
                         "not_a_backup.txt", "picasa_db_bad-date.tar.zst"]:
                open(os.path.join(tmp, name), "w").close()
            backups = pb.parse_backups(tmp)
            self.assertEqual(
                backups,
                [(d("2026-08-01"), "picasa_db_2026-08-01.tar.zst"),
                 (d("2026-08-15"), "picasa_db_2026-08-15.tar.zst")],
            )
        finally:
            shutil.rmtree(tmp)


class ClassifyTests(unittest.TestCase):
    def test_daily_tier_keeps_everything_within_window(self):
        today = d("2026-08-31")
        backups = [(today - datetime.timedelta(days=i), f"picasa_db_{today - datetime.timedelta(days=i)}.tar.zst")
                   for i in range(7)]
        keep, delete = pb.classify(backups, today)
        self.assertEqual(len(keep), 7)
        self.assertEqual(len(delete), 0)

    def test_weekly_tier_keeps_only_earliest_per_iso_week(self):
        # Two backups in the same ISO week, both past the daily window.
        today = d("2026-08-31")
        backups = [
            (d("2026-08-10"), "picasa_db_2026-08-10.tar.zst"),  # earlier in its ISO week
            (d("2026-08-12"), "picasa_db_2026-08-12.tar.zst"),  # later, same ISO week
        ]
        self.assertEqual(backups[0][0].isocalendar()[:2], backups[1][0].isocalendar()[:2])
        keep, delete = pb.classify(backups, today)
        self.assertIn("picasa_db_2026-08-10.tar.zst", keep)
        self.assertIn("picasa_db_2026-08-12.tar.zst", delete)

    def test_older_than_monthly_cap_is_deleted(self):
        today = d("2026-08-31")
        ancient = today - datetime.timedelta(days=400)
        backups = [(ancient, f"picasa_db_{ancient}.tar.zst")]
        keep, delete = pb.classify(backups, today)
        self.assertEqual(delete, {f"picasa_db_{ancient}.tar.zst"})


class FindNewlyPromotedWeeklyTests(unittest.TestCase):
    def test_no_backups_returns_none(self):
        self.assertIsNone(pb.find_newly_promoted_weekly([], d("2026-08-31")))

    def test_backup_at_exact_daily_cutoff_is_newly_promoted(self):
        today = d("2026-08-31")
        cutoff = today - datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        fname = f"picasa_db_{cutoff}.tar.zst"
        backups = [(cutoff, fname)]
        self.assertEqual(pb.find_newly_promoted_weekly(backups, today), fname)

    def test_backup_not_at_cutoff_boundary_is_not_newly_promoted(self):
        today = d("2026-08-31")
        cutoff = today - datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        older = cutoff - datetime.timedelta(days=3)  # already past the boundary on a prior day
        fname = f"picasa_db_{older}.tar.zst"
        backups = [(older, fname)]
        self.assertIsNone(pb.find_newly_promoted_weekly(backups, today))

    def test_second_backup_in_same_week_at_cutoff_is_not_promoted(self):
        """If an earlier backup already claimed this ISO week's slot, a
        second backup from the same week reaching the cutoff boundary
        isn't a NEW promotion -- the week's representative was already
        tested when the first one crossed the boundary some days ago."""
        cutoff = d("2026-08-12")  # a Wednesday
        earlier_same_week = d("2026-08-11")  # the Tuesday before it, same ISO week
        self.assertEqual(cutoff.isocalendar()[:2], earlier_same_week.isocalendar()[:2])
        today = cutoff + datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        backups = [
            (earlier_same_week, f"picasa_db_{earlier_same_week}.tar.zst"),
            (cutoff, f"picasa_db_{cutoff}.tar.zst"),
        ]
        self.assertIsNone(pb.find_newly_promoted_weekly(backups, today))

    def test_only_returns_for_the_transition_day_not_before_or_after(self):
        cutoff_date = d("2026-08-01")
        fname = f"picasa_db_{cutoff_date}.tar.zst"
        backups = [(cutoff_date, fname)]

        day_of_transition = cutoff_date + datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        self.assertEqual(pb.find_newly_promoted_weekly(backups, day_of_transition), fname)

        day_before = day_of_transition - datetime.timedelta(days=1)
        self.assertIsNone(pb.find_newly_promoted_weekly(backups, day_before))

        day_after = day_of_transition + datetime.timedelta(days=1)
        self.assertIsNone(pb.find_newly_promoted_weekly(backups, day_after))


class MainFlowTests(unittest.TestCase):
    """Covers main()'s marker-file gating logic with restore_and_verify
    mocked out -- the real restore/psql calls are exercised separately as
    a live integration check against db_picasa, not here."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _touch(self, fname):
        open(os.path.join(self.tmp, fname), "w").close()

    def test_existing_marker_blocks_pruning_without_retesting(self):
        marker = pb.failure_marker_path(self.tmp)
        with open(marker, "w") as f:
            f.write("previous failure")

        deletable = d("2026-01-01")  # old enough it would otherwise be deleted
        self._touch(f"picasa_db_{deletable}.tar.zst")

        with patch("prune_backups.restore_and_verify") as mock_restore, \
             patch("sys.argv", ["prune_backups.py", "--backup-dir", self.tmp]):
            pb.main()
            mock_restore.assert_not_called()

        self.assertTrue(os.path.exists(os.path.join(self.tmp, f"picasa_db_{deletable}.tar.zst")))

    def test_failed_restore_test_writes_marker_and_skips_pruning(self):
        today = datetime.date.today()
        cutoff = today - datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        promoted_fname = f"picasa_db_{cutoff}.tar.zst"
        self._touch(promoted_fname)
        old_fname = f"picasa_db_2020-01-01.tar.zst"
        self._touch(old_fname)

        with patch("prune_backups.restore_and_verify", return_value=(False, {"x": 0}, ["x: 0 rows"], "log")), \
             patch("sys.argv", ["prune_backups.py", "--backup-dir", self.tmp]):
            pb.main()

        self.assertTrue(os.path.exists(pb.failure_marker_path(self.tmp)))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, old_fname)))

    def test_passed_restore_test_allows_normal_pruning(self):
        today = datetime.date.today()
        cutoff = today - datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        promoted_fname = f"picasa_db_{cutoff}.tar.zst"
        self._touch(promoted_fname)
        ancient = today - datetime.timedelta(days=400)
        ancient_fname = f"picasa_db_{ancient}.tar.zst"
        self._touch(ancient_fname)

        with patch("prune_backups.restore_and_verify", return_value=(True, {"x": 5}, [], "log")), \
             patch("sys.argv", ["prune_backups.py", "--backup-dir", self.tmp]):
            pb.main()

        self.assertFalse(os.path.exists(pb.failure_marker_path(self.tmp)))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, ancient_fname)))

    def test_skip_restore_test_flag_never_calls_restore(self):
        today = datetime.date.today()
        cutoff = today - datetime.timedelta(days=pb.DAILY_RETENTION_DAYS)
        self._touch(f"picasa_db_{cutoff}.tar.zst")

        with patch("prune_backups.restore_and_verify") as mock_restore, \
             patch("sys.argv", ["prune_backups.py", "--backup-dir", self.tmp, "--skip-restore-test"]):
            pb.main()
            mock_restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
