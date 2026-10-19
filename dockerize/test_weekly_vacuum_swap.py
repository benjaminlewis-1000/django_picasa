#!/usr/bin/env python3
"""Tests for weekly_vacuum_swap.py's --only-first-monday gate. Plain
unittest, no Django, no Docker/Postgres -- mirrors
test_prune_backups.py's own "runs anywhere" design. The real dump/
restore/promote machinery needs a live DB and the Docker socket, so it
isn't covered here; this only covers the pure day-of-month gate added
2026-10-19 after a real cron scheduling bug (see crontab_root's own
comment) let --promote fire on every Monday instead of just the first.

Run with: python3 -m unittest dockerize.test_weekly_vacuum_swap -v
      or: python3 test_weekly_vacuum_swap.py -v   (from inside dockerize/)
"""
import datetime
import sys
import unittest
from unittest.mock import patch

import weekly_vacuum_swap as wvs


class OnlyFirstMondayGateTests(unittest.TestCase):

    def _run_main(self, argv, now):
        with patch.object(sys, 'argv', ['weekly_vacuum_swap.py'] + argv), \
             patch.object(wvs, 'datetime') as mock_dt, \
             patch.object(wvs, 'promote') as mock_promote, \
             patch.object(wvs, 'rehearse') as mock_rehearse, \
             self.assertRaises(SystemExit) as cm:
            mock_dt.now.return_value = now
            wvs.main()
        return cm.exception.code, mock_promote, mock_rehearse

    def test_promote_skipped_outside_first_week_when_gated(self):
        # A real Monday that is NOT in the first week (the exact shape
        # of the actual incident -- 2026-10-19 is a Monday, day 19).
        now = datetime.datetime(2026, 10, 19, 3, 0, 0)
        code, mock_promote, mock_rehearse = self._run_main(
            ['--promote', '--only-first-monday'], now,
        )
        self.assertEqual(code, 0)
        mock_promote.assert_not_called()
        mock_rehearse.assert_not_called()

    def test_promote_runs_in_first_week_when_gated(self):
        now = datetime.datetime(2026, 10, 5, 3, 0, 0)  # the real first Monday of Oct 2026
        mock_promote_return = True
        with patch.object(sys, 'argv', ['weekly_vacuum_swap.py', '--promote', '--only-first-monday']), \
             patch.object(wvs, 'datetime') as mock_dt, \
             patch.object(wvs, 'promote', return_value=mock_promote_return) as mock_promote, \
             self.assertRaises(SystemExit) as cm:
            mock_dt.now.return_value = now
            wvs.main()
        mock_promote.assert_called_once()
        self.assertEqual(cm.exception.code, 0)

    def test_promote_without_gate_always_runs_regardless_of_date(self):
        # A manual --promote (no --only-first-monday) must never be
        # blocked by this gate, per the script's own docstring ("can
        # still be run by hand ... any time in between").
        now = datetime.datetime(2026, 10, 19, 3, 0, 0)
        with patch.object(sys, 'argv', ['weekly_vacuum_swap.py', '--promote']), \
             patch.object(wvs, 'datetime') as mock_dt, \
             patch.object(wvs, 'promote', return_value=True) as mock_promote, \
             self.assertRaises(SystemExit):
            mock_dt.now.return_value = now
            wvs.main()
        mock_promote.assert_called_once()

    def test_rehearse_is_never_affected_by_the_gate(self):
        now = datetime.datetime(2026, 10, 19, 3, 0, 0)
        with patch.object(sys, 'argv', ['weekly_vacuum_swap.py', '--rehearse']), \
             patch.object(wvs, 'datetime') as mock_dt, \
             patch.object(wvs, 'rehearse', return_value=True) as mock_rehearse, \
             self.assertRaises(SystemExit):
            mock_dt.now.return_value = now
            wvs.main()
        mock_rehearse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
