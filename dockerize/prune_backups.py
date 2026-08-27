#!/usr/bin/env python3
"""Backup retention pruning for /backup/picasa_db_YYYY-MM-DD.tar.zst files.

Three tiers, applied in order:
  - Daily:   keep every backup from the last DAILY_RETENTION_DAYS days.
  - Weekly:  for backups older than that but within WEEKLY_RETENTION_DAYS,
             keep one per ISO week (the earliest backup found in that week).
  - Monthly: for backups older than that but within MONTHLY_RETENTION_MONTHS,
             keep one per calendar month (the earliest backup found in that
             month).
Anything older than MONTHLY_RETENTION_MONTHS is deleted outright, regardless
of tier.

No Django/DB dependency -- pure filesystem + stdlib datetime, so it can run
from cron in the (minimal) db_picasa image right after the dump completes.
"""

import argparse
import calendar
import datetime
import os
import re

FILENAME_RE = re.compile(r"^picasa_db_(\d{4}-\d{2}-\d{2})\.tar\.zst$")

DAILY_RETENTION_DAYS = 7
WEEKLY_RETENTION_DAYS = 35
MONTHLY_RETENTION_MONTHS = 3


def months_before(d, n):
    """d minus n calendar months, clamping the day to the target month's length."""
    month_index = d.month - 1 - n
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def parse_backups(backup_dir):
    backups = []
    for fname in os.listdir(backup_dir):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        date = datetime.datetime.strptime(m.group(1), "%Y-%m-%d").date()
        backups.append((date, fname))
    return sorted(backups)


def classify(backups, today):
    """Returns (keep, delete) sets of filenames."""
    daily_cutoff = today - datetime.timedelta(days=DAILY_RETENTION_DAYS)
    weekly_cutoff = today - datetime.timedelta(days=WEEKLY_RETENTION_DAYS)
    monthly_cutoff = months_before(today, MONTHLY_RETENTION_MONTHS)

    keep = set()

    # Daily tier: everything newer than daily_cutoff.
    for date, fname in backups:
        if date > daily_cutoff:
            keep.add(fname)

    # Weekly tier: one per ISO week.
    weekly_seen = set()
    for date, fname in backups:
        if weekly_cutoff < date <= daily_cutoff:
            key = date.isocalendar()[:2]  # (iso_year, iso_week)
            if key not in weekly_seen:
                weekly_seen.add(key)
                keep.add(fname)

    # Monthly tier: one per calendar month, but never older than the hard cap.
    monthly_seen = set()
    for date, fname in backups:
        if monthly_cutoff <= date <= weekly_cutoff:
            key = (date.year, date.month)
            if key not in monthly_seen:
                monthly_seen.add(key)
                keep.add(fname)

    all_names = {fname for _, fname in backups}
    delete = all_names - keep
    return keep, delete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", default="/backup")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    today = datetime.date.today()
    backups = parse_backups(args.backup_dir)
    keep, delete = classify(backups, today)

    print(f"Found {len(backups)} dated backups. Keeping {len(keep)}, deleting {len(delete)}.")
    for date, fname in backups:
        if fname in delete:
            path = os.path.join(args.backup_dir, fname)
            if args.dry_run:
                print(f"Would delete: {fname}")
            else:
                print(f"Deleting: {fname}")
                os.remove(path)


if __name__ == "__main__":
    main()
