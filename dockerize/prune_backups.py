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

No Django dependency -- pure filesystem + stdlib datetime + the local
psql/pg_restore/createdb/dropdb CLI tools (all present in this image
already, since this runs inside db_picasa itself), so it can run from cron
right after the dump completes.

Backup validity check: every time a daily backup ages out of the daily
window and becomes its ISO week's kept representative for the first time
(the "promotion" this module's classify() already computes), that specific
file gets restored into a scratch DB and sanity-checked before any pruning
happens. This only needs to happen once per representative file, not again
when it later ages into the monthly tier -- it's the same bytes, already
checked. A failure writes a persistent marker (BACKUP_TEST_FAILED in
--backup-dir): every future run logs it loudly and refuses to delete
anything until a human investigates and removes the marker by hand.
"""

import argparse
import calendar
import datetime
import os
import re
import subprocess

FILENAME_RE = re.compile(r"^picasa_db_(\d{4}-\d{2}-\d{2})\.tar\.zst$")

DAILY_RETENTION_DAYS = 7
WEEKLY_RETENTION_DAYS = 35
MONTHLY_RETENTION_MONTHS = 3

DB_USER = "benjamin"
SCRATCH_DB = "picasa_backup_test"
EXTRACT_DIR = "/tmp/backup_restore_test"
FAILURE_MARKER_NAME = "BACKUP_TEST_FAILED"

# Not exhaustive -- just enough real tables, across enough apps, that a
# badly wrong restore (empty database, truncated archive, wrong DB dumped)
# would be caught. "Sane" here means "nonzero," not "matches live exactly"
# -- by the time a backup is a week old, exact parity isn't the point.
SANITY_TABLES = ["face_manager_face", "filepopulator_imagefile", "django_migrations"]


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


def find_newly_promoted_weekly(backups, today):
    """The filename becoming its ISO week's kept representative for the
    FIRST time today -- i.e. its date is exactly daily_cutoff, the one day
    it transitions from the daily tier into the weekly tier. On every
    later day it's still kept (now purely by the weekly tier), but it's
    not "newly" anything, so this returns None for it from then on --
    matching "we don't have to recheck ... since they're older copies of
    a first week backup." Returns None if nothing is transitioning today."""
    daily_cutoff = today - datetime.timedelta(days=DAILY_RETENTION_DAYS)
    weekly_cutoff = today - datetime.timedelta(days=WEEKLY_RETENTION_DAYS)

    weekly_seen = set()
    for date, fname in backups:
        if weekly_cutoff < date <= daily_cutoff:
            key = date.isocalendar()[:2]
            if key not in weekly_seen:
                weekly_seen.add(key)
                if date == daily_cutoff:
                    return fname
    return None


def failure_marker_path(backup_dir):
    return os.path.join(backup_dir, FAILURE_MARKER_NAME)


def restore_and_verify(backup_path):
    """Restore backup_path into a throwaway scratch DB and sanity-check
    it. Returns (ok, counts, problems, restore_log). Always cleans up the
    scratch DB and extracted files, pass or fail."""
    subprocess.run(["rm", "-rf", EXTRACT_DIR], check=False)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    try:
        extract = subprocess.run(
            f"zstd -d -T0 -c {backup_path} | tar -x -C {EXTRACT_DIR}",
            shell=True, capture_output=True, text=True,
        )
        if extract.returncode != 0:
            return False, {}, [f"extraction failed: {extract.stderr}"], extract.stderr

        dump_dir = os.path.join(EXTRACT_DIR, "picasa_dump_dir")

        subprocess.run(["dropdb", "-U", DB_USER, "--if-exists", SCRATCH_DB], check=False)
        create = subprocess.run(["createdb", "-U", DB_USER, SCRATCH_DB], capture_output=True, text=True)
        if create.returncode != 0:
            return False, {}, [f"createdb failed: {create.stderr}"], create.stderr

        restore = subprocess.run(
            ["pg_restore", "-U", DB_USER, "-d", SCRATCH_DB, "--no-owner", "--no-acl", "-j", "4", dump_dir],
            capture_output=True, text=True,
        )
        # pg_restore commonly exits nonzero on harmless warnings against a
        # --no-owner/--no-acl restore -- the row-count sanity check below
        # is the real pass/fail signal, not its exit code.
        restore_log = restore.stdout + restore.stderr

        counts = {}
        for table in SANITY_TABLES:
            r = subprocess.run(
                ["psql", "-U", DB_USER, "-d", SCRATCH_DB, "-t", "-A", "-c", f"SELECT COUNT(*) FROM {table};"],
                capture_output=True, text=True,
            )
            try:
                counts[table] = int(r.stdout.strip())
            except ValueError:
                counts[table] = -1

        problems = [f"{t}: {n} rows" for t, n in counts.items() if n <= 0]
        return len(problems) == 0, counts, problems, restore_log
    finally:
        subprocess.run(["dropdb", "-U", DB_USER, "--if-exists", SCRATCH_DB], check=False)
        subprocess.run(["rm", "-rf", EXTRACT_DIR], check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", default="/backup")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-restore-test", action="store_true",
        help="Skip the weekly-promotion restore sanity check (e.g. for manual/test runs).",
    )
    args = parser.parse_args()

    today = datetime.date.today()
    backups = parse_backups(args.backup_dir)
    marker_path = failure_marker_path(args.backup_dir)

    if os.path.exists(marker_path):
        print("!" * 70)
        print(f"BACKUP RESTORE TEST PREVIOUSLY FAILED -- see {marker_path} for details.")
        print("Pruning is PAUSED until this is investigated and the marker is removed by hand.")
        print("!" * 70)
        keep, delete = classify(backups, today)
        print(
            f"Found {len(backups)} dated backups. Would keep {len(keep)}, "
            f"would delete {len(delete)} -- SKIPPED because of the failed backup test."
        )
        return

    if not args.skip_restore_test:
        promoted = find_newly_promoted_weekly(backups, today)
        if promoted:
            print(f"Testing newly-promoted weekly backup: {promoted}")
            backup_path = os.path.join(args.backup_dir, promoted)
            ok, counts, problems, log = restore_and_verify(backup_path)
            if ok:
                print(f"Backup restore test PASSED for {promoted}: {counts}")
            else:
                print(f"Backup restore test FAILED for {promoted}: {problems}")
                with open(marker_path, "w") as f:
                    f.write(f"Backup restore test failed for {promoted} on {today.isoformat()}\n")
                    f.write(f"Counts: {counts}\nProblems: {problems}\n\nRestore log:\n{log}\n")
                print(f"Wrote {marker_path}. Pruning SKIPPED this run.")
                return

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
