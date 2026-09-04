#!/usr/bin/env python3
"""Low-downtime VACUUM FULL via dump/restore/promote -- scheduled MONTHLY
(first Monday, 3am) as of 2026-09-04, downgraded from weekly. It exists
to reclaim disk space in the LIVE DB's on-disk footprint, which is
unrelated to backup size (pg_dump/postgres_bak.sh only ever dumps live
row data, never dead tuples/free space, regardless of how bloated the
source table is) -- weekly picasa_api downtime wasn't worth it for pure
disk reclaim, monthly is the compromise (see crontab_root's own note).
Can still be run by hand (--rehearse first, then --promote) any time in
between if live disk usage gets tight before the next scheduled run.

VACUUM FULL run in-place would exclusive-lock face_manager_face (the most
actively-written table) for its entire runtime, blocking all live traffic
and every scheduled face task for that whole window. Instead: dump the
live DB, restore it into a scratch DB (a freshly-restored table has no
bloat at all -- no separate VACUUM FULL is needed on the copy), verify
it's sane, then swap it in to replace the live DB.

Runs INSIDE db_picasa itself (bind-mounted), using the local
psql/pg_dump/pg_restore/createdb/dropdb tools directly for everything
DB-related. The one thing it can't do locally is control the separate
picasa_api container -- that goes through `docker stop`/`docker
start`/`docker exec` against the host's Docker socket, which db_picasa
has mounted in for exactly this purpose (see docker-compose.yaml's
db_django service).

Two modes:
  --rehearse (safe, default): dump+restore+verify into a scratch DB while
      the live app keeps running. Reports results, drops the scratch DB,
      touches nothing else. Run this manually as many times as you like
      before ever scheduling the real thing.
  --promote (the real, destructive weekly operation): stops picasa_api,
      dumps+restores+verifies fresh (with the app stopped, so this is a
      fully consistent snapshot -- no window where writes could be lost),
      renames the live DB aside (kept, not dropped, for
      KEEP_PREVACUUM_GENERATIONS runs) and the scratch DB into its place,
      restarts picasa_api, and health-checks the restart. Aborts loudly
      before touching anything live if verification fails at any point.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime

APP_CONTAINER = "picasa_api"
DB_USER = "benjamin"
LIVE_DB = "picasa"
SCRATCH_DB = "picasa_vacuum_scratch"
PREVACUUM_PREFIX = "picasa_prevacuum_"
KEEP_PREVACUUM_GENERATIONS = 2

# Tables whose row counts are checked between live and scratch as a sanity
# gate before ever promoting the scratch DB. Not exhaustive -- just enough
# real tables, across enough apps, that a badly wrong restore (empty
# database, wrong DB dumped, truncated restore) would be caught.
SANITY_TABLES = [
    "face_manager_face",
    "face_manager_person",
    "filepopulator_imagefile",
    "filepopulator_directory",
    "django_migrations",
]


def run(cmd, check=True, capture=True, shell=False):
    result = subprocess.run(cmd, capture_output=capture, text=True, shell=shell)
    if check and result.returncode != 0:
        label = cmd if shell else ' '.join(cmd)
        raise RuntimeError(f"Command failed ({result.returncode}): {label}\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return result


def docker(cmd, check=True):
    return run(["docker"] + cmd, check=check)


def psql(dbname, sql):
    result = run(["psql", "-U", DB_USER, "-d", dbname, "-t", "-A", "-c", sql])
    return result.stdout.strip()


def table_row_counts(dbname):
    counts = {}
    for table in SANITY_TABLES:
        out = psql(dbname, f"SELECT COUNT(*) FROM {table};")
        counts[table] = int(out)
    return counts


def dropdb_if_exists(dbname):
    run(["dropdb", "-U", DB_USER, "--if-exists", dbname])


def createdb(dbname):
    run(["createdb", "-U", DB_USER, dbname])


def dump_and_restore(source_db, target_db):
    # Piped locally (both processes run inside this same container), so
    # the dump never touches disk as an intermediate file -- it flows
    # through a pipe in memory.
    cmd = (
        f"pg_dump -U {DB_USER} -F c {source_db} | "
        f"pg_restore -U {DB_USER} -d {target_db} --no-owner --no-acl"
    )
    result = run(cmd, shell=True, check=False)
    # pg_restore commonly exits nonzero on harmless warnings (e.g. "role
    # does not exist" for --no-owner dumps against a differently-owned
    # source) -- treat it as fatal only if the target ended up empty.
    counts = table_row_counts(target_db)
    if all(v == 0 for v in counts.values()):
        raise RuntimeError(
            f"Restore into {target_db} produced an empty database. "
            f"pg_restore stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return counts


def verify(live_counts, scratch_counts, exact):
    problems = []
    for table in SANITY_TABLES:
        live_n = live_counts[table]
        scratch_n = scratch_counts[table]
        if scratch_n == 0 and live_n > 0:
            problems.append(f"{table}: scratch has 0 rows, live has {live_n}")
        elif exact and scratch_n != live_n:
            problems.append(f"{table}: scratch={scratch_n} != live={live_n}")
        elif not exact and abs(scratch_n - live_n) > max(50, live_n * 0.02):
            # Rehearsal mode: live traffic may have written a handful of
            # rows between the dump snapshot and this check. Allow a
            # small drift (2% or 50 rows, whichever is larger); anything
            # bigger is worth flagging rather than silently accepting.
            problems.append(f"{table}: scratch={scratch_n} vs live={live_n} (drift too large)")
    return problems


def health_check(retries=10, delay=2):
    for _ in range(retries):
        result = docker(["exec", APP_CONTAINER, "python", "manage.py", "check"], check=False)
        if result.returncode == 0:
            return True
        time.sleep(delay)
    return False


def prune_prevacuum_generations():
    result = run(["psql", "-U", DB_USER, "-l", "-t", "-A"])
    prevacuum_dbs = sorted(
        line.split("|")[0] for line in result.stdout.splitlines()
        if line.split("|")[0].startswith(PREVACUUM_PREFIX)
    )
    to_drop = prevacuum_dbs[:-KEEP_PREVACUUM_GENERATIONS] if KEEP_PREVACUUM_GENERATIONS else prevacuum_dbs
    for dbname in to_drop:
        print(f"Pruning old pre-vacuum generation: {dbname}")
        dropdb_if_exists(dbname)


def rehearse():
    print(f"[{datetime.now()}] Rehearsal: dump+restore+verify against a scratch DB, app stays live.")
    dropdb_if_exists(SCRATCH_DB)
    createdb(SCRATCH_DB)
    try:
        live_counts = table_row_counts(LIVE_DB)
        print(f"Live counts: {live_counts}")
        scratch_counts = dump_and_restore(LIVE_DB, SCRATCH_DB)
        print(f"Scratch counts: {scratch_counts}")
        problems = verify(live_counts, scratch_counts, exact=False)
        if problems:
            print("REHEARSAL FAILED:")
            for p in problems:
                print(f"  - {p}")
            return False
        print("Rehearsal passed -- restore is sane. Live DB and app were never touched.")
        return True
    finally:
        dropdb_if_exists(SCRATCH_DB)


def promote():
    print(f"[{datetime.now()}] Starting real weekly vacuum swap (app will be stopped).")
    dropdb_if_exists(SCRATCH_DB)
    createdb(SCRATCH_DB)

    print(f"Stopping {APP_CONTAINER}...")
    docker(["stop", APP_CONTAINER])

    try:
        live_counts = table_row_counts(LIVE_DB)
        scratch_counts = dump_and_restore(LIVE_DB, SCRATCH_DB)
        problems = verify(live_counts, scratch_counts, exact=True)
        if problems:
            print("VERIFICATION FAILED -- aborting before touching the live DB:")
            for p in problems:
                print(f"  - {p}")
            dropdb_if_exists(SCRATCH_DB)
            return False

        # Underscores, not hyphens -- a hyphen in an unquoted Postgres
        # identifier is a syntax error (matches the existing
        # picasa_pre_reset_2026_08_26 naming precedent).
        date_tag = datetime.now().strftime("%Y_%m_%d")
        prevacuum_name = f"{PREVACUUM_PREFIX}{date_tag}"
        print(f"Verification passed. Renaming {LIVE_DB} -> {prevacuum_name}, {SCRATCH_DB} -> {LIVE_DB}.")
        psql("postgres", f"ALTER DATABASE {LIVE_DB} RENAME TO {prevacuum_name};")
        try:
            psql("postgres", f"ALTER DATABASE {SCRATCH_DB} RENAME TO {LIVE_DB};")
        except Exception:
            # The first rename succeeded but the second didn't -- there is
            # currently NO database named LIVE_DB at all. Roll the first
            # rename back so the app has something to connect to, rather
            # than leaving it broken on top of an already-broken swap.
            print(f"Second rename failed -- rolling back: {prevacuum_name} -> {LIVE_DB}")
            psql("postgres", f"ALTER DATABASE {prevacuum_name} RENAME TO {LIVE_DB};")
            raise

        prune_prevacuum_generations()
        return True
    finally:
        print(f"Starting {APP_CONTAINER}...")
        docker(["start", APP_CONTAINER])
        if not health_check():
            print(
                f"ALERT: {APP_CONTAINER} did not pass its health check after restart. "
                "Manual intervention needed.",
                file=sys.stderr,
            )
            sys.exit(2)
        print("Health check passed -- app is back up.")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rehearse", action="store_true", help="Safe: dump+restore+verify only, app stays live.")
    mode.add_argument("--promote", action="store_true", help="Real: stops the app and swaps the DB.")
    args = parser.parse_args()

    ok = rehearse() if args.rehearse else promote()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
