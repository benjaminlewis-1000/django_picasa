#! /bin/bash
set -e

# Run a dated backup to the /backup directory, then prune old generations
# per prune_backups.py's retention policy (7 daily, then one/week through
# ~5 weeks, then one/month through 3 months, delete anything older).
#
# Directory-format dump (-F d --compress=none) + external multi-threaded
# zstd compression, rather than pg_dump's own -Z compressor (which is
# single-threaded regardless of format -- custom-format gzip took 17m23s
# for a ~3.4GB backup; this approach did the same DB in ~11.5m for
# ~2.3GB). Neither approach beats the other by much on *size* alone --
# this DB is dominated by face_manager_face's high-entropy float
# embeddings, and pg_dump's text-based COPY representation inflates them
# well past what any single-threaded compressor claws back (measured
# 2026-08-26: pg_dump's own -Z zstd:9 and -Z 6 both landed close to, or
# above, the live table's already-TOAST-compressed on-disk size). The
# real win of this approach is *speed*, from actually using more than one
# core.
#
# To restore a given generation:
#   mkdir /tmp/restore_dir
#   zstd -d -T0 -c /backup/picasa_db_YYYY-MM-DD.tar.zst | tar -x -C /tmp/restore_dir
#   cat /tmp/restore_dir/picasa_dump_dir/REQUIRED_EXTENSIONS.txt  # check this FIRST
#   pg_restore -U postgres -d <target_db> --no-owner --no-acl -j 4 /tmp/restore_dir/picasa_dump_dir
#   rm -rf /tmp/restore_dir
#
# IMPORTANT (2026-10-29, added while scoping a possible pgvector/halfvec
# embedding-storage migration -- see CLAUDE.md): if REQUIRED_EXTENSIONS.txt
# lists anything beyond plpgsql, the TARGET Postgres restoring this backup
# must have that extension's shared library installed (`CREATE EXTENSION`
# needs the .so/.control files present on disk, not just database
# permissions) -- confirmed via a real spike: restoring a pgvector-using
# dump into a vanilla Postgres without the extension built in fails loudly
# (pg_restore exits nonzero), but cascades through several follow-on
# "relation does not exist" errors that can obscure the actual root cause.
# Check the manifest FIRST, before running pg_restore, for a clean answer
# instead of diagnosing a cascade. This matters concretely for this
# project: `db_picasa_dev` and any future disaster-recovery host are NOT
# guaranteed to have the same extensions as whatever produced the backup
# unless explicitly built that way (see dockerize/Dockerfile_postgres).

DATE=$(date +%F)
DUMP_DIR=/backup/picasa_dump_dir
rm -rf "$DUMP_DIR"

/usr/local/bin/pg_dump -U benjamin -w -F d --compress=none -j 4 -f "$DUMP_DIR" picasa

# Self-describing manifest: lets a human (or script) restoring this backup
# on a different host know what's required WITHOUT needing to inspect the
# dump's binary TOC first. plpgsql is always present on any stock Postgres
# and excluded as noise -- only extensions that could plausibly be MISSING
# on a restore target are worth flagging.
/usr/local/bin/psql -U benjamin -w -d picasa -tAc \
    "SELECT extname FROM pg_extension WHERE extname != 'plpgsql' ORDER BY extname;" \
    > "$DUMP_DIR/REQUIRED_EXTENSIONS.txt"

# Git checkout snapshot (2026-10-29), per the user's explicit request --
# rides alongside each DB backup so a future restore also pairs with the
# exact code (including any uncommitted live hotfix, not just whatever
# `master` happens to be at restore time) that was actually running
# against that data. db_picasa has no access to /code itself -- this
# reuses the SAME cross-container docker-cli pattern weekly_vacuum_
# swap.py already established, exec'ing into picasa_api, where the real
# checkout lives (requires `git` + a safe.directory exception there,
# since the container runs as root over a host-owned bind mount -- see
# dockerize/Dockerfile_picasa). Deliberately best-effort: `set +e`
# around this block so a git/docker hiccup (picasa_api down, etc.) can
# never abort the actual DB backup below, which is the critical part --
# any failure just lands as plain text in the snapshot file instead.
set +e
{
    echo "=== git log -1 ==="
    docker exec picasa_api git -C /code log -1 --format='%H%n%ci%n%s'
    echo ""
    echo "=== git status --short (uncommitted changes, if any) ==="
    docker exec picasa_api git -C /code status --short
    echo ""
    echo "=== git diff (full content of any uncommitted changes above) ==="
    docker exec picasa_api git -C /code diff
} > "$DUMP_DIR/GIT_CHECKOUT.txt" 2>&1
set -e

tar -cf - -C /backup picasa_dump_dir | zstd -T0 -12 -o "/backup/picasa_db_${DATE}.tar.zst"

rm -rf "$DUMP_DIR"

python3 /etc/periodic/daily/prune_backups.py --backup-dir /backup
