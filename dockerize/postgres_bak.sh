#! /bin/bash
set -e

# Run a backup to the /backup directory.
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
# To restore:
#   mkdir /tmp/restore_dir
#   zstd -d -T0 -c /backup/picasa_db.tar.zst | tar -x -C /tmp/restore_dir
#   pg_restore -U postgres -d <target_db> --no-owner --no-acl -j 4 /tmp/restore_dir/picasa_dump_dir
#   rm -rf /tmp/restore_dir

DUMP_DIR=/backup/picasa_dump_dir
rm -rf "$DUMP_DIR"

/usr/local/bin/pg_dump -U benjamin -w -F d --compress=none -j 4 -f "$DUMP_DIR" picasa

tar -cf - -C /backup picasa_dump_dir | zstd -T0 -12 -o /backup/picasa_db.tar.zst

rm -rf "$DUMP_DIR"
