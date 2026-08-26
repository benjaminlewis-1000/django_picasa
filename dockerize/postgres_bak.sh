#! /bin/bash

# Run a backup to the /backup directory.
#
# Custom format (-F c), not tar (-F t): tar format doesn't support
# compression at all, and pg_dump's text-based COPY representation of
# large float-array columns (face_manager_face's 512-d/128-d embeddings)
# is itself ~2-2.5x larger than their raw binary size -- the combination
# made prior tar-format backups (~7.5GB) run nearly 3x the live DB's
# actual on-disk size (~2.7GB). Custom format supports pg_dump's own
# compression and restores via the same pg_restore workflow.
/usr/local/bin/pg_dump -U benjamin -w -F c -Z 6 picasa > /backup/picasa_db.tar
