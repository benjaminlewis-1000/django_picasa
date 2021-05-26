#! /bin/bash

# Run a backup to the /backup directory.

/usr/local/bin/pg_dump -U benjamin -w -F t picasa > /backup/picasa_db.tar
