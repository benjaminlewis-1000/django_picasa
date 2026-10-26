"""One shard of the one-time parallel LVFace-B/insightface catch-up
backfill -- see backfill_lvface_parallel.py (the supervisor that spawns
these as separate OS processes, one per shard, each pinned to its own
block of CPU cores via os.sched_setaffinity() right after spawning) for
the full design and reasoning.

A thin wrapper around FaceExtractor.reencode_missing_faces(shard_index=,
total_shards=, lvface_max_runtime_seconds=None) -- runs until that
shard's own id-partitioned queryset is exhausted, then exits naturally
(exit code 0, the signal the supervisor watches for to mark this shard
done rather than relaunch it). A non-zero exit (a real crash) leaves the
shard's own advisory lock released (Postgres releases it automatically
when this process's connection drops, regardless of how it exits -- see
common/advisory_lock.py) and the supervisor will simply retry it later.
"""
from django.core.management.base import BaseCommand, CommandError

from face_extract_encode import FaceExtractor


class Command(BaseCommand):
    help = (
        "Run one shard of the parallel reencode_missing_faces() backfill "
        "-- not meant to be run by hand, see backfill_lvface_parallel."
    )

    def add_arguments(self, parser):
        parser.add_argument('--shard-index', type=int, required=True)
        parser.add_argument('--total-shards', type=int, required=True)

    def handle(self, *args, **options):
        shard_index = options['shard_index']
        total_shards = options['total_shards']
        if total_shards < 1 or not (0 <= shard_index < total_shards):
            raise CommandError(f"--shard-index must be in [0, {total_shards}) with --total-shards >= 1")

        self.stdout.write(f"Starting shard {shard_index}/{total_shards}...", ending='\n')
        self.stdout.flush()
        extractor = FaceExtractor()
        extractor.reencode_missing_faces(
            shard_index=shard_index, total_shards=total_shards,
            lvface_max_runtime_seconds=None,
        )
        self.stdout.write(f"Shard {shard_index}/{total_shards} exhausted its backlog.", ending='\n')
