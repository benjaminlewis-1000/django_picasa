"""One-time supervisor for a parallel reencode_missing_faces() backfill
(2026-10-25) -- not a permanent feature, just a faster way to work
through the large existing image LVFace-B/insightface backlog (see
Face.face_encoding_lvface_b's own docstring in models.py) than the
single-process shape the hourly face_manager.reencode task uses, which
stays unchanged and keeps handling routine small ongoing catch-up on its
own once this backlog clears.

Real constraint this design works around: insightface's own FaceAnalysis
hardcodes sess_options=None when building each ONNX InferenceSession (see
FaceExtractor.REINIT_DETECTOR_EVERY_N_IMAGES's own comment -- already
found true for arena configuration, same limitation applies to thread-
count configuration), so there's no supported way to tell a worker
process's own ONNXRuntime sessions to use fewer threads. Plain Python-
level threading was already tried and found useless for a similar
workload elsewhere in this codebase (classify_unassigned(), see
CLAUDE.md) for the same underlying reason -- a single inference call
already spans multiple cores via ONNXRuntime's own intra-op
parallelism, so extra Python threads just oversubscribe the same cores.
Real additional throughput needs real additional OS processes, each
constrained to its own slice of the host's cores -- enforced here via
os.sched_setaffinity() on each spawned subprocess, which works
regardless of how many threads ONNXRuntime tries to create internally,
since the OS scheduler simply won't run them anywhere outside the
assigned core set.

Real constraint #2, found empirically (2026-10-25) while running a
single unbounded instance even AFTER the periodic session-reinit memory
fix: one worker process still sits around ~12GB RSS (dominated by
loaded model weights -- buffalo_l's several ONNX models plus LVFace-B's
455MB model -- and ONNXRuntime's own bounded-but-sizeable working-set
arena, not something the reinit fix was ever meant to shrink further,
only to stop it growing WITHOUT bound). Under this container's 32GiB
limit, with gunicorn/celery's own routine baseline already accounting
for several GB, that caps real safe parallelism closer to 2-3 workers
than the naive CPU-core math alone would suggest (24 cores / ~4 cores-
per-worker could otherwise imply 4-6). Both constraints are sized
below and the supervisor takes whichever is more restrictive.

Sharding: the backlog is split by Face.id % total_shards (see
FaceExtractor.reencode_missing_faces()'s own shard_index/total_shards
docstring) -- a disjoint, deterministic partition, not a live work
queue. Each shard is its own `reencode_missing_faces_shard` subprocess,
holding its own distinctly-named advisory lock (so shards never block
each other, but relaunching an already-running shard is a safe no-op).

Scaling: every --check-interval-seconds (default 600 = 10 minutes, per
the user's explicit request), re-reads real system headroom (1-minute
load average via os.getloadavg(), available memory via /proc/meminfo's
MemAvailable) and recomputes a target worker count -- scaling up while
there's real idle capacity, scaling down if the host gets busier from
other genuine production work (the regular celery workers/gunicorn need
room too). Between those checks, a tighter polling loop just watches
for workers that have exited on their own (shard exhausted, or crashed)
so a freed slot can be reused promptly rather than waiting a full 10
minutes. A worker is never force-killed mid-shutdown in a way that
risks correctness -- SIGTERM/SIGKILL are both safe here, since Postgres
releases that shard's advisory lock automatically the moment the
process's DB connection drops (see common/advisory_lock.py), regardless
of how the process actually exits; a killed shard simply gets relaunched
(from scratch, which is cheap -- it's just a query, not lost work, since
every already-processed face was already individually committed).
"""
import os
import subprocess
import sys
import time

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Q

from face_manager.models import Face

# Observed empirically (2026-10-25, see this file's own module
# docstring) against the real production backlog -- one worker's own
# intra-op thread pool spans about this many cores under default
# ONNXRuntime sizing, and its RSS plateaus around this many GB once the
# periodic session-reinit fix is in place. Deliberately conservative
# (rounds up a little from the raw observed numbers) rather than tuned
# to the exact edge, since getting this wrong in the "too many workers"
# direction risks a real OOM, not just suboptimal throughput.
CORES_PER_WORKER = 5
MEMORY_PER_WORKER_GB = 14

# Cores/memory always left unused regardless of headroom math, for the
# host's other real, ongoing work (routine celery workers, gunicorn,
# everything else sharing this machine -- see this project's own
# CLAUDE.md for how busy this host already is outside this backfill).
DEFAULT_RESERVED_CORES = 6
DEFAULT_RESERVED_MEMORY_GB = 6


def _available_memory_gb():
    """The real, binding ceiling for spawning another worker: whichever
    is SMALLER of (a) the host's own available memory (/proc/meminfo's
    MemAvailable -- this reflects the host's view even from inside a
    container under the default cgroup setup, not a container-scoped
    figure) and (b) this container's own cgroup v2 memory headroom
    (memory.max - memory.current). Using only (a) would be a real bug:
    the host could show plenty of room while this container's own
    mem_limit (docker-compose.yaml) is the actual thing that kills a
    new worker first. Falls back to host-only if the cgroup v2 files
    aren't present (e.g. a cgroup v1 host, or not running in a
    container at all)."""
    host_available_gb = None
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                host_available_gb = int(line.split()[1]) / (1024 * 1024)
                break
    if host_available_gb is None:
        host_available_gb = 0.0

    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            raw_max = f.read().strip()
        if raw_max == 'max':
            return host_available_gb
        cgroup_max_gb = int(raw_max) / (1024 ** 3)
        with open('/sys/fs/cgroup/memory.current') as f:
            cgroup_current_gb = int(f.read().strip()) / (1024 ** 3)
        cgroup_available_gb = max(0.0, cgroup_max_gb - cgroup_current_gb)
        return min(host_available_gb, cgroup_available_gb)
    except (FileNotFoundError, ValueError):
        return host_available_gb


def _total_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Command(BaseCommand):
    help = (
        "One-time supervisor: runs the image LVFace-B/insightface backfill "
        "across multiple parallel worker processes, scaling the worker count "
        "to real system headroom every --check-interval-seconds."
    )

    def add_arguments(self, parser):
        parser.add_argument('--total-shards', type=int, default=8)
        parser.add_argument('--min-workers', type=int, default=1)
        parser.add_argument('--max-workers', type=int, default=4)
        parser.add_argument('--reserved-cores', type=int, default=DEFAULT_RESERVED_CORES)
        parser.add_argument('--reserved-memory-gb', type=float, default=DEFAULT_RESERVED_MEMORY_GB)
        parser.add_argument('--check-interval-seconds', type=int, default=600)
        parser.add_argument('--poll-interval-seconds', type=int, default=30)
        parser.add_argument(
            '--scale-down-confirmations', type=int, default=3,
            help=(
                "Scaling up acts immediately on any single headroom check "
                "that shows more room. Scaling down only acts once a lower "
                "target has been computed this many CONSECUTIVE checks in a "
                "row -- a transient dip (e.g. right after this supervisor's "
                "own restart, or a brief spike elsewhere on the host) "
                "shouldn't kill a worker that's mid-shard and about to be "
                "productive again. Resets to 0 the moment a check shows "
                "enough room again, so it's consecutive pressure, not a "
                "running total."
            ),
        )

    def handle(self, *args, **options):
        total_shards = options['total_shards']
        min_workers = options['min_workers']
        max_workers = min(options['max_workers'], total_shards)
        reserved_cores = options['reserved_cores']
        reserved_memory_gb = options['reserved_memory_gb']
        check_interval = options['check_interval_seconds']
        poll_interval = options['poll_interval_seconds']
        scale_down_confirmations = options['scale_down_confirmations']

        total_cores = _total_cores()
        usable_cores = max(0, total_cores - reserved_cores)
        self.stdout.write(
            f"Host: {total_cores} cores ({reserved_cores} reserved, {usable_cores} usable), "
            f"total_shards={total_shards}, worker range [{min_workers}, {max_workers}], "
            f"~{CORES_PER_WORKER} cores / ~{MEMORY_PER_WORKER_GB}GB per worker, "
            f"headroom re-checked every {check_interval}s."
        )
        self.stdout.flush()

        shard_status = {i: 'pending' for i in range(total_shards)}
        workers = {}  # shard_index -> (Popen, assigned_core_list)
        next_core = 0
        last_headroom_check = 0.0
        consecutive_low_target = 0

        def log_progress():
            running = sorted(workers.keys())
            done = sorted(i for i, s in shard_status.items() if s == 'done')
            pending = sorted(i for i, s in shard_status.items() if s == 'pending')
            img_remaining = Face.objects.filter(
                face_encoding_lvface_b__isnull=True, kps__isnull=False, source_image_file__isnull=False,
            ).exclude(declared_name__person_name__in=[settings.SOFT_IGNORE_NAME, '.realignore']).count()
            self.stdout.write(
                f"[{time.strftime('%H:%M:%S')}] running={running} done={done} pending={pending} "
                f"(image backlog still flagged missing, includes below-floor: {img_remaining})"
            )
            self.stdout.flush()

        def launch(shard_index):
            nonlocal next_core
            cmd = [
                sys.executable, 'manage.py', 'reencode_missing_faces_shard',
                '--shard-index', str(shard_index), '--total-shards', str(total_shards),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cores = [c % total_cores for c in range(next_core, next_core + CORES_PER_WORKER)]
            next_core = (next_core + CORES_PER_WORKER) % max(total_cores, 1)
            try:
                os.sched_setaffinity(proc.pid, set(cores))
            except (AttributeError, OSError, ProcessLookupError) as e:
                # Affinity pinning is a best-effort throughput/memory
                # optimization, not a correctness requirement -- if it
                # fails for any reason (e.g. the process already exited
                # before we could pin it, or a platform without
                # sched_setaffinity), the worker still runs correctly,
                # just without the dedicated-core guarantee.
                settings.LOGGER.warning(f"Could not pin shard {shard_index} (pid {proc.pid}) to cores {cores}: {e}")
            workers[shard_index] = (proc, cores)
            shard_status[shard_index] = 'running'
            self.stdout.write(f"Launched shard {shard_index}/{total_shards} (pid {proc.pid}, cores {cores}).")
            self.stdout.flush()

        def reap_finished():
            for shard_index, (proc, cores) in list(workers.items()):
                ret = proc.poll()
                if ret is None:
                    continue
                del workers[shard_index]
                if ret == 0:
                    shard_status[shard_index] = 'done'
                    self.stdout.write(f"Shard {shard_index}/{total_shards} finished cleanly.")
                else:
                    shard_status[shard_index] = 'pending'
                    self.stdout.write(f"Shard {shard_index}/{total_shards} exited with code {ret} -- will retry.")
                self.stdout.flush()

        def compute_target_worker_count():
            load1 = os.getloadavg()[0]
            other_load = max(0.0, load1 - len(workers) * CORES_PER_WORKER)
            cpu_budget_cores = max(0.0, usable_cores - other_load)
            cpu_target = int(cpu_budget_cores // CORES_PER_WORKER)

            mem_available_gb = _available_memory_gb()
            mem_budget_gb = max(0.0, mem_available_gb - reserved_memory_gb)
            # mem_available_gb already reflects current workers' own
            # usage (it's a live system-wide reading), so this is an
            # ADDITIVE allowance on top of however many are running now.
            additional_by_memory = int(mem_budget_gb // MEMORY_PER_WORKER_GB)
            mem_target = len(workers) + additional_by_memory

            target = min(cpu_target, mem_target, max_workers)
            target = max(min_workers, target)
            self.stdout.write(
                f"Headroom check: load1={load1:.2f} other_load~={other_load:.2f} "
                f"cpu_target={cpu_target} mem_available={mem_available_gb:.1f}GB mem_target={mem_target} "
                f"-> target_workers={target}"
            )
            self.stdout.flush()
            return target

        while True:
            reap_finished()

            if time.time() - last_headroom_check >= check_interval or last_headroom_check == 0.0:
                target = compute_target_worker_count()
                last_headroom_check = time.time()

                # Scaling up is immediate -- act on this single reading.
                while len(workers) < target:
                    pending = [i for i, s in shard_status.items() if s == 'pending']
                    if not pending:
                        break
                    launch(pending[0])

                # Scaling down needs CONSECUTIVE pressure (see
                # --scale-down-confirmations' own help text) -- a single
                # low reading (e.g. right after this supervisor's own
                # restart, before the kernel has caught up reporting
                # memory the just-killed processes already released;
                # confirmed as a real, repeatable pattern 2026-10-25)
                # must not kill a worker that's mid-shard and about to
                # be productive again.
                if len(workers) > target:
                    consecutive_low_target += 1
                    self.stdout.write(
                        f"Target ({target}) below current workers ({len(workers)}) -- "
                        f"{consecutive_low_target}/{scale_down_confirmations} consecutive checks."
                    )
                    self.stdout.flush()
                else:
                    consecutive_low_target = 0

                if consecutive_low_target >= scale_down_confirmations:
                    consecutive_low_target = 0
                    while len(workers) > target:
                        victim = next(iter(workers))
                        proc, cores = workers[victim]
                        self.stdout.write(f"Scaling down: stopping shard {victim}/{total_shards} (pid {proc.pid}).")
                        self.stdout.flush()
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                        del workers[victim]
                        shard_status[victim] = 'pending'

                log_progress()

            if all(s == 'done' for s in shard_status.values()):
                self.stdout.write("All shards done -- image backfill complete.")
                break

            if not workers and not any(s == 'pending' for s in shard_status.values()):
                # Shouldn't happen (every shard is either running,
                # pending, or done), but exit rather than spin forever.
                break

            time.sleep(poll_interval)
