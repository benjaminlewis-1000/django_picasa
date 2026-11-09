#!/usr/bin/env python3
"""Supervises a long-running command (in practice, the celery worker),
restarting it immediately whenever it exits -- capped so a genuinely
unfixed problem doesn't spin forever. See CLAUDE.md (2026-11-09) and
common/tasks.py's check_schema_freshness for the incident this exists
for: celery's prefork pool executes tasks in child processes forked
from the master, which inherit the master's already-imported Python
modules -- a crashed CHILD process respawning does NOT pick up fresh
code (it's just forked again from the same stale master). Only killing
and relaunching the whole MASTER process (what this supervises) ever
re-imports from disk. Deliberately a plain stdlib-only script, no
Django import at module scope -- matches prune_backups.py's/
weekly_vacuum_swap.py's own "runs with nothing but stdlib" design, so
it can't itself be broken by whatever caused the thing it's supervising
to need restarting.

Run with: python3 supervise_celery_worker.py <max_restarts> <window_seconds> \
              <escape_hatch_argv_json> -- <command...>

escape_hatch_argv_json is a JSON-encoded argv list (e.g.
'["python", "manage.py", "shell", "-c", "..."]'), run once <command>
has exited <max_restarts> times within any trailing <window_seconds>
window -- a separate, direct process, not routed through celery at
all, since the whole point is it must still work when celery itself is
the thing that's broken.
"""
import json
import subprocess
import sys
import time


def supervise(
    command, max_restarts, window_seconds, escape_hatch_command,
    sleep_fn=time.sleep, now_fn=time.time, run_fn=subprocess.call,
):
    """Runs `command` (an argv list) in a loop, restarting it every time
    it exits for any reason (success or failure -- a long-running
    worker is never expected to exit on its own at all). If it exits
    `max_restarts` times within any trailing `window_seconds` window,
    runs `escape_hatch_command` (another argv list) once and returns 1
    instead of restarting again. sleep_fn/now_fn/run_fn are injected
    purely for testability -- real callers never need to pass them.
    """
    restart_times = []
    while True:
        run_fn(command)
        now = now_fn()
        restart_times.append(now)
        restart_times = [t for t in restart_times if now - t < window_seconds]
        print(
            f"supervise_celery_worker: {command!r} exited -- "
            f"{len(restart_times)} restart(s) in the last {window_seconds}s",
            flush=True,
        )
        if len(restart_times) >= max_restarts:
            print(
                f"supervise_celery_worker: {command!r} restarted "
                f"{len(restart_times)} times in {window_seconds}s -- giving up, "
                f"running the escape-hatch command",
                flush=True,
            )
            run_fn(escape_hatch_command)
            return 1
        sleep_fn(5)


if __name__ == '__main__':
    max_restarts = int(sys.argv[1])
    window_seconds = int(sys.argv[2])
    escape_hatch_command = json.loads(sys.argv[3])
    sep_index = sys.argv.index('--')
    command = sys.argv[sep_index + 1:]
    sys.exit(supervise(command, max_restarts, window_seconds, escape_hatch_command))
