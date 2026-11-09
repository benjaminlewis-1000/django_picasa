#!/usr/bin/env python3
"""Tests for supervise_celery_worker.py. Plain unittest, no Django --
matches the script's own "no Django dependency" design. sleep_fn/now_fn/
run_fn are all injected fakes, so these run instantly with no real
subprocess spawning or real sleeping.

Run with: python3 -m unittest dockerize.test_supervise_celery_worker -v
      or: python3 test_supervise_celery_worker.py -v   (from inside dockerize/)
"""
import unittest

import supervise_celery_worker as scw


class FakeClock:
    """A controllable fake now_fn()/sleep_fn() pair -- sleep_fn actually
    advances the fake clock (rather than a real time.sleep), so a test
    exercising the restart-window decay logic runs instantly."""

    def __init__(self, start=0.0):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class SuperviseTests(unittest.TestCase):
    def test_restarts_on_every_exit_until_the_cap(self):
        clock = FakeClock()
        run_calls = []

        def run_fn(command):
            run_calls.append(command)

        result = scw.supervise(
            ['echo', 'worker'], max_restarts=3, window_seconds=600,
            escape_hatch_command=['echo', 'escape'],
            sleep_fn=clock.sleep, now_fn=clock.now, run_fn=run_fn,
        )

        self.assertEqual(result, 1)
        # The supervised command runs 3 times (reaching the cap), plus
        # the escape hatch runs once as the 4th call.
        self.assertEqual(run_calls, [
            ['echo', 'worker'], ['echo', 'worker'], ['echo', 'worker'],
            ['echo', 'escape'],
        ])

    def test_escape_hatch_not_run_before_the_cap_is_reached(self):
        clock = FakeClock()
        run_calls = []
        call_count = 0

        def run_fn(command):
            nonlocal call_count
            run_calls.append(command)
            call_count += 1
            if call_count >= 2:
                # Simulate a real, resolved restart: stop looping by
                # raising, so the test can assert on state up to here
                # without actually reaching max_restarts.
                raise SystemExit(0)

        with self.assertRaises(SystemExit):
            scw.supervise(
                ['echo', 'worker'], max_restarts=5, window_seconds=600,
                escape_hatch_command=['echo', 'escape'],
                sleep_fn=clock.sleep, now_fn=clock.now, run_fn=run_fn,
            )
        self.assertNotIn(['echo', 'escape'], run_calls)

    def test_restarts_outside_the_window_do_not_count_toward_the_cap(self):
        # The FIRST restart must age out of the window before the THIRD
        # is recorded -- the command itself runs a very long time
        # (simulating a genuinely healthy, long-lived process) on its
        # second invocation, between the two short-lived failures. This
        # is what lets a worker that fails once, runs fine for a long
        # time, and then fails again NOT be treated as a death spiral.
        # Each run_fn call advances the fake clock by how long THAT run
        # itself took, before the exit is recorded.
        clock = FakeClock()
        run_calls = []
        run_durations = [0, 10_000, 0, 0]  # instant fail, long healthy run, instant fail, escape hatch
        call_index = [0]

        def run_fn(command):
            run_calls.append(command)
            clock.t += run_durations[call_index[0]]
            call_index[0] += 1

        result = scw.supervise(
            ['echo', 'worker'], max_restarts=2, window_seconds=600,
            escape_hatch_command=['echo', 'escape'],
            sleep_fn=clock.sleep, now_fn=clock.now, run_fn=run_fn,
        )
        # Without window decay this would hit the cap on the 2nd
        # restart; with it, the first restart ages out (>600s before
        # the 3rd), so it takes a 3rd restart -- immediately following
        # the 2nd, still within the window of each other -- to trip the
        # cap.
        self.assertEqual(result, 1)
        self.assertEqual(run_calls, [
            ['echo', 'worker'], ['echo', 'worker'], ['echo', 'worker'],
            ['echo', 'escape'],
        ])

    def test_sleeps_between_restarts(self):
        clock = FakeClock()
        sleep_calls = []

        def sleep_fn(seconds):
            sleep_calls.append(seconds)
            clock.sleep(seconds)

        scw.supervise(
            ['echo', 'worker'], max_restarts=2, window_seconds=600,
            escape_hatch_command=['echo', 'escape'],
            sleep_fn=sleep_fn, now_fn=clock.now, run_fn=lambda c: None,
        )
        # One sleep between the 1st and 2nd restart; none after the 2nd
        # (the cap), since it gives up instead of looping again.
        self.assertEqual(sleep_calls, [5])


class CliParsingTests(unittest.TestCase):
    def test_escape_hatch_json_and_dash_dash_split_parse_correctly(self):
        import json
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, scw.__file__, '1', '600', json.dumps(['echo', 'escaped']),
             '--', 'python3', '-c', 'import sys; sys.exit(0)'],
            capture_output=True, text=True, timeout=30,
        )
        # The command and escape hatch both ran (exit 0 from the
        # supervised command counts as a restart like any other exit);
        # with max_restarts=1 the escape hatch fires on the very first
        # exit. Confirmed via the script's own stdout logging, not by
        # mocking -- this is the one real subprocess-spawning test here,
        # deliberately small and fast (two trivial real processes).
        self.assertIn("restarted 1 times", proc.stdout)
        self.assertIn("giving up", proc.stdout)
        self.assertEqual(proc.returncode, 1)


if __name__ == '__main__':
    unittest.main()
