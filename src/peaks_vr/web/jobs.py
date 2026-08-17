"""A minimal single-job background runner for the control panel.

Embedding a library takes a long time, so the browser starts it and polls
progress rather than blocking on one HTTP request. Only one job runs at a time
(embedding is GPU-bound — no point overlapping), which keeps this tiny: a thread,
a bit of shared state behind a lock, a rolling log, and a cooperative stop flag.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class Job:
    """Live state of the one running (or last-finished) background task."""

    def __init__(self, name: str):
        self.name = name
        self.status = "running"        # running | done | stopped | error
        self.total = 0
        self.done = 0
        self.current = ""              # the item being processed now
        self.current_started: float | None = None  # monotonic when it started
        self.error: str | None = None
        self.stats: dict = {}
        self.started = time.monotonic()
        self.finished: float | None = None
        self._log: deque[str] = deque(maxlen=200)

    def log(self, line: str) -> None:
        self._log.append(line.rstrip())

    def set_current(self, name: str) -> None:
        """Mark the item now being processed and (re)start its elapsed timer."""
        self.current = name
        self.current_started = time.monotonic()

    def snapshot(self) -> dict:
        elapsed = (self.finished or time.monotonic()) - self.started
        eta = None
        if self.status == "running" and self.done and self.total:
            per = elapsed / self.done
            eta = round(per * (self.total - self.done), 1)
        current_elapsed = None
        if self.status == "running" and self.current_started is not None:
            current_elapsed = round(time.monotonic() - self.current_started, 1)
        return {
            "name": self.name,
            "status": self.status,
            "total": self.total,
            "done": self.done,
            "current": self.current,
            "current_elapsed": current_elapsed,
            "error": self.error,
            "stats": self.stats,
            "elapsed": round(elapsed, 1),
            "eta_seconds": eta,
            "log": list(self._log),
        }


class JobManager:
    """Runs at most one :class:`Job` at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_job(self) -> "Job | None":
        """The running (or last-finished) job, for out-of-band loggers like the
        RAM watchdog to append to."""
        return self._job

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def start(self, name: str, target: Callable[[Job, "JobManager"], None]) -> Job:
        """Start ``target(job, manager)`` on a daemon thread. Raises if a job is
        already running."""
        with self._lock:
            if self.running:
                raise RuntimeError("a job is already running")
            self._stop.clear()
            job = Job(name)
            self._job = job

            def _run() -> None:
                try:
                    target(job, self)
                    if job.status == "running":
                        job.status = "stopped" if self._stop.is_set() else "done"
                except Exception as exc:  # surface failures to the UI
                    job.status = "error"
                    job.error = str(exc)
                    job.log(f"! {exc}")
                finally:
                    job.finished = time.monotonic()

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            return job

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict | None:
        job = self._job
        return job.snapshot() if job is not None else None
