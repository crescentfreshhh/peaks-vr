"""A self-regulating RAM watchdog for the embedding run.

Embedding 8K VR is memory-hungry in ways that accumulate in a long-lived
container: PyAV/ffmpeg decode buffers, a whole scene's frames pushed to the GPU
at once, the torch CUDA caching allocator, and glibc malloc-arena fragmentation
across thousands of scenes. Left unchecked the container creeps up until the
kernel OOM-killer reaps the process mid-run — which crashes the app and throws
away an in-flight (resumable) embed.

This watchdog keeps peaks-vr under a hard ceiling (**24 GB by default**, set with
``PEAKS_VR_MAX_RAM_GB``). It is *self-regulating*: a background thread samples the
whole process group's resident memory and, as it rises,

  * **soft breach** (≈80% of the cap): reclaim — ``gc.collect()``, torch CUDA
    ``empty_cache()`` if torch is loaded, and ``malloc_trim`` to hand freed
    arenas back to the OS — so usage recedes without stopping work;
  * **hard breach** (≈90% of the cap): stop the run *cooperatively* via the
    registered ``on_trip`` callback (the job's stop flag), before the OOM-killer
    does it violently. Because embedding is resumable, a clean stop loses
    nothing — the user re-pulls headroom (raise the cap / raise the interval /
    pick a lighter model) and continues.

The embed loop also calls :meth:`MemoryWatchdog.gate` at each scene boundary, so
it never *starts* a new scene while already at the ceiling (synchronous
backpressure on top of the continuous background reclaim).

The memory signal is the process group's anonymous/resident set, read from the
cgroup the container lives in (v2 ``anon`` / v1 ``total_rss`` — both hierarchical,
so child ffmpeg/worker processes are included) and falling back to summing
``/proc/<pid>`` RSS across our process subtree. No third-party dependency.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from typing import Callable

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
GB = 1024 ** 3


# --- reading resident memory ------------------------------------------------

def _cgroup_bytes() -> int | None:
    """Resident (anonymous) memory of this container's cgroup, or None. Both the
    v2 ``anon`` and v1 ``total_rss`` fields are hierarchical — they already sum
    every process in the cgroup, so spawned ffmpeg/de-warp workers are counted.
    Page cache is excluded (it is reclaimable and not what triggers an OOM)."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/memory.stat") as fh:
            for line in fh:
                if line.startswith("anon "):
                    return int(line.split()[1])
    except OSError:
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/memory/memory.stat") as fh:
            fields = {}
            for line in fh:
                k, _, v = line.partition(" ")
                fields[k] = v.strip()
        for key in ("total_rss", "rss"):
            if key in fields:
                return int(fields[key])
    except OSError:
        pass
    return None


def _proc_tree_rss() -> int | None:
    """Fallback: sum RSS across this process and all its descendants via /proc."""
    try:
        children: dict[int, list[int]] = {}
        rss: dict[int, int] = {}
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                with open(f"/proc/{pid}/stat") as fh:
                    data = fh.read()
                after = data[data.rindex(")") + 2:].split()
                ppid = int(after[1])  # field 4 (ppid), counting from state
                with open(f"/proc/{pid}/statm") as fh:
                    resident = int(fh.read().split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(ppid, []).append(pid)
            rss[pid] = resident * _PAGE
        root = os.getpid()
        total, stack = 0, [root]
        seen = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            total += rss.get(pid, 0)
            stack.extend(children.get(pid, ()))
        return total or None
    except OSError:
        return None


def read_rss_bytes() -> int:
    """Best-effort resident memory of the whole peaks-vr process group, in bytes.
    cgroup first (captures children in one read), then a /proc-tree sum."""
    return _cgroup_bytes() or _proc_tree_rss() or 0


def read_container_bytes() -> int | None:
    """Total container memory the way ``docker stats`` reports it, or None.

    ``docker stats`` = current usage minus *inactive* (reclaimable) file cache.
    This is larger than :func:`read_rss_bytes` (the anonymous working set) because
    it also counts file-backed page cache and the multi-GB CUDA/torch library
    mappings — reclaimable memory that inflates the number but does not cause an
    OOM. Shown next to the working set so the two reconcile, but the watchdog caps
    on the working set (the real OOM signal)."""
    def _stat(path: str) -> dict[str, int]:
        out: dict[str, int] = {}
        with open(path) as fh:
            for line in fh:
                k, _, v = line.partition(" ")
                try:
                    out[k] = int(v)
                except ValueError:
                    pass
        return out

    try:  # cgroup v2: memory.current − inactive_file
        with open("/sys/fs/cgroup/memory.current") as fh:
            current = int(fh.read().strip())
        inactive = _stat("/sys/fs/cgroup/memory.stat").get("inactive_file", 0)
        return max(0, current - inactive)
    except OSError:
        pass
    try:  # cgroup v1: usage_in_bytes − total_inactive_file
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as fh:
            usage = int(fh.read().strip())
        inactive = _stat("/sys/fs/cgroup/memory/memory.stat").get(
            "total_inactive_file", 0)
        return max(0, usage - inactive)
    except OSError:
        pass
    return None


def _malloc_trim() -> None:
    """Return freed glibc malloc arenas to the OS (no-op off glibc)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _torch_empty_cache() -> None:
    """Drop the torch CUDA caching allocator's reserve — only if torch is already
    imported (never import it here, to keep the watchdog cheap/offline)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fmt(b: int) -> str:
    return f"{b / GB:.1f} GB"


def limit_from_env(default_gb: float = 24.0) -> float:
    """Cap in bytes from ``PEAKS_VR_MAX_RAM_GB`` (default 24). 0/blank disables
    the watchdog (returns 0)."""
    raw = os.environ.get("PEAKS_VR_MAX_RAM_GB")
    try:
        gb = float(raw) if raw not in (None, "") else default_gb
    except ValueError:
        gb = default_gb
    return max(0.0, gb) * GB


# --- the watchdog -----------------------------------------------------------

class MemoryWatchdog:
    """Samples resident memory on a daemon thread and keeps it under ``limit``.

    ``on_trip`` is called (once per breach episode) when memory crosses the hard
    threshold — wire it to the job's stop so the run halts cleanly. ``read_bytes``
    is injectable for tests. A ``limit`` of 0 disables everything (all methods
    become no-ops), so the feature can be turned off with the env var."""

    def __init__(
        self,
        limit_bytes: float,
        *,
        on_trip: Callable[[], None] | None = None,
        soft: float = 0.80,
        hard: float = 0.90,
        resume: float = 0.70,
        interval: float = 2.0,
        log: Callable[[str], None] | None = None,
        read_bytes: Callable[[], int] = read_rss_bytes,
    ):
        self.limit = float(limit_bytes)
        self.on_trip = on_trip
        self.soft = self.limit * soft
        self.hard = self.limit * hard
        self.resume = self.limit * resume
        self.interval = interval
        self._log = log or (lambda _m: None)
        self._read = read_bytes
        self._peak = 0
        self._tripped = False
        self._last_free = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def current(self) -> int:
        return self._read() if self.enabled else 0

    # --- reclaim ---
    def _free(self) -> None:
        gc.collect()
        _torch_empty_cache()
        _malloc_trim()
        self._last_free = time.monotonic()

    def _free_throttled(self, every: float = 8.0) -> None:
        if time.monotonic() - self._last_free >= every:
            self._free()

    # --- lifecycle ---
    def start(self) -> "MemoryWatchdog":
        if self.enabled and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="mem-watchdog")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._tick()

    def _tick(self) -> None:
        """One sampling step: reclaim on a soft breach, trip (once) on a hard
        breach, rearm once memory recedes. Separated from :meth:`_run` so it is
        directly testable without racing the thread."""
        cur = self._read()
        self._peak = max(self._peak, cur)
        if cur >= self.hard:
            if not self._tripped:
                self._tripped = True
                self._free()
                self._log(f"⚠ RAM {_fmt(cur)} ≥ cap {_fmt(self.limit)} "
                          f"— stopping the run to avoid an OOM kill "
                          f"(resume after freeing headroom)")
                if self.on_trip:
                    try:
                        self.on_trip()
                    except Exception:
                        pass
        elif cur >= self.soft:
            self._free_throttled()
        if cur < self.resume:
            self._tripped = False  # rearm once we're comfortably back down

    # --- synchronous backpressure at scene boundaries ---
    def gate(self, log: Callable[[str], None] | None = None,
             should_stop: Callable[[], bool] | None = None,
             max_wait: float = 30.0) -> bool:
        """Call before starting a scene. Returns True to proceed, False if memory
        is at the cap and won't recede (caller should stop the run — it's
        resumable). Frees on a soft breach; on a hard breach frees and waits
        (bounded, cooperatively) for usage to fall before giving up."""
        if not self.enabled:
            return True
        log = log or self._log
        cur = self._read()
        if cur >= self.hard:
            self._free()
            cur = self._read()
        if cur >= self.hard:
            log(f"⏸ RAM {_fmt(cur)} at cap {_fmt(self.limit)} — pausing before "
                f"next scene to reclaim")
            waited = 0.0
            while (cur >= self.resume and waited < max_wait
                   and not (should_stop and should_stop())):
                time.sleep(1.0)
                waited += 1.0
                cur = self._read()
            if cur >= self.hard:
                log(f"✗ RAM still {_fmt(cur)} after {waited:.0f}s — stopping the "
                    f"run (raise PEAKS_VR_MAX_RAM_GB or the Interval, or use a "
                    f"lighter model, then resume)")
                return False
        elif cur >= self.soft:
            self._free()
        return True

    def snapshot(self) -> dict:
        cur = self.current()  # working set — the capped, OOM-relevant number
        self._peak = max(self._peak, cur)
        container = read_container_bytes() if self.enabled else None
        return {
            "enabled": self.enabled,
            "current_gb": round(cur / GB, 2),
            "container_gb": round(container / GB, 2) if container is not None else None,
            "peak_gb": round(self._peak / GB, 2),
            "limit_gb": round(self.limit / GB, 2),
            "soft_gb": round(self.soft / GB, 2),
            "hard_gb": round(self.hard / GB, 2),
            "pct": round(100 * cur / self.limit, 1) if self.enabled else 0.0,
            "tripped": self._tripped,
        }
