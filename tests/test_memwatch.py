"""Offline tests for the self-regulating RAM watchdog. The memory reader is
injectable, so pressure is simulated deterministically (no real allocation)."""

import peaks_vr.memwatch as mw
from peaks_vr.memwatch import GB, MemoryWatchdog, limit_from_env, read_rss_bytes


def test_read_rss_bytes_positive():
    # cgroup or /proc — this process has *some* resident memory
    assert read_rss_bytes() > 0


def test_container_total_at_least_working_set():
    from peaks_vr.memwatch import read_container_bytes
    ct = read_container_bytes()
    # docker-stats total (if cgroup exposes it) is >= the anonymous working set
    assert ct is None or ct >= read_rss_bytes()


def test_snapshot_carries_container_total():
    w = MemoryWatchdog(24 * GB)
    snap = w.snapshot()
    assert "container_gb" in snap
    # working set is the capped number; container is either a number or null
    assert snap["current_gb"] > 0
    assert snap["container_gb"] is None or snap["container_gb"] >= snap["current_gb"]


def test_limit_from_env(monkeypatch):
    monkeypatch.delenv("PEAKS_VR_MAX_RAM_GB", raising=False)
    assert limit_from_env() == 24 * GB
    monkeypatch.setenv("PEAKS_VR_MAX_RAM_GB", "12")
    assert limit_from_env() == 12 * GB
    monkeypatch.setenv("PEAKS_VR_MAX_RAM_GB", "0")   # disabled
    assert limit_from_env() == 0
    monkeypatch.setenv("PEAKS_VR_MAX_RAM_GB", "junk")  # bad → default
    assert limit_from_env() == 24 * GB


def test_disabled_watchdog_is_noop():
    w = MemoryWatchdog(0)
    assert not w.enabled
    assert w.gate() is True            # never blocks
    assert w.snapshot()["enabled"] is False
    w.start(); w.stop()               # lifecycle is safe when disabled


def test_gate_frees_on_soft_and_stops_on_hard(monkeypatch):
    # avoid touching torch/libc in the reclaim path during the test
    monkeypatch.setattr(mw, "_torch_empty_cache", lambda: None)
    monkeypatch.setattr(mw, "_malloc_trim", lambda: None)
    freed = []
    usage = {"b": 0}
    logs = []
    w = MemoryWatchdog(10 * GB, log=logs.append, read_bytes=lambda: usage["b"])
    monkeypatch.setattr(w, "_free", lambda: freed.append(1))

    usage["b"] = 5 * GB           # below soft (8 GB) → proceed, no reclaim
    assert w.gate() is True
    assert not freed

    usage["b"] = 8.5 * GB         # soft breach (>=8, <9) → reclaim, proceed
    assert w.gate() is True
    assert freed

    usage["b"] = 9.5 * GB         # hard breach (>=9) that won't recede → stop
    assert w.gate(max_wait=0) is False
    assert any("stopping" in m for m in logs)


def test_trip_calls_on_trip_once_then_rearms(monkeypatch):
    monkeypatch.setattr(mw, "_torch_empty_cache", lambda: None)
    monkeypatch.setattr(mw, "_malloc_trim", lambda: None)
    trips = []
    usage = {"b": 9.9 * GB}       # over hard (9 GB)
    w = MemoryWatchdog(10 * GB, on_trip=lambda: trips.append(1),
                       read_bytes=lambda: usage["b"])

    for _ in range(3):            # drive the real loop body, no thread race
        w._tick()
    assert trips == [1]           # tripped once, not re-fired while still high
    assert w.snapshot()["tripped"] is True

    usage["b"] = 6 * GB           # recedes below resume (7 GB) → rearm
    w._tick()
    assert w.snapshot()["tripped"] is False
    usage["b"] = 9.9 * GB         # over hard again → fires a second time
    w._tick()
    assert trips == [1, 1]
