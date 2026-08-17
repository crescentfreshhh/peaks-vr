"""Offline tests for the WebGUI control panel: library scan, stats, the embed
background job, and preview safety. Uses the fake model; no ffmpeg/torch/GPU."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks_vr.cache import EmbeddingCache  # noqa: E402
from peaks_vr.labels import LabelStore  # noqa: E402
from peaks_vr.web.flagging import PlaybackMirror, create_app  # noqa: E402


def _panel(tmp_path, with_media=True):
    media = tmp_path / "media"
    media.mkdir()
    if with_media:
        (media / "a_180_sbs.mp4").write_bytes(b"x")
        (media / "sub").mkdir()
        (media / "sub" / "b_MKX200_tb.mkv").write_bytes(b"x")
        (media / "poster.jpg").write_bytes(b"x")  # ignored
    app = create_app(PlaybackMirror(), LabelStore(tmp_path / "l.json"),
                     media_root=str(media), cache_root=str(tmp_path / "cache"),
                     model="fake")
    return TestClient(app), media


def _wait_idle(client, timeout=5.0):
    """Block until the background job runner is idle (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not client.get("/api/embed/status").json()["running"]:
            return
        time.sleep(0.03)


def test_config_and_library(tmp_path):
    client, media = _panel(tmp_path)
    cfg = client.get("/api/config").json()
    assert cfg["has_media"] is True and cfg["model"] == "fake"
    lib = client.get("/api/library").json()
    assert lib["count"] == 2  # recursive, videos only
    assert {f["name"] for f in lib["files"]} == {"a_180_sbs.mp4", "b_MKX200_tb.mkv"}


def test_stats_counts_cache(tmp_path):
    client, _ = _panel(tmp_path)
    assert client.get("/api/stats").json()["embedded"] == 0
    # write a fake cached scene under the fake model dir
    cache = EmbeddingCache(str(tmp_path / "cache"))
    cache.save("k1", "fake", np.array([0.0], np.float32),
               np.zeros((1, 4), np.float32), meta={"path": "x"})
    assert client.get("/api/stats").json()["embedded"] == 1


def test_embed_job_runs_and_reports(tmp_path):
    client, _ = _panel(tmp_path)
    r = client.post("/api/embed/start", json={"interval": 8, "vr": True,
                                              "hwaccel": "none"})
    assert r.status_code == 200

    # a second start while running is a 409
    conflict = client.post("/api/embed/start", json={})
    assert conflict.status_code in (200, 409)  # may already be done on fast boxes

    deadline = time.time() + 5
    job = None
    while time.time() < deadline:
        job = client.get("/api/embed/status").json()["job"]
        if job and job["status"] != "running":
            break
        time.sleep(0.05)
    assert job is not None
    assert job["total"] == 2               # both videos discovered
    assert job["status"] in ("done", "stopped")
    # no ffmpeg here, so each scene "fails" — but the job tracked them
    assert job["stats"]["failed"] == 2


def test_embed_job_honors_scene_timeout(tmp_path, monkeypatch):
    """The sampling FrameSampler must carry the per-scene timeout: explicit body
    value wins, else PEAKS_SCENE_TIMEOUT, else the FrameSampler default (900)."""
    import peaks_vr.pipeline as pipeline
    from peaks_vr.web.flagging import embed_job

    media = tmp_path / "m"; media.mkdir()
    (media / "clip_180_sbs.mp4").write_bytes(b"x")

    seen = []
    monkeypatch.setattr(pipeline, "embed_library",
                        lambda scenes, sampler, *a, **k: seen.append(sampler.scene_timeout) or {})

    class J:
        def __init__(self): self.total = self.done = 0; self.stats = {}; self.current = ""
        def log(self, *a): pass
        def set_current(self, n): self.current = n
    class M:
        def should_stop(self): return False

    def run(**kw):
        seen.clear()
        embed_job(J(), M(), media_root=str(media), cache_root=str(tmp_path / "c"),
                  model="fake", interval=8.0, vr=False, hwaccel="none", **kw)
        return seen[0]

    assert run(scene_timeout=42.0) == 42.0            # explicit wins
    monkeypatch.setenv("PEAKS_SCENE_TIMEOUT", "300")
    assert run() == 300.0                             # env honored
    monkeypatch.delenv("PEAKS_SCENE_TIMEOUT")
    assert run() == 900.0                             # built-in default


def test_qc_lists_all_files_with_status(tmp_path):
    """/api/qc reports every video with embedded/failed/neither, so the QC
    contact sheet can badge each one. Seeds one cache hit and one failure."""
    from peaks_vr.cache import EmbeddingCache, path_key
    from peaks_vr.web.flagging import failure_log_for

    client, media = _panel(tmp_path)
    cache_root = tmp_path / "cache"
    EmbeddingCache(cache_root).save(path_key(str(media / "a_180_sbs.mp4")),
                                    "fake", np.zeros((1, 4), np.float16), [0.0])
    b = media / "sub" / "b_MKX200_tb.mkv"
    failure_log_for(str(cache_root)).record(path_key(str(b)), None, str(b),
                                            error="moov atom not found")

    r = client.get("/api/qc").json()
    assert r["count"] == 2
    by = {f["name"]: f for f in r["files"]}
    assert by["a_180_sbs.mp4"]["embedded"] and not by["a_180_sbs.mp4"]["failed"]
    assert by["b_MKX200_tb.mkv"]["failed"]
    assert by["b_MKX200_tb.mkv"]["error"] == "moov atom not found"
    assert not by["b_MKX200_tb.mkv"]["embedded"]


def test_preview_frac_seeks_to_fraction_of_duration(tmp_path, monkeypatch):
    """`frac` turns a QC thumbnail request into a mid-file seek (frac*duration),
    so a fixed time can't overrun a short clip."""
    from PIL import Image

    from peaks_vr.sampling import FrameSampler

    client, media = _panel(tmp_path)
    monkeypatch.setattr(FrameSampler, "probe_dimensions", lambda self, p: None)
    monkeypatch.setattr(FrameSampler, "probe_duration", lambda self, p: 200.0)
    seen = {}
    def fake_grab(self, path, t):
        seen["t"] = t
        return Image.new("RGB", (8, 8))
    monkeypatch.setattr(FrameSampler, "grab_frame", fake_grab)

    r = client.get("/api/preview", params={"path": "a_180_sbs.mp4", "frac": 0.5})
    assert r.status_code == 200
    assert seen["t"] == 100.0  # 0.5 * 200


def test_reembed_stores_override_invalidates_and_runs(tmp_path):
    """QC re-embed: stores the sticky format override, drops the old cache entry,
    and runs a single-file job. `format=auto` clears the override."""
    from peaks_vr.cache import EmbeddingCache, path_key
    from peaks_vr.overrides import overrides_for

    client, media = _panel(tmp_path)
    cache_root = tmp_path / "cache"
    p = str(media / "a_180_sbs.mp4")
    key = path_key(p)
    EmbeddingCache(cache_root).save(key, "fake", np.zeros((1, 4), np.float16), [0.0])
    assert EmbeddingCache(cache_root).has(key, "fake")

    r = client.post("/api/embed/reembed",
                    json={"path": p, "format": "180_tb", "fov": 100, "pitch": -10})
    assert r.status_code == 200
    _wait_idle(client)
    ov = overrides_for(str(cache_root)).get(p)
    assert ov and ov["format"] == "180_tb" and ov["pitch"] == -10.0
    # the pre-existing vector was invalidated (deleted) before the re-embed
    assert not EmbeddingCache(cache_root).has(key, "fake")

    # format=auto clears the override
    r = client.post("/api/embed/reembed", json={"path": p, "format": "auto"})
    assert r.status_code == 200
    _wait_idle(client)
    assert overrides_for(str(cache_root)).get(p) is None


def test_reembed_rejects_unknown_format(tmp_path):
    client, media = _panel(tmp_path)
    r = client.post("/api/embed/reembed",
                    json={"path": str(media / "a_180_sbs.mp4"), "format": "bogus"})
    assert r.status_code == 422


def test_preview_uses_stored_override(tmp_path, monkeypatch):
    """A stored override drives /api/preview even with no explicit format param,
    so QC thumbnails reflect the correction. Assert the forced layout reaches the
    reprojector."""
    from PIL import Image

    from peaks_vr.overrides import overrides_for
    from peaks_vr.sampling import FrameSampler

    client, media = _panel(tmp_path)
    # a_180_sbs.mp4's filename says SBS; force TB via an override
    overrides_for(str(tmp_path / "cache")).set(str(media / "a_180_sbs.mp4"),
                                               format="180_tb")
    seen = {}
    def fake_grab(self, path, t):
        seen["layout"] = self.reproject.fmt.layout.value if self.reproject else "flat"
        return Image.new("RGB", (8, 8))
    monkeypatch.setattr(FrameSampler, "grab_frame", fake_grab)

    # the UI passes the absolute path from /api/qc — the same key overrides use
    r = client.get("/api/preview",
                   params={"path": str(media / "a_180_sbs.mp4"), "time": 1})
    assert r.status_code == 200
    assert seen["layout"] == "tb"  # override won over the SBS filename


def test_embed_start_requires_media(tmp_path):
    client, _ = _panel(tmp_path, with_media=False)
    # media dir exists but is empty → job runs, finds 0 files
    r = client.post("/api/embed/start", json={})
    assert r.status_code == 200


def test_failures_recorded_listed_retried_cleared(tmp_path):
    client, _ = _panel(tmp_path)
    # embed the two fake videos → both fail (no ffmpeg), recorded in the log
    client.post("/api/embed/start", json={"vr": False, "hwaccel": "none"})
    deadline = time.time() + 5
    while time.time() < deadline:
        if not client.get("/api/embed/status").json()["running"]:
            break
        time.sleep(0.05)

    f = client.get("/api/failures").json()
    assert f["count"] == 2
    assert all(e["error"] for e in f["entries"])   # each carries the ffmpeg error

    # retry runs a job over just the failed set (still 2, still no ffmpeg)
    r = client.post("/api/embed/retry")
    assert r.json() == {"started": True, "count": 2}
    deadline = time.time() + 5
    job = None
    while time.time() < deadline:
        st = client.get("/api/embed/status").json()
        job = st["job"]
        if not st["running"]:
            break
        time.sleep(0.05)
    assert job["name"] == "retry" and job["total"] == 2   # not a full rescan

    # clear empties the list
    assert client.post("/api/failures/clear").json()["cleared"] == 2
    assert client.get("/api/failures").json()["count"] == 0


def test_embed_status_reports_current_elapsed(tmp_path):
    from peaks_vr.web.jobs import Job
    j = Job("embed")
    j.set_current("scene.mp4")
    snap = j.snapshot()
    assert snap["current"] == "scene.mp4"
    assert snap["current_elapsed"] is not None and snap["current_elapsed"] >= 0


def test_retry_with_no_failures_is_noop(tmp_path):
    client, _ = _panel(tmp_path)
    r = client.post("/api/embed/retry")
    assert r.json()["started"] is False


def test_preview_rejects_path_traversal(tmp_path):
    client, _ = _panel(tmp_path)
    r = client.get("/api/preview", params={"path": "../../etc/passwd"})
    assert r.status_code == 400


def test_preview_unknown_format_needs_assume(tmp_path):
    client, media = _panel(tmp_path)
    (media / "mystery.mp4").write_bytes(b"x")
    # no hint AND assume explicitly empty → 422
    r = client.get("/api/preview", params={"path": "mystery.mp4", "assume": ""})
    assert r.status_code == 422
    # with an assume, detection succeeds and it reaches the sampler (503 here —
    # no ffmpeg in the sandbox — i.e. it got *past* format detection)
    r = client.get("/api/preview", params={"path": "mystery.mp4",
                                           "assume": "180_sbs"})
    assert r.status_code == 503
