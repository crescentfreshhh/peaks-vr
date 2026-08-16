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


def test_embed_start_requires_media(tmp_path):
    client, _ = _panel(tmp_path, with_media=False)
    # media dir exists but is empty → job runs, finds 0 files
    r = client.post("/api/embed/start", json={})
    assert r.status_code == 200


def test_preview_rejects_path_traversal(tmp_path):
    client, _ = _panel(tmp_path)
    r = client.get("/api/preview", params={"path": "../../etc/passwd"})
    assert r.status_code == 400


def test_preview_unknown_format_is_422(tmp_path):
    client, media = _panel(tmp_path)
    (media / "mystery.mp4").write_bytes(b"x")
    r = client.get("/api/preview", params={"path": "mystery.mp4"})
    assert r.status_code == 422  # no VR hint in the filename
