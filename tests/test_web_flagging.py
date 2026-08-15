"""Offline tests for the flagging UI backend, driven through FastAPI's
TestClient — no headset, no uvicorn. A PlaybackMirror is fed directly to stand
in for the live HereSphere feed (the same decoupling `--demo` uses)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks_vr.cache import path_key  # noqa: E402
from peaks_vr.heresphere import PlaybackState  # noqa: E402
from peaks_vr.labels import LabelStore  # noqa: E402
from peaks_vr.web.flagging import PlaybackMirror, create_app  # noqa: E402

SCENE = "D:/vr/clip_180_sbs.mp4"


def _client(tmp_path):
    mirror = PlaybackMirror()
    store = LabelStore(tmp_path / "labels.json")
    app = create_app(mirror, store, profile="apex")
    return TestClient(app), mirror, store


def test_state_reflects_mirror(tmp_path):
    client, mirror, _ = _client(tmp_path)

    # nothing playing yet
    s = client.get("/api/state").json()
    assert s["connected"] is False and s["path"] is None

    mirror.update(PlaybackState(path=SCENE, current_time=12.5, playing=True,
                                duration=100.0))
    s = client.get("/api/state").json()
    assert s["connected"] is True
    assert s["path"] == SCENE
    assert s["key"] == path_key(SCENE)     # the scene key a mark will attach to
    assert s["current_time"] == 12.5 and s["playing"] is True


def test_mark_writes_positive_label(tmp_path):
    client, mirror, store = _client(tmp_path)
    mirror.update(PlaybackState(path=SCENE, current_time=42.0, playing=True))

    r = client.post("/api/mark", json={"time": 40.5})
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == path_key(SCENE)
    assert body["times"] == [40.5]
    assert body["positives"] == 1

    # persisted to the store as a positive label for this scene key
    labs = LabelStore(tmp_path / "labels.json").for_profile("apex")
    assert len(labs) == 1
    assert labs[0].label == 1 and labs[0].key == path_key(SCENE)
    assert labs[0].time == 40.5 and labs[0].scene_id == SCENE


def test_mark_in_out_window_records_both_ends(tmp_path):
    client, mirror, _ = _client(tmp_path)
    mirror.update(PlaybackState(path=SCENE, current_time=50.0, playing=True))

    r = client.post("/api/mark", json={"time": 50.0, "in_time": 48.0,
                                       "out_time": 53.0})
    assert r.status_code == 200
    # dedup + sorted: in, point, out
    assert r.json()["times"] == [48.0, 50.0, 53.0]

    marks = client.get("/api/marks").json()
    assert {m["time"] for m in marks} == {48.0, 50.0, 53.0}


def test_mark_without_playback_is_conflict(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/api/mark", json={"time": 1.0})
    assert r.status_code == 409


def test_frame_preview_unavailable_without_sampler(tmp_path):
    client, mirror, _ = _client(tmp_path)
    mirror.update(PlaybackState(path=SCENE, current_time=5.0, playing=True))
    assert client.get("/api/frame", params={"time": 5.0}).status_code == 503


def test_index_page_served(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "flag moments" in r.text
