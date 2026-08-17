"""Offline tests for DJ taste-profile curation: bulk suggestions, category-tagged
likes, exclusion of already-liked frames, and per-category summary. No headset,
no ffmpeg, no torch — a seeded fake cache stands in for an embedded library."""

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks_vr import taste  # noqa: E402
from peaks_vr.cache import EmbeddingCache, path_key  # noqa: E402
from peaks_vr.labels import LabelStore  # noqa: E402
from peaks_vr.web.flagging import PlaybackMirror, create_app  # noqa: E402


def _seed_cache(cache_root, scenes=4, frames=10):
    cache = EmbeddingCache(cache_root)
    rng = np.random.default_rng(0)
    for i in range(scenes):
        p = f"/data/scene_{i}_180_sbs.mp4"
        cache.save(path_key(p), "fake",
                   np.arange(frames) * 8.0,
                   rng.standard_normal((frames, 16)).astype(np.float16),
                   meta={"path": p})
    return cache


def _app(tmp_path, base="dj"):
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root)
    store = LabelStore(tmp_path / "labels.json")
    app = create_app(PlaybackMirror(), store, media_root=None,
                     cache_root=str(cache_root), model="fake", profile=base)
    return TestClient(app), store


# --- module-level helpers ---------------------------------------------------

def test_suggest_cold_start_is_random(tmp_path):
    cache = _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    frames = taste.suggest_frames(store, cache, "fake", "dj", count=20, seed=1)
    assert frames and all(set(f) >= {"key", "path", "time", "score"} for f in frames)
    assert all(f["score"] == 0.0 for f in frames)   # no taste yet → no signal


def test_like_tags_category_and_excludes(tmp_path):
    cache = _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    frames = taste.suggest_frames(store, cache, "fake", "dj", count=40, seed=2)
    f = frames[0]
    prof = taste.like_frame(store, f["key"], f["time"], f["path"], "dj", "cowgirl")
    assert prof == "dj:cowgirl"
    store.save()
    # a second suggestion batch must not re-show the liked frame
    again = taste.suggest_frames(store, cache, "fake", "dj", count=200, seed=3)
    assert (f["key"], round(f["time"], 2)) not in {
        (g["key"], round(g["time"], 2)) for g in again}


def test_summary_counts_per_category(tmp_path):
    cache = _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    frames = taste.suggest_frames(store, cache, "fake", "dj", count=40, seed=4)
    taste.like_frame(store, frames[0]["key"], frames[0]["time"], None, "dj", "cowgirl")
    taste.like_frame(store, frames[1]["key"], frames[1]["time"], None, "dj", "cowgirl")
    taste.like_frame(store, frames[2]["key"], frames[2]["time"], None, "dj", "blowjob")
    taste.like_frame(store, frames[3]["key"], frames[3]["time"], None, "dj", None)  # untagged
    s = taste.taste_summary(store, "dj")
    assert s["total"] == 4
    by = {c["name"]: c["count"] for c in s["categories"]}
    assert by["cowgirl"] == 2 and by["blowjob"] == 1 and by["(untagged)"] == 1


def test_liked_vectors_span_all_categories(tmp_path):
    cache = _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    frames = taste.suggest_frames(store, cache, "fake", "dj", count=40, seed=5)
    for i, cat in enumerate(("cowgirl", "blowjob", "cowgirl")):
        taste.like_frame(store, frames[i]["key"], frames[i]["time"], None, "dj", cat)
    vecs = taste.liked_vectors(store, cache, "fake", "dj")
    assert vecs.shape[0] == 3          # union across both categories, not averaged


# --- HTTP surface -----------------------------------------------------------

def test_taste_endpoints_flow(tmp_path):
    client, _ = _app(tmp_path, base="dj")
    s = client.get("/api/taste/suggest?count=12").json()
    assert s["base"] == "dj" and s["count"] > 0
    f = s["frames"][0]
    r = client.post("/api/taste/like",
                    json={"key": f["key"], "time": f["time"], "path": f["path"],
                          "category": "cowgirl"}).json()
    assert r["profile"] == "dj:cowgirl" and r["count"] == 1
    assert client.get("/api/taste/summary").json()["total"] == 1
    # unlike toggles it back off
    d = client.post("/api/taste/unlike",
                    json={"key": f["key"], "time": f["time"],
                          "category": "cowgirl"}).json()
    assert d["dropped"] is True
    assert client.get("/api/taste/summary").json()["total"] == 0
