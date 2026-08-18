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

def test_profile_picker_switch_and_persist(tmp_path):
    """The active taste profile is switchable at runtime and persisted to /config,
    so it survives a restart without env vars — and everything profile-scoped
    (taste base) follows it."""
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root)
    store = LabelStore(tmp_path / "labels.json")

    def build():
        return TestClient(create_app(PlaybackMirror(), store, media_root=None,
                                     cache_root=str(cache_root), model="fake",
                                     profile="apex"))

    c = build()
    assert c.get("/api/profiles").json() == {"active": "apex", "profiles": ["apex"]}
    assert c.get("/api/config").json()["profile"] == "apex"
    assert c.post("/api/profile", json={"profile": "dj"}).json()["active"] == "dj"
    assert c.get("/api/taste/summary").json()["base"] == "dj"          # follows switch
    assert set(c.get("/api/profiles").json()["profiles"]) == {"apex", "dj"}
    # a fresh app (restart) with the same /config keeps the chosen profile
    assert build().get("/api/config").json()["profile"] == "dj"


def test_profile_name_sanitized(tmp_path):
    _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    c = TestClient(create_app(PlaybackMirror(), store, media_root=None,
                              cache_root=str(tmp_path / "cache"), model="fake"))
    # colons (the category separator) and spaces are neutralised
    assert c.post("/api/profile", json={"profile": "dj:hack me"}).json()["active"] \
        == "dj_hack_me"


def test_random_forces_exploration_when_warm(tmp_path):
    """A warm profile normally gets active-learning suggestions, but random=True
    forces pure exploration (used by untagged 'Load more')."""
    cache = _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    frames = taste.suggest_frames(store, cache, "fake", "dj", count=40, seed=1)
    for f in frames[:taste.WARM_AT]:          # push over the warm threshold
        taste.like_frame(store, f["key"], f["time"], f["path"], "dj", "cowgirl")
    auto = taste.suggest_frames(store, cache, "fake", "dj", count=20, seed=2)
    rand = taste.suggest_frames(store, cache, "fake", "dj", count=20, seed=2,
                                random=True)
    assert any(f["score"] != 0 for f in auto)          # warm → scored
    assert all(f["score"] == 0 for f in rand)          # random → unscored


def test_similar_frames_ranked_and_excludes_anchor(tmp_path):
    cache = _seed_cache(tmp_path / "cache", scenes=5, frames=12)
    store = LabelStore(tmp_path / "labels.json")
    anchor = taste.suggest_frames(store, cache, "fake", "dj", count=1, seed=3)[0]
    sim = taste.similar_frames(store, cache, "fake", "dj",
                               anchor["key"], anchor["time"], count=15)
    assert sim and all(set(f) >= {"key", "path", "time", "score"} for f in sim)
    scores = [f["score"] for f in sim]
    assert scores == sorted(scores, reverse=True)      # ranked most-similar first
    ids = {(f["key"], round(f["time"], 2)) for f in sim}
    assert (anchor["key"], round(anchor["time"], 2)) not in ids   # not the anchor
    # per-scene variety cap
    from collections import Counter
    assert max(Counter(f["key"] for f in sim).values()) <= 3


def test_similar_endpoint(tmp_path):
    client, _ = _app(tmp_path, base="dj")
    f = client.get("/api/taste/suggest?count=1&random=1").json()["frames"][0]
    r = client.get(f"/api/taste/similar?key={f['key']}&time={f['time']}&count=8").json()
    assert r["base"] == "dj" and r["count"] >= 1


def test_reset_clears_all_categories(tmp_path):
    """Full reset removes every like across the base profile and all its
    categories, and persists."""
    cache = _seed_cache(tmp_path / "cache")
    store = LabelStore(tmp_path / "labels.json")
    frames = taste.suggest_frames(store, cache, "fake", "dj", count=40, seed=6)
    taste.like_frame(store, frames[0]["key"], frames[0]["time"], None, "dj", "cowgirl")
    taste.like_frame(store, frames[1]["key"], frames[1]["time"], None, "dj", "blowjob")
    taste.like_frame(store, frames[2]["key"], frames[2]["time"], None, "dj", None)
    assert taste.taste_summary(store, "dj")["total"] == 3
    removed = taste.reset_taste(store, "dj")
    store.save()
    assert removed == 3
    assert taste.taste_summary(store, "dj")["total"] == 0
    # a fresh store from disk confirms it persisted
    assert len(LabelStore(tmp_path / "labels.json")) == 0


def test_reset_endpoint(tmp_path):
    client, _ = _app(tmp_path, base="dj")
    f = client.get("/api/taste/suggest?count=1&random=1").json()["frames"][0]
    client.post("/api/taste/like", json={"key": f["key"], "time": f["time"],
                                         "path": f["path"], "category": "x"})
    assert client.get("/api/taste/summary").json()["total"] == 1
    r = client.post("/api/taste/reset").json()
    assert r["removed"] == 1 and r["profile"] == "dj"
    assert client.get("/api/taste/summary").json()["total"] == 0


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
