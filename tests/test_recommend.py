"""Offline tests for the recommender (#3): ❤️ marks + cached vectors → a ranked
Playlist. Uses synthetic embedding vectors written straight into the cache, so
no ffmpeg / torch / real model is involved."""

import numpy as np

from peaks_vr.cache import EmbeddingCache
from peaks_vr.config import ScoringConfig
from peaks_vr.labels import LabelStore
from peaks_vr.recommend import Moment, Playlist, recommend_from_labels

MODEL = "fake"
DIM = 8


def _unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _seed_cache(tmp_path):
    """Two scenes: A's frames all point one way, B's an orthogonal way."""
    cache = EmbeddingCache(tmp_path / "cache")
    times = np.arange(0, 20, 2, dtype=np.float32)  # 10 frames, 0..18s
    a_vecs = np.tile(_unit(0), (len(times), 1))
    b_vecs = np.tile(_unit(1), (len(times), 1))
    cache.save("keyA", MODEL, times, a_vecs,
               meta={"path": "D:/vr/A_180_sbs.mp4", "interval": 2.0})
    cache.save("keyB", MODEL, times, b_vecs,
               meta={"path": "D:/vr/B_180_sbs.mp4", "interval": 2.0})
    return cache


def _scoring():
    return ScoringConfig(high=0.5, low=0.4, min_duration=1.0, merge_gap=2.0,
                         max_duration=30.0, pad=0.0)


def test_recommends_moments_similar_to_likes(tmp_path):
    cache = _seed_cache(tmp_path)
    store = LabelStore(tmp_path / "labels.json")
    store.add("keyA", 4.0, 1, "apex", scene_id="D:/vr/A_180_sbs.mp4")
    store.add("keyA", 8.0, 1, "apex", scene_id="D:/vr/A_180_sbs.mp4")

    pl = recommend_from_labels(cache, store, MODEL, "apex", _scoring(), limit=10)

    assert isinstance(pl, Playlist) and len(pl) >= 1
    top = pl.moments[0]
    assert top.key == "keyA"                 # the scene like our likes
    assert top.score > 0.9                   # ~1.0 similarity
    assert all(m.key != "keyB" for m in pl.moments)  # orthogonal scene excluded


def test_exclude_seed_scenes_leaves_only_new(tmp_path):
    cache = _seed_cache(tmp_path)
    store = LabelStore(tmp_path / "labels.json")
    store.add("keyA", 4.0, 1, "apex", scene_id="D:/vr/A_180_sbs.mp4")
    # Only scene A matches the taste, and we exclude the seed scene → nothing new
    pl = recommend_from_labels(cache, store, MODEL, "apex", _scoring(),
                               exclude_seed_scenes=True)
    assert len(pl) == 0


def test_playlist_roundtrips_to_disk(tmp_path):
    pl = Playlist(profile="apex", moments=[
        Moment(path="D:/vr/A.mp4", key="keyA", start=4.0, end=12.0, score=0.97),
    ])
    p = pl.save(tmp_path / "playlist.json")
    back = Playlist.load(p)
    assert back.profile == "apex" and len(back) == 1
    m = back.moments[0]
    assert m.key == "keyA" and m.start == 4.0 and m.end == 12.0
    assert abs(m.duration - 8.0) < 1e-9
