"""Orchestration glue, exercised offline with fakes (no ffmpeg/torch/Stash)."""

import numpy as np

from peaks_vr.cache import EmbeddingCache
from peaks_vr.config import ScoringConfig
from peaks_vr.embedding import FakeEmbedder
from peaks_vr.models import Scene
from peaks_vr.pipeline import embed_library, score_library, scene_key, sync_cache
from peaks_vr.scoring import make_similarity_scorer


class _StubImage:
    def __init__(self, payload: bytes):
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload

    def copy(self):
        return self


class _StubSampler:
    """Yields a fixed set of (timestamp, image) pairs, ignoring the path."""

    interval = 2.0

    def __init__(self, frames):
        self._frames = frames

    def iter_frames(self, path):
        for ts, img in self._frames:
            yield ts, img


class _CapturingClient:
    def __init__(self):
        self.markers = []

    def find_or_create_tag(self, name):
        return type("Tag", (), {"id": "tag1", "name": name})()

    def create_scene_marker(self, *, scene_id, seconds, primary_tag_id, title, end_seconds):
        self.markers.append((scene_id, seconds, end_seconds, title))


def _scene(id_, path):
    return Scene.from_dict(
        {
            "id": id_,
            "title": "",
            "files": [{"path": path, "duration": 50.0, "fingerprints": [
                {"type": "oshash", "value": f"fp{id_}"}]}],
            "scene_markers": [],
        }
    )


def test_scene_key_prefers_fingerprint():
    s = _scene("1", "/m/1.mp4")
    assert scene_key(s) == "fp1"


def test_embed_library_populates_cache_and_is_resumable(tmp_path):
    emb = FakeEmbedder(dim=16)
    frames = [(i * 2.0, _StubImage(f"f{i}".encode())) for i in range(5)]
    sampler = _StubSampler(frames)
    cache = EmbeddingCache(tmp_path)
    scenes = [_scene("1", "/m/1.mp4"), _scene("2", "/m/2.mp4")]

    stats = embed_library(scenes, sampler, emb, cache, log=lambda *_: None)
    assert stats["embedded"] == 2 and stats["frames"] == 10
    assert cache.has("fp1", "fake") and cache.has("fp2", "fake")

    # second run skips everything already cached
    stats2 = embed_library(scenes, sampler, emb, cache, log=lambda *_: None)
    assert stats2["embedded"] == 0 and stats2["skipped"] == 2


def _populate_loved_scene(cache, emb, key, loved_img):
    """Cache a scene whose middle frames equal the loved reference frame."""
    frames = [_StubImage(f"noise{i}".encode()) for i in range(20)]
    for i in range(8, 12):
        frames[i] = loved_img
    vecs = emb.embed_images(frames)
    times = np.arange(20, dtype=np.float32) * 2.0
    cache.save(key, emb.name, times, vecs)


def test_score_library_dry_run_finds_segment_without_writing(tmp_path):
    emb = FakeEmbedder(dim=24)
    cache = EmbeddingCache(tmp_path)
    loved = _StubImage(b"loved")
    _populate_loved_scene(cache, emb, "fp1", loved)
    references = emb.embed_images([loved])

    scoring = ScoringConfig(high=0.9, low=0.7, min_duration=2.0, merge_gap=2.0, smooth_window=1, pad=0.0)
    stats = score_library(
        [_scene("1", "/m/1.mp4")], cache, emb.name,
        make_similarity_scorer(references, "max"), scoring,
        write=False, log=lambda *_: None,
    )
    assert stats["scenes"] == 1
    assert stats["segments"] == 1  # the loved stretch


def test_score_library_write_creates_markers(tmp_path):
    emb = FakeEmbedder(dim=24)
    cache = EmbeddingCache(tmp_path)
    loved = _StubImage(b"loved")
    _populate_loved_scene(cache, emb, "fp1", loved)
    references = emb.embed_images([loved])
    client = _CapturingClient()

    scoring = ScoringConfig(high=0.9, low=0.7, min_duration=2.0, merge_gap=2.0, smooth_window=1, pad=0.0)
    score_library(
        [_scene("1", "/m/1.mp4")], cache, emb.name,
        make_similarity_scorer(references, "max"), scoring,
        client=client, tag_name="apex", write=True, log=lambda *_: None,
    )
    assert len(client.markers) == 1
    scene_id, start, end, title = client.markers[0]
    assert scene_id == "1" and title.startswith("apex ")  # "apex <peak score>"
    assert start == 16.0 and end == 22.0  # frames 8-11 -> times 16..22


class _FlakySampler:
    """interval=2.0; raises for scenes whose path contains 'bad'."""

    interval = 2.0

    def __init__(self, frames):
        self._frames = frames

    def iter_frames(self, path):
        if "bad" in path:
            raise RuntimeError("Invalid NAL unit size")
        for ts, img in self._frames:
            yield ts, img


def test_embed_library_records_and_clears_failures(tmp_path):
    from peaks_vr.failures import FailureLog

    emb = FakeEmbedder(dim=8)
    frames = [(0.0, _StubImage(b"f0")), (2.0, _StubImage(b"f1"))]
    sampler = _FlakySampler(frames)
    cache = EmbeddingCache(tmp_path)
    flog = FailureLog(tmp_path / "failures.json")
    scenes = [_scene("1", "/m/good.mp4"), _scene("2", "/m/bad.mp4")]

    stats = embed_library(
        scenes, sampler, emb, cache, log=lambda *_: None, failure_log=flog
    )
    assert stats["embedded"] == 1 and stats["failed"] == 1
    assert flog.keys() == {"fp2"}  # the 'bad' scene got logged
    entry = flog.entries()[0]
    assert entry["scene_id"] == "2" and "NAL" in entry["error"]

    # a later pass where the same scene now succeeds clears its entry
    good_sampler = _StubSampler(frames)
    embed_library(
        [_scene("2", "/m/bad.mp4")], good_sampler, emb, cache,
        log=lambda *_: None, failure_log=flog,
    )
    assert flog.keys() == set()


def test_embed_library_honors_should_stop(tmp_path):
    emb = FakeEmbedder(dim=8)
    sampler = _StubSampler([(0.0, _StubImage(b"f0"))])
    cache = EmbeddingCache(tmp_path)
    scenes = [_scene("1", "/m/1.mp4"), _scene("2", "/m/2.mp4"), _scene("3", "/m/3.mp4")]
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1  # let the first scene through, then halt

    stats = embed_library(scenes, sampler, emb, cache, log=lambda *_: None, should_stop=stop)
    assert stats["embedded"] == 1  # stopped before the rest


def test_score_library_honors_should_stop(tmp_path):
    emb = FakeEmbedder(dim=8)
    cache = EmbeddingCache(tmp_path)
    for k in ("fp1", "fp2"):
        cache.save(k, emb.name, np.array([0.0], dtype="float32"), np.zeros((1, 8), dtype="float32"))
    refs = emb.embed_images([_StubImage(b"x")])
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1

    stats = score_library(
        [_scene("1", "/m/1.mp4"), _scene("2", "/m/2.mp4")],
        cache, emb.name, make_similarity_scorer(refs, "max"), ScoringConfig(),
        should_stop=stop, log=lambda *_: None,
    )
    assert stats["scenes"] == 1


def _cache_scene(cache, key, scene_id, path, model="fake"):
    t = np.array([0.0], dtype=np.float32)
    v = np.zeros((1, 4), dtype=np.float32)
    cache.save(key, model, t, v, meta={"scene_id": scene_id, "path": path})


def test_sync_cache_refreshes_moved_scene(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # cached under the stable fingerprint key, but at the OLD path
    _cache_scene(cache, "fp1", "1", "/data/Rando/old/a.mp4")
    # Stash now reports the same scene (same fingerprint) at a new path
    scenes = [_scene("1", "/data/Rando/new/a.mp4")]

    stats = sync_cache(scenes, cache, "fake", prune=True, log=lambda *_: None)
    assert stats["moved"] == 1 and stats["pruned"] == 0
    _, _, meta = cache.load("fp1", "fake")
    assert meta["path"] == "/data/Rando/new/a.mp4"  # refreshed in place


def test_sync_cache_prunes_deleted_scene(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _cache_scene(cache, "fp1", "1", "/data/Rando/a.mp4")
    _cache_scene(cache, "fp2", "2", "/data/Rando/gone.mp4")
    scenes = [_scene("1", "/data/Rando/a.mp4")]  # fp2 no longer in Stash

    stats = sync_cache(scenes, cache, "fake", prune=True, log=lambda *_: None)
    assert stats["orphaned"] == 1 and stats["pruned"] == 1
    assert cache.has("fp1", "fake") and not cache.has("fp2", "fake")


def test_sync_cache_dry_run_reports_without_deleting(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _cache_scene(cache, "fp2", "2", "/data/Rando/gone.mp4")

    stats = sync_cache([], cache, "fake", prune=False, log=lambda *_: None)
    assert stats["orphaned"] == 1 and stats["pruned"] == 0
    assert cache.has("fp2", "fake")  # dry run leaves the orphan intact


def test_score_library_skips_uncached_scene(tmp_path):
    emb = FakeEmbedder(dim=8)
    cache = EmbeddingCache(tmp_path)
    references = emb.embed_images([_StubImage(b"x")])
    stats = score_library(
        [_scene("404", "/m/404.mp4")], cache, emb.name,
        make_similarity_scorer(references, "max"),
        ScoringConfig(), write=False, log=lambda *_: None,
    )
    assert stats["skipped"] == 1 and stats["scenes"] == 0
