import numpy as np
import pytest

from peaks_vr.cache import EmbeddingCache, path_key


def test_path_key_stable_and_prefixed():
    k1 = path_key("/movies/a.mp4")
    k2 = path_key("/movies/a.mp4")
    assert k1 == k2 and k1.startswith("path-")
    assert path_key("/movies/b.mp4") != k1


def test_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path)
    times = np.array([0.5, 2.5, 4.5], dtype=np.float32)
    vecs = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    assert not cache.has("key1", "fake")
    cache.save("key1", "fake", times, vecs, meta={"interval": 2.0, "path": "/x.mp4"})
    assert cache.has("key1", "fake")

    lt, lv, meta = cache.load("key1", "fake")
    np.testing.assert_array_equal(lt, times)  # times stay exact float32
    np.testing.assert_allclose(lv, vecs, atol=2e-3)  # vecs stored float16
    assert lv.dtype == np.float32
    assert meta["interval"] == 2.0 and meta["path"] == "/x.mp4"


def test_models_are_namespaced(tmp_path):
    cache = EmbeddingCache(tmp_path)
    t = np.array([0.0], dtype=np.float32)
    v = np.zeros((1, 4), dtype=np.float32)
    cache.save("k", "dino", t, v)
    assert cache.has("k", "dino")
    assert not cache.has("k", "clip")  # same key, different model = separate cache


def test_keys_listing(tmp_path):
    cache = EmbeddingCache(tmp_path)
    t = np.array([0.0], dtype=np.float32)
    v = np.zeros((1, 4), dtype=np.float32)
    cache.save("b", "fake", t, v)
    cache.save("a", "fake", t, v)
    assert cache.keys("fake") == ["a", "b"]
    assert cache.keys("missing") == []


def test_mismatched_lengths_rejected(tmp_path):
    cache = EmbeddingCache(tmp_path)
    with pytest.raises(ValueError):
        cache.save("k", "fake", np.zeros(3), np.zeros((2, 4)))


def test_delete_removes_entry(tmp_path):
    cache = EmbeddingCache(tmp_path)
    t = np.array([0.0], dtype=np.float32)
    v = np.zeros((1, 4), dtype=np.float32)
    cache.save("k", "fake", t, v)
    assert cache.delete("k", "fake") is True
    assert not cache.has("k", "fake")
    assert cache.delete("k", "fake") is False  # already gone


def test_models_lists_cache_dirs(tmp_path):
    cache = EmbeddingCache(tmp_path)
    t = np.array([0.0], dtype=np.float32)
    v = np.zeros((1, 4), dtype=np.float32)
    assert cache.models() == []
    cache.save("k", "clip", t, v)
    cache.save("k", "dino", t, v)
    assert cache.models() == ["clip", "dino"]
