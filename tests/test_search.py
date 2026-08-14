import numpy as np

from peaks_vr.cache import EmbeddingCache
from peaks_vr.search import Hit, SearchIndex


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _seed(cache, key, scene_id, vecs, times):
    cache.save(
        key, "dino",
        np.asarray(times, dtype=np.float32),
        np.asarray(vecs, dtype=np.float32),
        meta={"scene_id": scene_id},
    )


def test_build_stacks_all_scenes(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    _seed(cache, "k2", "2", [_unit([0, 0, 1])], [0.0])
    idx = SearchIndex(cache, "dino").build()
    assert idx.size == 3 and idx.dim == 3
    assert set(idx.scene_ids) == {"1", "2"}


def test_build_preallocates_matches_manual_stack(tmp_path):
    # the preallocated build must produce the exact same matrix/times/rows as a
    # naive per-scene stack (parity guard for the memory-lean rewrite).
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    _seed(cache, "k2", "2", [_unit([0, 0, 1])], [4.0])
    _seed(cache, "k3", "3", [_unit([1, 1, 0]), _unit([0, 1, 1])], [1.0, 2.0])
    keys = ["k1", "k2", "k3"]
    idx = SearchIndex(cache, "dino").build(keys)

    expected = np.concatenate([cache.load(k, "dino")[1] for k in keys], axis=0)
    assert idx.matrix.dtype == np.float32
    assert np.array_equal(idx.matrix, expected)
    assert idx.times.tolist() == [0.0, 8.0, 4.0, 1.0, 2.0]
    assert idx._key_rows == {"k1": (0, 2), "k2": (2, 3), "k3": (3, 5)}
    assert idx.keys == ["k1", "k1", "k2", "k3", "k3"]


def test_peek_count_reads_frame_count(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    assert cache.peek_count("k1", "dino") == 2


def test_build_tolerates_count_growth_between_passes(tmp_path, monkeypatch):
    # a scene may gain frames between the count pass and the load pass during a
    # live embed; the extra rows are clamped, never an out-of-bounds write.
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])], [0, 1, 2])
    real_peek = cache.peek_count
    monkeypatch.setattr(cache, "peek_count", lambda k, m: 1 if k == "k1" else real_peek(k, m))
    idx = SearchIndex(cache, "dino").build(["k1"])
    assert idx.size == 1 and idx.dim == 3  # sized to the undercount, clamped safely


def test_search_min_score_returns_all_above_floor(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # 5 scenes at varying cosine to the query axis [1,0,0]
    for i, v in enumerate([[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0.2, 1, 0], [0, 1, 0]]):
        _seed(cache, f"k{i}", str(i), [_unit(v)], [0.0])
    idx = SearchIndex(cache, "dino").build()
    q = _unit([1, 0, 0])

    # unbounded threshold search returns EVERY frame with cosine >= floor (not top 60).
    # note vectors are unit-normalized, so [0.8,0.2,0] scores ~0.97, not 0.8.
    hits = idx.search(q, top_k=None, min_score=0.7)
    assert len(hits) == 3  # scenes 0,1,2 clear 0.7; 3 (~0.20) and 4 (0.0) don't
    assert all(h.score >= 0.7 for h in hits)
    assert [h.scene_id for h in hits] == ["0", "1", "2"]  # ordered best-first
    # a looser floor returns more; a stricter one fewer
    assert len(idx.search(q, top_k=None, min_score=0.0)) == 5
    assert len(idx.search(q, top_k=None, min_score=0.999)) == 1  # only the exact match


def test_search_min_score_respects_per_scene(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # one scene with 4 near-identical high-scoring frames
    _seed(cache, "k1", "1", [_unit([1, 0, 0])] * 4, [0, 1, 2, 3])
    _seed(cache, "k2", "2", [_unit([0.95, 0.05, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()
    hits = idx.search(_unit([1, 0, 0]), top_k=None, per_scene=1, min_score=0.5)
    # per_scene=1 caps scene 1 to a single moment even though 4 qualify
    assert sum(1 for h in hits if h.scene_id == "1") == 1
    assert {h.scene_id for h in hits} == {"1", "2"}


def test_search_ranks_by_cosine(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0])], [0.0])
    _seed(cache, "k2", "2", [_unit([0.9, 0.1, 0])], [0.0])
    _seed(cache, "k3", "3", [_unit([0, 1, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()

    hits = idx.search(_unit([1, 0, 0]), top_k=3)
    assert [h.scene_id for h in hits] == ["1", "2", "3"]  # by descending cosine
    assert hits[0].score > hits[1].score > hits[2].score
    assert isinstance(hits[0], Hit)


def test_search_empty_index(tmp_path):
    idx = SearchIndex(EmbeddingCache(tmp_path), "dino").build()
    assert idx.size == 0
    assert idx.search(np.array([1.0, 0, 0])) == []


def test_per_scene_cap_spreads_results(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # scene 1 has 5 near-identical strong matches; scene 2 has one slightly worse
    _seed(cache, "k1", "1", [_unit([1, 0, 0.01 * i]) for i in range(5)],
          [i * 8.0 for i in range(5)])
    _seed(cache, "k2", "2", [_unit([0.8, 0.2, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()

    capped = idx.search(_unit([1, 0, 0]), top_k=10, per_scene=2)
    from_scene1 = sum(1 for h in capped if h.scene_id == "1")
    assert from_scene1 == 2  # capped
    assert any(h.scene_id == "2" for h in capped)  # scene 2 still surfaces


def test_search_by_frame_excludes_own_scene(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0.99, 0.01, 0])], [0.0, 8.0])
    _seed(cache, "k2", "2", [_unit([0.95, 0.05, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()

    hits = idx.search_by_frame("k1", 0.0, top_k=5)
    assert all(h.key != "k1" for h in hits)  # own scene excluded
    assert hits and hits[0].scene_id == "2"


def test_vector_at_picks_nearest_time(tmp_path):
    cache = EmbeddingCache(tmp_path)
    a, b = _unit([1, 0, 0]), _unit([0, 1, 0])
    _seed(cache, "k1", "1", [a, b], [0.0, 10.0])
    idx = SearchIndex(cache, "dino").build()

    np.testing.assert_allclose(idx.vector_at("k1", 1.0), a, atol=2e-3)  # nearest 0.0
    np.testing.assert_allclose(idx.vector_at("k1", 9.0), b, atol=2e-3)  # nearest 10.0
    assert idx.vector_at("missing", 0.0) is None


def test_build_only_requested_keys(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0])], [0.0])
    _seed(cache, "k2", "2", [_unit([0, 1, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build(keys=["k1"])
    assert idx.size == 1 and idx.scene_ids == ["1"]
