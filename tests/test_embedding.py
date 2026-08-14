"""FakeEmbedder + a full offline pipeline pass (no torch / ffmpeg / Stash)."""

import numpy as np

from peaks_vr.cache import EmbeddingCache
from peaks_vr.embedding import FakeEmbedder, get_embedder
from peaks_vr.scoring import extract_segments, similarity_scores, smooth


class _StubImage:
    """Stands in for a PIL image: the embedder only needs .tobytes()."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


def test_fake_embedder_deterministic_and_normalized():
    emb = FakeEmbedder(dim=16)
    a1 = emb.embed_images([_StubImage(b"frame-a")])
    a2 = emb.embed_images([_StubImage(b"frame-a")])
    b = emb.embed_images([_StubImage(b"frame-b")])
    np.testing.assert_array_equal(a1, a2)  # deterministic
    assert not np.allclose(a1, b)  # different inputs differ
    np.testing.assert_allclose(np.linalg.norm(a1, axis=1), [1.0], atol=1e-6)


def test_fake_embedder_empty():
    out = FakeEmbedder(dim=8).embed_images([])
    assert out.shape == (0, 8)


def test_registry_resolves_fake():
    assert isinstance(get_embedder("fake", dim=4), FakeEmbedder)


def test_unknown_embedder_raises():
    import pytest

    with pytest.raises(ValueError):
        get_embedder("nope")


def test_end_to_end_offline_pipeline(tmp_path):
    """frames -> embed -> cache -> score-vs-reference -> segments.

    Build a synthetic scene where frames 10-14 are copies of a 'loved' frame;
    everything else is noise. The reference is that loved frame, so the scorer
    should light up exactly that stretch and produce one segment there.
    """
    emb = FakeEmbedder(dim=24)
    interval = 2.0

    loved = _StubImage(b"the-good-stuff")
    frames = [_StubImage(f"noise-{i}".encode()) for i in range(25)]
    for i in range(10, 15):
        frames[i] = loved  # the apex region

    vecs = emb.embed_images(frames)
    times = np.arange(len(frames), dtype=np.float32) * interval

    # cache round-trip (proves resumable storage works on these vectors)
    cache = EmbeddingCache(tmp_path)
    cache.save("scene1", emb.name, times, vecs, meta={"interval": interval})
    times, vecs, _ = cache.load("scene1", emb.name)

    reference = emb.embed_images([loved])
    scores = smooth(similarity_scores(vecs, reference), window=1)

    # the loved frames score ~1.0; noise scores well below
    assert scores[12] > 0.99
    assert scores[0] < 0.9

    segs = extract_segments(
        scores, times, high=0.9, low=0.7, min_duration=2.0, merge_gap=2.0
    )
    assert len(segs) == 1
    # region is frames 10-14 -> times 20..28
    assert segs[0].start == 20.0 and segs[0].end == 28.0


def test_clip_cache_name_namespaces_by_variant():
    from peaks_vr.embedding import clip_cache_name

    # ViT-B-32 keeps the legacy dir; other variants get their own so their
    # different-dim vectors never collide in the same cache
    assert clip_cache_name("ViT-B-32") == "clip"
    assert clip_cache_name("ViT-L-14") == "clip-vit-l-14"
    assert clip_cache_name("ViT-H-14") == "clip-vit-h-14"
