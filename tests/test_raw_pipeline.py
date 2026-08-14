"""Raw frame pipeline: fixed-chunk framing, embed_array, and the embed_library
fast path — all offline (no ffmpeg/torch)."""

from io import BytesIO

import numpy as np
import pytest

from peaks_vr.cache import EmbeddingCache
from peaks_vr.embedding import FakeEmbedder
from peaks_vr.models import Scene
from peaks_vr.pipeline import embed_library
from peaks_vr.sampling import FrameSampler, iter_fixed_chunks


# --- fixed-size chunk framing --------------------------------------------------


class _DribbleStream:
    """Returns at most `dribble` bytes per read to simulate pipe behaviour."""

    def __init__(self, data: bytes, dribble: int = 7):
        self._buf = BytesIO(data)
        self._dribble = dribble

    def read(self, n: int) -> bytes:
        return self._buf.read(min(n, self._dribble))


def test_fixed_chunks_exact():
    data = b"".join(bytes([i]) * 10 for i in range(5))  # 5 chunks of 10
    out = list(iter_fixed_chunks(BytesIO(data), 10))
    assert [c[0] for c in out] == [0, 1, 2, 3, 4]
    assert all(len(c) == 10 for c in out)


def test_fixed_chunks_survive_short_reads():
    data = b"".join(bytes([i]) * 10 for i in range(5))
    out = list(iter_fixed_chunks(_DribbleStream(data, dribble=3), 10))
    assert len(out) == 5
    assert out == [bytes([i]) * 10 for i in range(5)]


def test_fixed_chunks_drop_truncated_tail():
    data = b"a" * 10 + b"b" * 4  # one full frame + truncated tail
    out = list(iter_fixed_chunks(BytesIO(data), 10))
    assert out == [b"a" * 10]


def test_fixed_chunks_empty():
    assert list(iter_fixed_chunks(BytesIO(b""), 10)) == []


# --- raw vf + config -------------------------------------------------------------


def test_raw_vf_geometry():
    s = FrameSampler(interval_seconds=8.0, pipeline="raw")
    vf = s._raw_vf(resize_short=256, crop=224)
    assert "fps=1/8" in vf
    assert "scale=w=256:h=256" in vf and "flags=bicubic" in vf
    assert vf.endswith("crop=224:224")


def test_unknown_pipeline_rejected():
    with pytest.raises(ValueError):
        FrameSampler(pipeline="carrier-pigeon")


def test_peaks_pipeline_env_override(monkeypatch, tmp_path):
    from peaks_vr.config import Config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[sampling]\npipeline = "jpeg"\n')
    assert Config.load(cfg_file).sampling.pipeline == "jpeg"
    monkeypatch.setenv("PEAKS_PIPELINE", "raw")
    assert Config.load(cfg_file).sampling.pipeline == "raw"


# --- FakeEmbedder raw path --------------------------------------------------------


def test_fake_embed_array_deterministic_and_normalized():
    emb = FakeEmbedder(dim=16)
    frames = np.stack(
        [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)]
    )
    a = emb.embed_array(frames)
    b = emb.embed_array(frames)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 16)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), np.ones(3), atol=1e-6)
    assert not np.allclose(a[0], a[1])


def test_fake_embed_array_matches_bytes_hashing():
    """Raw path must agree with the PIL path for identical pixel bytes."""

    class _StubImage:
        def __init__(self, arr):
            self._arr = arr

        def tobytes(self):
            return self._arr.tobytes()

    emb = FakeEmbedder(dim=8)
    arr = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    via_array = emb.embed_array(arr[None])
    via_image = emb.embed_images([_StubImage(arr)])
    np.testing.assert_array_equal(via_array, via_image)


def test_fake_embed_array_empty():
    assert FakeEmbedder(dim=4).embed_array(np.zeros((0, 8, 8, 3), np.uint8)).shape == (0, 4)


# --- embed_library chooses the raw branch ------------------------------------------


class _RawStubSampler:
    """Sampler stub advertising the raw pipeline."""

    interval = 2.0
    mode = "interval"
    pipeline = "raw"

    def __init__(self, n_frames=6):
        self.n_frames = n_frames
        self.raw_calls = []

    def iter_frames_raw(self, path, *, resize_short, crop):
        self.raw_calls.append((path, resize_short, crop))
        for i in range(self.n_frames):
            yield i * self.interval, np.full((crop, crop, 3), i, dtype=np.uint8)

    def iter_frames(self, path):  # must NOT be used on the raw branch
        raise AssertionError("jpeg path used despite pipeline='raw'")


def _scene(id_, key):
    return Scene.from_dict(
        {
            "id": id_,
            "title": "",
            "files": [{"path": f"/m/{id_}.mp4", "fingerprints": [
                {"type": "oshash", "value": key}]}],
            "scene_markers": [],
        }
    )


def test_embed_library_raw_branch(tmp_path):
    emb = FakeEmbedder(dim=12)
    emb.raw_resize, emb.raw_crop = 64, 48  # sampler must receive these
    sampler = _RawStubSampler(n_frames=6)
    cache = EmbeddingCache(tmp_path)

    stats = embed_library(
        [_scene("1", "k1")], sampler, emb, cache, batch_size=4, log=lambda *_: None
    )
    assert stats["embedded"] == 1 and stats["frames"] == 6
    assert sampler.raw_calls == [("/m/1.mp4", 64, 48)]

    times, vecs, meta = cache.load("k1", "fake")
    assert list(times) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert vecs.shape == (6, 12)
    assert meta["pipeline"] == "raw"


def test_embed_library_parallel_workers(tmp_path):
    """workers>1: concurrent decode, same results, cache complete."""
    emb = FakeEmbedder(dim=10)
    emb.raw_resize, emb.raw_crop = 32, 32
    sampler = _RawStubSampler(n_frames=5)
    cache = EmbeddingCache(tmp_path)
    scenes = [_scene(str(i), f"k{i}") for i in range(7)]

    stats = embed_library(
        scenes, sampler, emb, cache, batch_size=3, workers=3, log=lambda *_: None
    )
    assert stats["embedded"] == 7 and stats["frames"] == 35
    for i in range(7):
        times, vecs, meta = cache.load(f"k{i}", "fake")
        assert len(times) == 5 and vecs.shape == (5, 10)
        assert meta["pipeline"] == "raw"


def test_embed_library_parallel_skips_cached_and_survives_failures(tmp_path):
    emb = FakeEmbedder(dim=8)
    emb.raw_resize, emb.raw_crop = 16, 16

    class _FlakySampler(_RawStubSampler):
        def iter_frames_raw(self, path, *, resize_short, crop):
            if "bad" in path:
                raise RuntimeError("corrupt file")
            yield from super().iter_frames_raw(
                path, resize_short=resize_short, crop=crop
            )

    sampler = _FlakySampler(n_frames=4)
    cache = EmbeddingCache(tmp_path)
    good = _scene("1", "k1")
    bad = Scene.from_dict(
        {
            "id": "2",
            "title": "",
            "files": [{"path": "/m/bad.mp4", "fingerprints": [
                {"type": "oshash", "value": "k2"}]}],
            "scene_markers": [],
        }
    )
    # pre-cache one scene so skip logic is exercised too
    embed_library([good], sampler, emb, cache, workers=1, log=lambda *_: None)

    stats = embed_library(
        [good, bad, _scene("3", "k3")], sampler, emb, cache,
        workers=2, log=lambda *_: None,
    )
    assert stats["skipped"] == 1  # k1 already cached
    assert stats["failed"] == 1  # the corrupt one
    assert stats["embedded"] == 1  # k3
    assert cache.has("k3", "fake")
    assert not cache.has("k2", "fake")


def test_embed_library_jpeg_branch_still_works(tmp_path):
    """Samplers without the raw attrs keep using the legacy path."""

    class _StubImage:
        def __init__(self, payload):
            self._payload = payload

        def tobytes(self):
            return self._payload

    class _JpegStubSampler:
        interval = 2.0

        def iter_frames(self, path):
            for i in range(3):
                yield i * 2.0, _StubImage(f"f{i}".encode())

    emb = FakeEmbedder(dim=8)
    cache = EmbeddingCache(tmp_path)
    stats = embed_library(
        [_scene("1", "k1")], _JpegStubSampler(), emb, cache, log=lambda *_: None
    )
    assert stats["embedded"] == 1 and stats["frames"] == 3
    _, _, meta = cache.load("k1", "fake")
    assert meta["pipeline"] == "jpeg"
