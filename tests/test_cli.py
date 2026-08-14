"""End-to-end CLI test: sample -> embed -> cache -> score on a REAL video
generated with PyAV, driven entirely through ``peaks_vr.cli.main`` with the
offline FakeEmbedder. No torch, no GPU, no system ffmpeg (sparse mode decodes
via PyAV's bundled codecs)."""

import numpy as np
import pytest

av = pytest.importorskip("av")

from peaks_vr.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 30s, 5fps, 64x48 mp4 with a keyframe every 2s (gop=10)."""
    path = str(tmp_path_factory.mktemp("vid") / "scene.mp4")
    container = av.open(path, "w")
    stream = container.add_stream("mpeg4", rate=5)
    stream.width, stream.height = 64, 48
    stream.pix_fmt = "yuv420p"
    stream.codec_context.gop_size = 10
    for i in range(150):  # 30 seconds
        arr = np.full((48, 64, 3), (i * 7) % 256, dtype=np.uint8)
        arr[:, : (i % 64), 0] = 255
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _stills_dir(tmp_path):
    """A couple of reference stills for the score command."""
    from PIL import Image

    d = tmp_path / "refs"
    d.mkdir()
    for i, shade in enumerate((40, 200)):
        Image.new("RGB", (64, 48), (shade, shade, shade)).save(d / f"ref{i}.png")
    return str(d)


def test_embed_then_score_end_to_end(video, tmp_path, capsys):
    cache = str(tmp_path / "cache")

    # embed: sample -> FakeEmbedder -> cache
    rc = main(["--cache", cache, "embed", video, "--interval", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 embedded" in out
    # the cache now holds a fake-model .npz for this file
    assert list((tmp_path / "cache" / "fake").glob("*.npz"))

    # embed again is idempotent (resumable): the scene is skipped
    rc = main(["--cache", cache, "embed", video, "--interval", "4"])
    assert rc == 0
    assert "1 skipped" in capsys.readouterr().out

    # score: cached vectors vs reference stills, dry run
    rc = main(["--cache", cache, "score", video, "-r", _stills_dir(tmp_path),
               "--high", "0.2", "--low", "0.1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "score:" in out and "scene(s)" in out


def test_score_before_embed_warns(video, tmp_path, capsys):
    # Scoring a scene that was never embedded is a clean no-op, not a crash.
    rc = main(["--cache", str(tmp_path / "empty"), "score", video,
               "-r", _stills_dir(tmp_path)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "not embedded yet" in err
