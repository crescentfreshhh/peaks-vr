"""Offline tests for the embedding CLI pieces: recursive video discovery, the
seek-based VR de-warp ffmpeg command, and de-warp-aware single-frame grabs.
subprocess is monkeypatched so no real ffmpeg runs."""

import subprocess
import types

import numpy as np
import pytest

from peaks_vr.cli import VIDEO_EXTS, iter_video_files
from peaks_vr.reprojection import Reprojector
from peaks_vr.sampling import FrameSampler
from peaks_vr.vr_format import detect


# --- directory discovery ----------------------------------------------------

def test_iter_video_files_scans_dirs_recursively(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "s1_180_sbs.mp4").write_bytes(b"x")
    (tmp_path / "s2_MKX200_tb.mkv").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")          # ignored
    (tmp_path / "poster.jpg").write_bytes(b"x")         # ignored

    found = iter_video_files([str(tmp_path)])
    assert len(found) == 2
    assert all(f.rsplit(".", 1)[-1] in {e[1:] for e in VIDEO_EXTS} for f in found)
    assert found == sorted(found)  # stable order


def test_iter_video_files_keeps_explicit_files_and_dedupes(tmp_path):
    f = tmp_path / "clip_180_sbs.mp4"
    f.write_bytes(b"x")
    out = iter_video_files([str(f), str(f), str(tmp_path)])
    assert out == [str(f)]


# --- in-process VR de-warp sampling (the fast default) ----------------------

def _make_sbs(path, *, frames=100, w=512, h=256):
    """A synthetic 180 SBS-shaped video (two eyes side by side)."""
    av = pytest.importorskip("av")
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=5)
    s.width, s.height = w, h
    s.pix_fmt = "yuv420p"
    s.codec_context.gop_size = 10  # keyframe every 2s
    for i in range(frames):
        a = np.full((h, w, 3), (i * 7) % 256, np.uint8)
        a[:, : (i % w), 1] = 255
        for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def test_inproc_vr_dewarp_end_to_end(tmp_path):
    """The default VR path decodes keyframes in ONE process and de-warps them
    in-process — validated on a real (synthetic) SBS video, no per-sample ffmpeg."""
    vid = tmp_path / "scene_180_sbs.mp4"
    _make_sbs(vid)
    rep = Reprojector.for_format(detect("scene_180_sbs.mp4"), input_size=256)
    sampler = FrameSampler(interval_seconds=4.0, mode="sparse", reproject=rep)

    frames = list(sampler.iter_frames_raw(str(vid), resize_short=256, crop=224))
    assert len(frames) >= 3
    for t, arr in frames:
        assert arr.shape == (224, 224, 3) and arr.dtype == np.uint8
    times = [t for t, _ in frames]
    assert times == sorted(times)


# --- fallback: per-sample ffmpeg seek path (used only if `av` is missing) ----

def _fake_run_factory(record, *, crop):
    def fake_run(cmd, capture_output=False, check=False, **kw):
        record.append(cmd)
        buf = (np.arange(crop * crop * 3, dtype=np.uint8)).tobytes()
        return types.SimpleNamespace(stdout=buf, stderr=b"", returncode=0)
    return fake_run


def test_dewarp_seek_fallback_builds_correct_ffmpeg(tmp_path, monkeypatch):
    rep = Reprojector.for_format(detect("scene_180_sbs.mp4"))
    sampler = FrameSampler(interval_seconds=8.0, mode="sparse",
                           hwaccel="cuda", reproject=rep)
    monkeypatch.setattr(sampler, "probe_duration", lambda p: 12.0)
    cmds = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(cmds, crop=224))

    # call the fallback path directly (iter_frames_raw uses in-process when av is present)
    frames = list(sampler._iter_frames_dewarp_seek("x.mp4", resize_short=256, crop=224))
    assert [round(t) for t, _ in frames] == [0, 8]
    assert frames[0][1].shape == (224, 224, 3)
    cmd = cmds[0]
    # per-sample seek is CPU-decoded even with hwaccel="cuda" (NVDEC init per
    # process would dwarf a single keyframe decode)
    assert "-hwaccel" not in cmd
    assert "-noaccurate_seek" in cmd and "-ss" in cmd
    assert cmd.index("-ss") < cmd.index("-i")
    vf = cmd[cmd.index("-vf") + 1]
    assert "v360=input=he" in vf and "scale=1600:1600" in vf and "crop=224:224" in vf
    assert "rawvideo" in cmd


def test_grab_frame_injects_dewarp_when_reprojector_set(monkeypatch):
    from PIL import Image

    rep = Reprojector.for_format(detect("s_MKX200_tb.mp4"))
    sampler = FrameSampler(reproject=rep)
    cmds = []

    def fake_run(cmd, capture_output=False, check=False, **kw):
        cmds.append(cmd)
        import io
        b = io.BytesIO(); Image.new("RGB", (8, 8)).save(b, format="JPEG")
        return types.SimpleNamespace(stdout=b.getvalue(), stderr=b"", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler.grab_frame("x.mp4", 30.0)
    vf = cmds[0][cmds[0].index("-vf") + 1]
    assert "input=fisheye" in vf and "crop=iw:ih/2:0:0" in vf  # TB top eye


def test_grab_frame_no_dewarp_without_reprojector(monkeypatch):
    from PIL import Image

    sampler = FrameSampler()  # no reprojector
    cmds = []

    def fake_run(cmd, capture_output=False, check=False, **kw):
        cmds.append(cmd)
        import io
        b = io.BytesIO(); Image.new("RGB", (8, 8)).save(b, format="JPEG")
        return types.SimpleNamespace(stdout=b.getvalue(), stderr=b"", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler.grab_frame("x.mp4", 30.0)
    assert "-vf" not in cmds[0]  # flat grab, no de-warp
