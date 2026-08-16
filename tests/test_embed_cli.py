"""Offline tests for the embedding CLI pieces: recursive video discovery, the
seek-based VR de-warp ffmpeg command, and de-warp-aware single-frame grabs.
subprocess is monkeypatched so no real ffmpeg runs."""

import subprocess
import types

import numpy as np

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


# --- seek-based VR de-warp sampling -----------------------------------------

def _fake_run_factory(record, *, crop):
    def fake_run(cmd, capture_output=False, check=False, **kw):
        record.append(cmd)
        # emit exactly one crop×crop rgb24 frame
        buf = (np.arange(crop * crop * 3, dtype=np.uint8)).tobytes()
        return types.SimpleNamespace(stdout=buf, stderr=b"", returncode=0)
    return fake_run


def test_dewarp_seek_builds_correct_ffmpeg_and_yields_frames(tmp_path, monkeypatch):
    rep = Reprojector.for_format(detect("scene_180_sbs.mp4"))
    sampler = FrameSampler(interval_seconds=8.0, mode="sparse",
                           hwaccel="cuda", reproject=rep)
    # skip the ffprobe duration call — fix a short duration so 2 samples plan
    monkeypatch.setattr(sampler, "probe_duration", lambda p: 12.0)
    cmds = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(cmds, crop=224))

    frames = list(sampler.iter_frames_raw("x.mp4", resize_short=256, crop=224))

    # 12s / 8s → samples at 0 and 8
    assert [round(t) for t, _ in frames] == [0, 8]
    arr = frames[0][1]
    assert arr.shape == (224, 224, 3) and arr.dtype == np.uint8
    # the ffmpeg command carries seek, NVDEC, the v360 de-warp, and rawvideo
    cmd = cmds[0]
    assert "-ss" in cmd and "-hwaccel" in cmd and "cuda" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "v360=input=he" in vf and "crop=iw/2:ih:0:0" in vf
    assert "crop=224:224" in vf
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
