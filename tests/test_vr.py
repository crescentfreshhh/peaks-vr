"""Offline tests for the VR-specific scaffold: format detection, the
reprojection filter builder, and the sampler de-warp wiring. All pure string /
dataclass logic — no ffmpeg, torch, or GPU required."""

import pytest

from peaks_vr.reprojection import Reprojector
from peaks_vr.sampling import FrameSampler, SamplerError
from peaks_vr.vr_format import Projection, StereoLayout, VRFormat, detect


# --- format detection -------------------------------------------------------

def test_detect_180_sbs_equirect():
    fmt = detect("MyScene_180_sbs_8K.mp4")
    assert fmt.projection is Projection.EQUIRECT
    assert fmt.layout is StereoLayout.SBS
    assert fmt.fov_deg == 180.0
    assert fmt.is_stereo and fmt.is_known
    assert fmt.confidence > 0


def test_detect_fisheye_mkx200_tb():
    fmt = detect("clip_MKX200_TB.mp4")
    assert fmt.projection is Projection.FISHEYE
    assert fmt.layout is StereoLayout.TB
    assert fmt.fov_deg == 200.0


def test_detect_unknown_is_not_known():
    fmt = detect("random_flat_video.mp4")
    assert not fmt.is_known
    assert fmt.layout is StereoLayout.UNKNOWN


# --- reprojection filter builder -------------------------------------------

def test_reprojector_sbs_equirect_filter():
    fmt = detect("s_180_sbs.mp4")
    filt = Reprojector.for_format(fmt, viewport_fov_deg=100).ffmpeg_filter()
    assert filt.startswith("crop=iw/2:ih:0:0,")   # left eye
    assert "v360=input=he:output=flat" in filt
    assert "h_fov=100" in filt


def test_reprojector_tb_fisheye_filter():
    fmt = detect("s_MKX200_tb.mp4")
    filt = Reprojector.for_format(fmt).ffmpeg_filter()
    assert filt.startswith("crop=iw:ih/2:0:0,")    # top eye
    assert "input=fisheye" in filt
    assert "ih_fov=200" in filt


def test_reprojector_unsupported_projection_raises():
    fmt = VRFormat(Projection.EQUIRECT, StereoLayout.SBS, fov_deg=360.0)
    with pytest.raises(NotImplementedError):
        Reprojector(fmt).ffmpeg_filter()


# --- sampler wiring ---------------------------------------------------------

def test_sampler_without_reproject_is_flat_2d():
    # No de-warp by default → filtergraph identical to 2D-peaks (no v360).
    vf = FrameSampler(interval_seconds=2, frame_size=288)._vf()
    assert "v360" not in vf
    assert "crop=" not in vf


def test_sampler_with_reproject_prepends_dewarp():
    rep = Reprojector.for_format(detect("s_180_sbs.mp4"))
    sampler = FrameSampler(interval_seconds=2, frame_size=288, reproject=rep)
    vf = sampler._vf()
    assert "v360=input=he" in vf
    # de-warp comes before the model-input scale
    assert vf.index("v360") < vf.index("scale=")
    # ...and also in the raw pipeline filtergraph
    raw = sampler._raw_vf(resize_short=256, crop=224)
    assert "v360=input=he" in raw
    assert raw.index("v360") < raw.index("crop=224")


def test_sampler_sparse_with_reproject_fails_loud():
    rep = Reprojector.for_format(detect("s_180_sbs.mp4"))
    sampler = FrameSampler(mode="sparse", reproject=rep)
    with pytest.raises(SamplerError):
        list(sampler.iter_frames_raw("nonexistent.mp4", resize_short=256, crop=224))
