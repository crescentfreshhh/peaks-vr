from peaks_vr.sampling import FrameSampler, plan_timestamps


def test_plan_matches_ffmpeg_fps_filter():
    # fps=1/interval emits frame i from source time ~i*interval, starting at 0
    assert plan_timestamps(duration=10.0, interval=2.0) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_plan_custom_offset():
    assert plan_timestamps(10.0, 5.0, offset=1.0) == [1.0, 6.0]


def test_plan_short_clip_yields_first_frame():
    # even a clip shorter than one interval produces its first frame
    assert plan_timestamps(0.5, 2.0) == [0.0]


def test_plan_zero_or_negative():
    assert plan_timestamps(0.0, 2.0) == []
    assert plan_timestamps(10.0, 0.0) == []
    assert plan_timestamps(-5.0, 2.0) == []


def test_plan_all_within_duration():
    ts = plan_timestamps(100.0, 3.0)
    assert all(0 <= t < 100.0 for t in ts)
    assert ts == sorted(ts)


def test_vf_includes_fps_and_downscale():
    s = FrameSampler(interval_seconds=2.0, frame_size=288)
    vf = s._vf()
    assert "fps=1/2" in vf
    assert "scale=w=288:h=288" in vf and "force_original_aspect_ratio=increase" in vf


def test_vf_no_scale_when_disabled():
    s = FrameSampler(interval_seconds=2.5, frame_size=0)
    assert s._vf() == "fps=1/2.5"
