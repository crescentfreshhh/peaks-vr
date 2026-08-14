import numpy as np

from peaks_vr.scoring import (
    Segment,
    extract_segments,
    l2_normalize,
    similarity_scores,
    smooth,
)


def test_l2_normalize_unit_rows():
    v = np.array([[3.0, 4.0], [0.0, 2.0]])
    out = l2_normalize(v)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_similarity_identical_is_one():
    refs = np.array([[1.0, 0.0, 0.0]])
    frames = np.array([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0]])  # parallel, orthogonal
    scores = similarity_scores(frames, refs)
    np.testing.assert_allclose(scores, [1.0, 0.0], atol=1e-6)


def test_similarity_max_vs_mean_reduce():
    refs = np.array([[1.0, 0.0], [0.0, 1.0]])
    frames = np.array([[1.0, 0.0]])  # matches ref 0 exactly, orthogonal to ref 1
    assert similarity_scores(frames, refs, reduce="max")[0] == 1.0
    np.testing.assert_allclose(similarity_scores(frames, refs, reduce="mean"), [0.5])


def test_smooth_constant_unchanged_and_length_preserved():
    s = np.ones(10, dtype=np.float32)
    out = smooth(s, window=3)
    assert out.shape == s.shape
    np.testing.assert_allclose(out, s, atol=1e-6)


def test_smooth_window_one_is_noop():
    s = np.array([0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(smooth(s, 1), s)


def _times(n, interval=1.0):
    return np.arange(n, dtype=np.float32) * interval


def test_single_bump_one_segment():
    scores = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.float32)
    segs = extract_segments(scores, _times(10), high=0.5, min_duration=1.0, merge_gap=0)
    assert len(segs) == 1
    assert segs[0].start == 3.0 and segs[0].end == 6.0
    assert segs[0].peak_score == 1.0


def test_hysteresis_keeps_segment_through_dip():
    # dips to 0.4 (below high=0.5 but above low=0.3) -> stays one segment
    scores = np.array([0, 0.8, 0.4, 0.8, 0, 0], dtype=np.float32)
    segs = extract_segments(
        scores, _times(6), high=0.5, low=0.3, min_duration=1.0, merge_gap=0
    )
    assert len(segs) == 1
    assert segs[0].start == 1.0 and segs[0].end == 3.0


def test_min_duration_drops_short_spike():
    scores = np.array([0, 0, 1, 0, 0], dtype=np.float32)  # single-sample spike
    segs = extract_segments(scores, _times(5), high=0.5, min_duration=2.0, merge_gap=0)
    assert segs == []


def test_merge_gap_joins_neighbours():
    # two bumps at [0,1] and [4,5]; the silent span between them is 3s (t=1→4)
    scores = np.array([1, 1, 0, 0, 1, 1], dtype=np.float32)
    merged = extract_segments(
        scores, _times(6), high=0.5, min_duration=1.0, merge_gap=3.0
    )
    assert len(merged) == 1
    assert merged[0].start == 0.0 and merged[0].end == 5.0
    unmerged = extract_segments(
        scores, _times(6), high=0.5, min_duration=1.0, merge_gap=1.0
    )
    assert len(unmerged) == 2


def test_max_duration_splits():
    scores = np.ones(10, dtype=np.float32)
    segs = extract_segments(
        scores, _times(10), high=0.5, min_duration=1.0, merge_gap=0, max_duration=3.0
    )
    assert len(segs) >= 3
    assert all(s.duration <= 3.0 + 1e-6 for s in segs)


def test_pad_clamped_to_bounds():
    scores = np.array([0, 1, 1, 0], dtype=np.float32)
    segs = extract_segments(
        scores, _times(4), high=0.5, min_duration=1.0, merge_gap=0, pad=10.0
    )
    assert segs[0].start == 0.0  # clamped to series start, not negative
    assert segs[0].end == 3.0  # clamped to series end


def test_empty_input():
    assert extract_segments(np.array([]), np.array([]), high=0.5) == []


def test_segment_helpers():
    s = Segment(start=10.0, end=20.0, peak_score=0.9, mean_score=0.7)
    assert s.duration == 10.0 and s.midpoint == 15.0
