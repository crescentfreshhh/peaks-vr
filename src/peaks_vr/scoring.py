"""Turn per-frame taste scores into timestamp-segments ("apexes").

This module is pure numpy — no Stash, no torch, no ffmpeg — so it is fully
unit-tested offline. Two pieces:

  1. scoring     — frame embeddings → a per-frame score (Tier 1: similarity to
                   reference vectors; Tier 2 will swap in a trained classifier
                   that emits the same shape of score array).
  2. segmenting  — a score array + timestamps → merged segments, via smoothing
                   and hysteresis thresholding (standard highlight detection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization. Accepts (n, d) or (d,)."""
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        n = np.linalg.norm(vecs) + eps
        return vecs / n
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + eps
    return vecs / norms


def similarity_scores(
    frames: np.ndarray, references: np.ndarray, reduce: str = "max"
) -> np.ndarray:
    """Cosine similarity of each frame to a set of reference vectors.

    frames:     (n, d) frame embeddings
    references: (m, d) reference embeddings (the examples you love)
    reduce:     "max"  → score = closest single reference (sharp, specific)
                "mean" → score = average over references (smoother, broader)

    Returns (n,) scores in roughly [-1, 1].
    """
    frames = l2_normalize(frames)
    references = l2_normalize(references)
    sims = frames @ references.T  # (n, m) cosine sims
    if reduce == "max":
        return sims.max(axis=1)
    if reduce == "mean":
        return sims.mean(axis=1)
    raise ValueError(f"unknown reduce: {reduce!r}")


def make_similarity_scorer(references: np.ndarray, reduce: str = "max"):
    """Return a frame-scorer closure for Tier 1: vecs -> per-frame similarity.

    Matches the signature of a trained classifier's `predict_proba`, so the two
    tiers are interchangeable downstream.
    """

    def score_frames(vecs: np.ndarray) -> np.ndarray:
        return similarity_scores(vecs, references, reduce=reduce)

    return score_frames


def normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    """Optionally rescale a scene's raw scores before thresholding.

    "none"    — raw scores; thresholds are absolute (cosine sims / probas).
    "scene-z" — z-score within the scene: thresholds become "N standard
                deviations above this scene's own baseline" (e.g. high=2.0).
                Robust to the fact that raw DINOv2/CLIP cosine baselines vary
                a lot between libraries; the trade-off is that even a scene
                with nothing you like still has relative peaks.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if mode in ("", "none") or scores.size == 0:
        return scores
    if mode == "scene-z":
        std = float(scores.std())
        if std < 1e-8:
            return np.zeros_like(scores)
        return (scores - float(scores.mean())) / std
    raise ValueError(f"unknown normalize mode: {mode!r}")


def smooth(scores: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoothing. window<=1 is a no-op.

    Reflect-pads the edges so the output length matches the input.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if window <= 1 or scores.size == 0:
        return scores
    window = min(window, scores.size)
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(scores, pad, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad : pad + scores.size]


@dataclass
class Segment:
    """One apex: a contiguous high-scoring stretch of a scene."""

    start: float  # seconds
    end: float  # seconds
    peak_score: float
    mean_score: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def extract_segments(
    scores: np.ndarray,
    times: np.ndarray,
    *,
    high: float,
    low: float | None = None,
    min_duration: float = 3.0,
    merge_gap: float = 2.0,
    max_duration: float | None = None,
    pad: float = 0.0,
) -> list[Segment]:
    """Hysteresis-threshold a score series into segments.

    Enter a segment when the score rises to `high`; stay in it until the score
    drops below `low` (defaults to high). Two thresholds prevent flicker around
    a single cutoff.

    times: timestamp (seconds) of each score sample, same length as `scores`,
           assumed sorted ascending.

    Post-processing, in order: drop segments shorter than `min_duration`, merge
    neighbours closer than `merge_gap`, then split anything longer than
    `max_duration`. `pad` extends each side (clamped to the series bounds).
    """
    scores = np.asarray(scores, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if scores.size != times.size:
        raise ValueError("scores and times must be the same length")
    if scores.size == 0:
        return []
    if low is None:
        low = high

    # 1. hysteresis scan → index ranges [start_idx, end_idx] inclusive
    ranges: list[tuple[int, int]] = []
    in_seg = False
    start_idx = 0
    for i, sc in enumerate(scores):
        if not in_seg and sc >= high:
            in_seg, start_idx = True, i
        elif in_seg and sc < low:
            ranges.append((start_idx, i - 1))
            in_seg = False
    if in_seg:
        ranges.append((start_idx, scores.size - 1))

    # 2. ranges → Segments with time bounds + stats
    segs: list[Segment] = []
    for a, b in ranges:
        segs.append(
            Segment(
                start=float(times[a]),
                end=float(times[b]),
                peak_score=float(scores[a : b + 1].max()),
                mean_score=float(scores[a : b + 1].mean()),
            )
        )

    # 3. merge neighbours closer than merge_gap (means weighted by duration)
    def _merge_pair(a: Segment, b: Segment) -> Segment:
        total = a.duration + b.duration
        mean = (
            (a.mean_score * a.duration + b.mean_score * b.duration) / total
            if total > 0
            else (a.mean_score + b.mean_score) / 2.0
        )
        return Segment(
            start=a.start,
            end=max(a.end, b.end),
            peak_score=max(a.peak_score, b.peak_score),
            mean_score=mean,
        )

    merged: list[Segment] = []
    for seg in segs:
        if merged and seg.start - merged[-1].end <= merge_gap:
            merged[-1] = _merge_pair(merged[-1], seg)
        else:
            merged.append(seg)

    # 4. drop too-short
    kept = [s for s in merged if s.duration >= min_duration]

    # 5. optional max-duration split
    if max_duration:
        split: list[Segment] = []
        for s in kept:
            if s.duration <= max_duration:
                split.append(s)
                continue
            t = s.start
            while t < s.end:
                split.append(
                    Segment(
                        start=t,
                        end=min(t + max_duration, s.end),
                        peak_score=s.peak_score,
                        mean_score=s.mean_score,
                    )
                )
                t += max_duration
        kept = split

    # 6. padding (clamped to the sampled time range); padding can make
    #    neighbours overlap, so re-merge any that now touch
    if pad:
        lo, hi = float(times[0]), float(times[-1])
        padded = [
            Segment(
                start=max(lo, s.start - pad),
                end=min(hi, s.end + pad),
                peak_score=s.peak_score,
                mean_score=s.mean_score,
            )
            for s in kept
        ]
        kept = []
        for s in padded:
            if kept and s.start <= kept[-1].end:
                kept[-1] = _merge_pair(kept[-1], s)
            else:
                kept.append(s)
    return kept
