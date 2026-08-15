"""Recommend similar moments (README feature #3).

Once you've ❤️-flagged moments (feature #2) and embedded your library
(``peaks-vr embed``), this turns your likes into a ranked list of *more moments
like them*. Per the README it's a direct reuse of the 2D engine's taste /
similarity math onto the VR embedding space — nothing here re-implements the
scoring:

    marks (LabelStore)                pipeline.build_training_set
        + cached vectors      ─────►  → the liked frame vectors
                                       scoring.make_similarity_scorer(liked)
    every cached scene        ─────►  pipeline.score_scene → high-similarity
                                       segments, ranked into a Playlist.

Using ``make_similarity_scorer(..., reduce="max")`` over *all* the liked vectors
is the nearest-neighbor / multi-mode taste model: a candidate scores by how
close it is to your closest liked moment, so distinct interests each pull in
their own neighbors (no single averaged centroid that blurs them together).

The result is a :class:`Playlist` of :class:`Moment`s — the input the VR DJ
(feature #4, :mod:`peaks_vr.dj`) plays back in the headset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .cache import EmbeddingCache
from .config import ScoringConfig
from .pipeline import build_training_set, score_scene
from .scoring import make_similarity_scorer


@dataclass
class Moment:
    """One recommended moment: a scored, bounded stretch of a scene.

    ``path`` is the source file the DJ loads; ``key`` is the cache key it came
    from; ``score`` is the peak similarity to your liked moments.
    """

    path: str | None
    key: str
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Moment":
        return cls(path=d.get("path"), key=d["key"], start=float(d["start"]),
                   end=float(d["end"]), score=float(d.get("score", 0.0)))


@dataclass
class Playlist:
    """An ordered set of recommended moments + the profile it came from."""

    profile: str
    moments: list[Moment] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.moments)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "profile": self.profile,
            "count": len(self.moments),
            "moments": [m.to_dict() for m in self.moments],
        }, indent=2))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Playlist":
        d = json.loads(Path(path).read_text())
        return cls(profile=d.get("profile", "apex"),
                   moments=[Moment.from_dict(m) for m in d.get("moments", [])])


def taste_vectors(store, cache: EmbeddingCache, model_name: str,
                  profile: str) -> np.ndarray:
    """The embedding vectors of your ❤️-flagged moments for ``profile``.

    Reuses :func:`peaks_vr.pipeline.build_training_set`, which snaps each label
    to its nearest cached frame vector and skips marks whose scene isn't embedded
    yet. Marks are all positive, so we keep the positive rows.
    """
    X, y = build_training_set(store, cache, model_name, profile)
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return X[y == 1]


def recommend(cache: EmbeddingCache, model_name: str, liked: np.ndarray,
              scoring: ScoringConfig | None = None, *, limit: int = 50,
              reduce: str = "max", exclude_keys: set[str] | None = None,
              profile: str = "apex") -> Playlist:
    """Score every cached scene against the ``liked`` vectors and return the
    top ``limit`` moments as a ranked :class:`Playlist`.

    ``exclude_keys`` drops scenes from the results (e.g. the scenes the likes
    came from, if you only want *new* discoveries).
    """
    if liked.shape[0] == 0:
        raise ValueError("no liked vectors — flag some moments and embed their "
                         "scenes first")
    scoring = scoring or ScoringConfig()
    score_frames = make_similarity_scorer(liked, reduce=reduce)
    exclude_keys = exclude_keys or set()

    moments: list[Moment] = []
    for key in cache.keys(model_name):
        if key in exclude_keys:
            continue
        times, vecs, meta = cache.load(key, model_name)
        if len(times) == 0:
            continue
        for seg in score_scene(times, vecs, score_frames, scoring):
            moments.append(Moment(
                path=meta.get("path"), key=key,
                start=round(seg.start, 3), end=round(seg.end, 3),
                score=round(seg.peak_score, 4),
            ))
    moments.sort(key=lambda m: m.score, reverse=True)
    return Playlist(profile=profile, moments=moments[:limit])


def recommend_from_labels(cache: EmbeddingCache, store, model_name: str,
                          profile: str, scoring: ScoringConfig | None = None,
                          *, limit: int = 50, reduce: str = "max",
                          exclude_seed_scenes: bool = False) -> Playlist:
    """End-to-end: your marks → a ranked :class:`Playlist`.

    With ``exclude_seed_scenes`` the scenes your likes came from are left out, so
    the playlist is only *new* moments rather than replaying the seeds.
    """
    liked = taste_vectors(store, cache, model_name, profile)
    exclude = None
    if exclude_seed_scenes:
        exclude = {l.key for l in store.for_profile(profile) if l.label == 1}
    return recommend(cache, model_name, liked, scoring, limit=limit,
                     reduce=reduce, exclude_keys=exclude, profile=profile)
