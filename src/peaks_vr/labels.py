"""Label store for Tier-2 training.

A label is a yes/no verdict on a single frame: "(this scene, this timestamp)
is / isn't the thing I want." Labels are grouped by **profile** (the taste tag,
e.g. "apex" or "apex:heels") so each profile trains its own classifier.

Backed by a plain JSON file — small, human-inspectable, gitignore-friendly.
Upserts are keyed by (cache key, rounded time, profile) so re-rating a frame
overwrites rather than duplicates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time as _now


@dataclass
class Label:
    key: str  # scene cache key (file fingerprint)
    time: float  # timestamp in seconds
    label: int  # 1 = positive (want it), 0 = negative
    profile: str  # taste tag this label belongs to
    scene_id: str | None = None
    ts: float = 0.0  # when the rating was made (unix time); 0 = pre-dates the field


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._labels: dict[tuple, Label] = {}
        self.load()

    @staticmethod
    def _id(key: str, time: float, profile: str) -> tuple:
        return (key, round(float(time), 2), profile)

    def load(self) -> None:
        self._labels = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text()):
                lab = Label(**d)
                self._labels[self._id(lab.key, lab.time, lab.profile)] = lab

    def save(self) -> Path:
        # atomic: write to a temp file, then replace — a crash mid-save must
        # not corrupt an existing labels file (it's hand-collected work)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps([asdict(l) for l in self._labels.values()], indent=2)
        )
        tmp.replace(self.path)
        return self.path

    def add(
        self,
        key: str,
        time: float,
        label: int,
        profile: str,
        scene_id: str | None = None,
    ) -> None:
        lab = Label(
            key=key, time=float(time), label=int(label), profile=profile,
            scene_id=scene_id, ts=_now(),  # re-rating counts as recent activity
        )
        self._labels[self._id(key, time, profile)] = lab

    def for_profile(self, profile: str) -> list[Label]:
        return [l for l in self._labels.values() if l.profile == profile]

    def labeled_ids(self, profile: str) -> set[tuple[str, float]]:
        """(key, rounded time) pairs already labeled for `profile` — used to
        exclude them from fresh labeling candidates."""
        return {
            (l.key, round(l.time, 2))
            for l in self._labels.values()
            if l.profile == profile
        }

    def counts(self, profile: str) -> tuple[int, int]:
        labs = self.for_profile(profile)
        pos = sum(1 for l in labs if l.label == 1)
        return pos, len(labs) - pos

    def remove(self, profile: str, newer_than: float | None = None) -> int:
        """Delete labels for `profile`. With `newer_than` (unix ts), only those
        rated at/after it — an "undo the last N minutes". Returns how many were
        removed. Caller should save()."""
        drop = [
            k for k, lab in self._labels.items()
            if lab.profile == profile
            and (newer_than is None or (lab.ts or 0.0) >= newer_than)
        ]
        for k in drop:
            del self._labels[k]
        return len(drop)

    def profiles(self) -> list[str]:
        return sorted({l.profile for l in self._labels.values()})

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self):
        return iter(self._labels.values())
