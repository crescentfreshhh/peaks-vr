"""Lightweight dataclasses mirroring the slice of the Stash schema we use.

These intentionally cover only the fields step 1 needs. As later steps touch
more of the schema (galleries, performers, etc.) we extend these.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tag:
    id: str
    name: str

    @classmethod
    def from_dict(cls, d: dict) -> "Tag":
        return cls(id=str(d["id"]), name=d.get("name", ""))


@dataclass
class Marker:
    id: str
    seconds: float
    end_seconds: float | None
    title: str
    primary_tag: Tag | None

    @classmethod
    def from_dict(cls, d: dict) -> "Marker":
        pt = d.get("primary_tag")
        return cls(
            id=str(d["id"]),
            seconds=float(d.get("seconds") or 0.0),
            end_seconds=(float(d["end_seconds"]) if d.get("end_seconds") else None),
            title=d.get("title", ""),
            primary_tag=Tag.from_dict(pt) if pt else None,
        )


@dataclass
class SceneFile:
    path: str
    duration: float | None
    width: int | None
    height: int | None
    frame_rate: float | None
    video_codec: str | None
    size: int | None
    fingerprints: dict[str, str] = field(default_factory=dict)

    def fingerprint(self, *preferred: str) -> str | None:
        """Best stable fingerprint, trying `preferred` types in order then any."""
        for t in (*preferred, "oshash", "md5", "phash"):
            if t in self.fingerprints:
                return self.fingerprints[t]
        return next(iter(self.fingerprints.values()), None)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneFile":
        return cls(
            path=d.get("path", ""),
            duration=(float(d["duration"]) if d.get("duration") is not None else None),
            width=d.get("width"),
            height=d.get("height"),
            frame_rate=(
                float(d["frame_rate"]) if d.get("frame_rate") is not None else None
            ),
            video_codec=d.get("video_codec"),
            size=(int(d["size"]) if d.get("size") is not None else None),
            fingerprints={
                fp["type"]: fp["value"] for fp in (d.get("fingerprints") or [])
            },
        )


@dataclass
class Scene:
    id: str
    title: str
    files: list[SceneFile] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)

    @property
    def primary_file(self) -> SceneFile | None:
        return self.files[0] if self.files else None

    @property
    def path(self) -> str | None:
        pf = self.primary_file
        return pf.path if pf else None

    @property
    def duration(self) -> float | None:
        pf = self.primary_file
        return pf.duration if pf else None

    @property
    def fingerprint(self) -> str | None:
        pf = self.primary_file
        return pf.fingerprint() if pf else None

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        return cls(
            id=str(d["id"]),
            title=d.get("title") or "",
            files=[SceneFile.from_dict(f) for f in (d.get("files") or [])],
            markers=[Marker.from_dict(m) for m in (d.get("scene_markers") or [])],
        )
