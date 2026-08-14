"""A persistent record of scenes that failed to embed.

Sparse sampling seeks to a keyframe and decodes one frame at a time — a far
pickier path than the linear decode VLC/Stash use for playback, so a file that
plays effortlessly can still fail here (broken seek index, open-GOP, unusual
pixel formats). Rather than let those casualties scroll past in a long run,
`embed_library` records each one here (keyed by the same stable fingerprint the
cache uses) and clears the entry if a later pass succeeds.

`peaks fix` reads this log and retries each scene through a fallback ladder
(sparse without NVDEC, then a full linear decode) — see `Service.run_fix`.

The JSON file is a dict keyed by cache key so re-recording a scene overwrites
rather than duplicates. Writes are atomic (temp + replace) and guarded by a
lock, since an embed pass records from several worker-fed calls in one process.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


def failure_log_for(cfg) -> "FailureLog":
    """The failure log that sits alongside the embedding cache (so it lands in
    the same persistent `/config/cache` volume on the container)."""
    return FailureLog(Path(cfg.embedding.cache_dir).parent / "failures.json")


class FailureLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}  # corrupt/half-written log: treat as empty, don't crash a run

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def entries(self) -> list[dict]:
        """All recorded failures (newest-recorded last)."""
        return sorted(self._load().values(), key=lambda e: e.get("ts", 0))

    def keys(self) -> set[str]:
        return set(self._load().keys())

    def __len__(self) -> int:
        return len(self._load())

    def record(
        self,
        key: str,
        scene_id: str | None,
        path: str | None,
        *,
        error: str,
        mode: str = "",
        hwaccel: str = "",
        pipeline: str = "",
        model: str = "",
    ) -> None:
        with self._lock:
            data = self._load()
            data[key] = {
                "key": key,
                "scene_id": scene_id,
                "path": path,
                "error": str(error)[:500],
                "mode": mode,
                "hwaccel": hwaccel,
                "pipeline": pipeline,
                "model": model,
                "ts": time.time(),
            }
            self._write(data)

    def resolve(self, key: str) -> bool:
        """Drop an entry once its scene embeds successfully. Returns True if
        there was one to drop."""
        with self._lock:
            data = self._load()
            if key in data:
                del data[key]
                self._write(data)
                return True
        return False

    def clear(self) -> int:
        with self._lock:
            n = len(self._load())
            self._write({})
        return n
