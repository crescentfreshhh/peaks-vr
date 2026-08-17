"""Per-file VR-format overrides — the sticky corrections made during QC.

Auto-detection (:mod:`peaks_vr.vr_format`) is usually right, but a file with a
missing or *wrong* filename hint gets mis-classified — and a misclassification
silently corrupts its embeddings (the de-warp splits the wrong eye or uses the
wrong projection). When that shows up in the QC contact sheet, the user forces
the correct format and re-embeds just that file.

This module persists those corrections so they **stick**: `_reprojector_for`
(web layer) consults the override before falling back to detection, so the fix
applies to the QC preview, the re-embed, and any future embed of the file — even
after a cache clear. It sits beside the embedding cache (same `/config` volume as
`failures.json`), and writes atomically like :class:`peaks_vr.failures.FailureLog`.

An entry is ``{path, format, fov, pitch, flat, ts}`` keyed by the file path:

- ``format`` — a format token parsed exactly like a filename
  (``"180_sbs"``, ``"mkx200_tb"``, ``"fisheye190_sbs"``, …); authoritative.
- ``flat`` — de-warp disabled for this file (embed the raw centre crop).
- ``fov`` / ``pitch`` — viewport aim, matching the reprojector knobs.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


def overrides_for(cache_root: str) -> "FormatOverrides":
    """The override store beside the cache (persists in the /config volume,
    alongside ``failures.json``)."""
    return FormatOverrides(Path(cache_root).parent / "overrides.json")


class FormatOverrides:
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
            return {}  # corrupt/half-written: treat as empty, don't crash a run

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def get(self, path: str) -> dict | None:
        """The override for ``path``, or None if the file uses auto-detection."""
        return self._load().get(path)

    def all(self) -> dict[str, dict]:
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    def set(
        self,
        path: str,
        *,
        format: str = "",
        fov: float = 100.0,
        pitch: float = 0.0,
        flat: bool = False,
    ) -> dict:
        """Store (or replace) the correction for ``path`` and return the entry."""
        entry = {
            "path": path,
            "format": format,
            "fov": float(fov),
            "pitch": float(pitch),
            "flat": bool(flat),
            "ts": time.time(),
        }
        with self._lock:
            data = self._load()
            data[path] = entry
            self._write(data)
        return entry

    def remove(self, path: str) -> bool:
        """Drop the override so ``path`` returns to auto-detection. Returns True
        if there was one to drop."""
        with self._lock:
            data = self._load()
            if path in data:
                del data[path]
                self._write(data)
                return True
        return False
