"""On-disk embedding cache — the thing that makes the GPU pass a one-time cost.

Layout:  <root>/<model_name>/<key>.npz
Each .npz holds:
    times : (n,) float32   sample timestamps in seconds
    vecs  : (n, d) float32  L2-normalized frame embeddings
    meta  : 0-d json string  (source path, interval, model, dim, ...)

`key` should be a stable file fingerprint (Stash exposes one via
VideoFile.fingerprints) so renaming/moving a file still hits cache. We fall
back to a hash of the path when no fingerprint is available.

`has()` is what makes the embedding pass resumable: skip any scene already
cached, so it can run in bursts and only embed newly-added videos.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def path_key(path: str) -> str:
    """Fallback cache key derived from the file path."""
    return "path-" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, root: Path | str, dtype: str = "float16"):
        """`dtype` is the on-disk storage type. float16 halves disk usage for
        a relative error ~1e-3 — irrelevant to cosine similarity or the
        classifier. `load` always returns float32 regardless of what was
        stored, so old float32 caches remain readable."""
        self.root = Path(root)
        self.dtype = np.dtype(dtype)

    def _file(self, key: str, model_name: str) -> Path:
        return self.root / model_name / f"{key}.npz"

    def has(self, key: str, model_name: str, *, interval: float | None = None) -> bool:
        """True if `key` is cached — and, when `interval` is given, was built
        at that sampling interval. A mismatch (or missing meta) returns False
        so the caller re-embeds instead of trusting stale-density samples."""
        f = self._file(key, model_name)
        if not f.exists():
            return False
        if interval is None:
            return True
        try:
            _, _, meta = self.load(key, model_name)
        except Exception:
            return False  # unreadable cache entry: treat as absent
        cached = meta.get("interval")
        return cached is not None and abs(float(cached) - interval) < 1e-6

    def save(
        self,
        key: str,
        model_name: str,
        times: np.ndarray,
        vecs: np.ndarray,
        meta: dict | None = None,
    ) -> Path:
        times = np.asarray(times, dtype=np.float32)
        vecs = np.asarray(vecs, dtype=self.dtype)
        if times.shape[0] != vecs.shape[0]:
            raise ValueError("times and vecs must have matching first dimension")
        dest = self._file(key, model_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # atomic-ish: write to a temp file then replace. Pass an open handle so
        # numpy doesn't "helpfully" append .npz to the temp filename.
        tmp = dest.with_name(dest.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                times=times,
                vecs=vecs,
                meta=np.array(json.dumps(meta or {})),
            )
        tmp.replace(dest)
        return dest

    def load(self, key: str, model_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
        with np.load(self._file(key, model_name), allow_pickle=False) as data:
            times = data["times"]
            vecs = data["vecs"].astype(np.float32)  # regardless of storage dtype
            meta = json.loads(str(data["meta"]))
        return times, vecs, meta

    def peek_count(self, key: str, model_name: str) -> int:
        """Frame count for an entry without loading its (large) vectors — reads
        only the small `times` array. Lets the search index size its matrix up
        front and fill it in place, instead of stacking per-scene arrays and
        paying a transient second full copy at np.concatenate time."""
        with np.load(self._file(key, model_name), allow_pickle=False) as data:
            return int(data["times"].shape[0])

    def keys(self, model_name: str) -> list[str]:
        model_dir = self.root / model_name
        if not model_dir.exists():
            return []
        return sorted(p.stem for p in model_dir.glob("*.npz"))

    def models(self) -> list[str]:
        """Names of every model that has a cache directory (dinov2, clip, ...)."""
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def delete(self, key: str, model_name: str) -> bool:
        """Remove a cached scene entry. Returns True if a file was deleted.
        Used by the sync/prune pass when a scene leaves the library."""
        f = self._file(key, model_name)
        if f.exists():
            f.unlink()
            return True
        return False
