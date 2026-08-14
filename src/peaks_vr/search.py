"""In-memory nearest-neighbour search over the embedding cache.

The cache is a per-scene set of L2-normalized frame vectors. This module
stacks them into one matrix so any query vector (a frame you pick, an uploaded
image, or a CLIP text prompt) can be matched against every moment in the
library by cosine similarity — the engine behind "find more like this" and
text search.

Brute-force numpy is plenty for libraries up to a few million frames (a query
is one matmul, single-digit milliseconds). A faiss backend can slot in later
behind the same interface if the library outgrows RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import EmbeddingCache


@dataclass
class Hit:
    scene_id: str | None
    key: str
    time: float
    score: float


class SearchIndex:
    def __init__(self, cache: EmbeddingCache, model_name: str):
        self.cache = cache
        self.model_name = model_name
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.keys: list[str] = []
        self.scene_ids: list[str | None] = []
        self.times: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._key_rows: dict[str, tuple[int, int]] = {}  # key -> (start, end)
        self.key_meta: dict[str, dict] = {}  # key -> {scene_id, path}

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    @property
    def dim(self) -> int:
        return self.matrix.shape[1] if self.matrix.ndim == 2 else 0

    def build(self, keys: list[str] | None = None) -> "SearchIndex":
        """Load every cached scene for this model into one matrix.

        Vectors are L2-normalized at write time, so cosine similarity is just a
        dot product. Stored float32 for fast matmul (the cache is float16).

        The matrix is **preallocated** from a cheap first pass (frame counts via
        `peek_count`, no vectors) and filled in place, so building a
        whole-library index needs one copy of the data — not the ~2x transient a
        per-scene `mats` list plus `np.concatenate` would hold. For a large
        library (millions of frames) that halves the peak RAM of a rebuild.
        """
        keys = keys if keys is not None else self.cache.keys(self.model_name)

        # pass 1: cheap frame counts (reads only the tiny `times` arrays) to
        # size the matrix without a transient second copy.
        kept: list[str] = []
        counts: list[int] = []
        for key in keys:
            try:
                n = self.cache.peek_count(key, self.model_name)
            except Exception:
                continue
            if n > 0:
                kept.append(key)
                counts.append(n)

        total = sum(counts)
        if total == 0:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            self.times = np.zeros((0,), dtype=np.float32)
            self.keys = []
            self.scene_ids = []
            return self

        # dim from the first kept scene, then fill the preallocated matrix.
        _, first_vecs, _ = self.cache.load(kept[0], self.model_name)
        dim = int(first_vecs.shape[1])
        matrix = np.empty((total, dim), dtype=np.float32)
        times_all = np.empty((total,), dtype=np.float32)
        all_keys: list[str] = []
        all_scenes: list[str | None] = []
        pos = 0
        for key in kept:
            try:
                times, vecs, meta = self.cache.load(key, self.model_name)
            except Exception:
                continue
            n = vecs.shape[0]
            if n == 0:
                continue
            if pos + n > total:  # count grew between passes (mid-embed write)
                n = total - pos
                vecs, times = vecs[:n], times[:n]
            matrix[pos : pos + n] = vecs  # cache.load already returns float32
            times_all[pos : pos + n] = times.astype(np.float32)
            self._key_rows[key] = (pos, pos + n)
            all_keys.extend([key] * n)
            sid = meta.get("scene_id")
            all_scenes.extend([sid] * n)
            self.key_meta[key] = {"scene_id": sid, "path": meta.get("path")}
            pos += n
            if pos >= total:
                break

        # trim if the live cache shrank between passes (any count drift)
        self.matrix = matrix[:pos]
        self.times = times_all[:pos]
        self.keys = all_keys
        self.scene_ids = all_scenes
        return self

    # --- querying ------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int | None = 60,
        *,
        per_scene: int | None = None,
        exclude_key: str | None = None,
        min_score: float | None = None,
    ) -> list[Hit]:
        """Moments most similar to `query` (a 1D vector or (1,d)).

        `top_k` caps how many hits come back; pass `None` for *unbounded* (every
        qualifying moment). `min_score` returns **all** frames whose cosine is at
        least that — the "match strength" floor for Explore — instead of a fixed
        top slice. `per_scene` caps how many hits any single scene contributes, so
        results span the library instead of clustering in one video. `exclude_key`
        drops a scene from the results (e.g. the one you searched *from*).
        """
        if self.size == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.matrix @ q  # cosine (rows are unit vectors)

        if min_score is not None:
            # every qualifying frame, ordered best-first (all scores already computed)
            idx = np.where(scores >= min_score)[0]
            idx = idx[np.argsort(-scores[idx])]
        else:
            # bounded top-k: partition a generous pool, then order just that slice
            pool = min(self.size, max((top_k or 60) * 8, (top_k or 60) + 50))
            idx = np.argpartition(-scores, pool - 1)[:pool]
            idx = idx[np.argsort(-scores[idx])]

        hits: list[Hit] = []
        per_scene_counts: dict[str | None, int] = {}
        for i in idx:
            key = self.keys[i]
            if exclude_key is not None and key == exclude_key:
                continue
            sid = self.scene_ids[i]
            if per_scene is not None:
                c = per_scene_counts.get(sid, 0)
                if c >= per_scene:
                    continue
                per_scene_counts[sid] = c + 1
            hits.append(Hit(scene_id=sid, key=key, time=float(self.times[i]), score=float(scores[i])))
            if top_k is not None and len(hits) >= top_k:
                break
        return hits

    def vector_at(self, key: str, time: float) -> np.ndarray | None:
        """The stored vector for the frame nearest `time` in scene `key`."""
        rows = self._key_rows.get(key)
        if rows is None:
            return None
        start, end = rows
        seg_times = self.times[start:end]
        if seg_times.size == 0:
            return None
        i = start + int(np.argmin(np.abs(seg_times - time)))
        return self.matrix[i]

    def search_by_frame(
        self, key: str, time: float, top_k: int | None = 60, *,
        per_scene: int | None = 3, min_score: float | None = None,
    ) -> list[Hit]:
        """Find moments like the frame at (key, time). Excludes its own scene."""
        v = self.vector_at(key, time)
        if v is None:
            return []
        return self.search(
            v, top_k=top_k, per_scene=per_scene, exclude_key=key, min_score=min_score
        )
