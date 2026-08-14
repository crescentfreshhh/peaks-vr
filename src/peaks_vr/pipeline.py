"""Orchestration: the two passes that tie the pieces together.

  embed_library  — Stash scenes → sampled frames → embeddings → on-disk cache.
                   The GPU-heavy, resumable, one-time pass.

  score_library  — cached embeddings → similarity vs your references → segments
                   → Stash `apex` markers (or a dry-run preview).

These need ffmpeg + the `[ml]` extra + a live Stash to actually run; the
building blocks they call (sampler, embedder, cache, scorer) are unit-tested
offline. Kept deliberately thin so the logic lives in the tested modules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .cache import EmbeddingCache, path_key
from .embedding import Embedder
from .models import Scene
from .sampling import FrameSampler
from .scoring import Segment, extract_segments, normalize_scores, smooth

Logger = Callable[[str], None]

# A frame scorer maps (n, dim) embeddings -> (n,) per-frame scores. Both the
# Tier-1 similarity closure and a trained classifier's predict_proba fit this.
ScoreFn = Callable[[np.ndarray], np.ndarray]


def scene_key(scene: Scene) -> str:
    """Cache key for a scene: prefer the file fingerprint, else hash the path."""
    return scene.fingerprint or path_key(scene.path or scene.id)


def safe_tag(tag: str) -> str:
    """Filesystem-safe name for a tag: 'apex:heels' -> 'apex_heels'. Used for
    model filenames and per-profile reference folders (SMB-safe on Windows,
    where ':' is forbidden). Hyphenated tags like 'apex-heels' pass through
    unchanged, so they map 1:1 to folder names."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def reference_files(references_dir: str | Path) -> list[Path]:
    """Top-level images in a references dir. Deliberately NOT recursive:
    subfolders hold other taste profiles' references and must not bleed in."""
    return sorted(
        p
        for p in Path(references_dir).glob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def resolve_references_dir(base: str | Path, tag: str) -> Path:
    """Per-profile references: use `<base>/<safe_tag>/` when it exists, else
    the base dir itself. Lets each taste profile keep its own example stills
    (references/apex-heels/...) with zero flags."""
    sub = Path(base) / safe_tag(tag)
    return sub if sub.is_dir() else Path(base)


def embed_library(
    scenes: Iterable[Scene],
    sampler: FrameSampler,
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    batch_size: int = 64,
    total: int | None = None,
    workers: int = 1,
    log: Logger = print,
    failure_log=None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Embed every scene not already cached. Resumable + idempotent.

    Frames are embedded in rolling batches as they stream off ffmpeg, so at
    most `batch_size` decoded frames are held in memory — not the whole scene.
    Cache hits are only honoured when built with the same sampling settings
    (interval, or keyframe mode); a config change re-embeds instead of
    silently serving stale samples. Pass `total` for progress/ETA in the log.

    `workers` > 1 decodes that many scenes concurrently (raw path only) while
    the main thread embeds — sparse sampling is seek/I-O-latency-bound, so
    parallel decode is a near-linear speedup until the disks saturate.
    """
    import time as _time

    signature = getattr(sampler, "interval_signature", sampler.interval)
    # raw path: numpy frames straight to the GPU — no JPEG/PIL round-trip.
    # Samplers advertise it via wants_raw (sparse mode is always raw).
    use_raw = getattr(sampler, "wants_raw", None)
    if use_raw is None:  # older/stub samplers: infer from attrs
        use_raw = (
            getattr(sampler, "pipeline", "jpeg") == "raw"
            and getattr(sampler, "mode", "interval") == "interval"
        )

    pipeline_name = "raw" if use_raw else "jpeg"

    def _record_failure(scene: Scene, key: str, exc) -> None:
        if failure_log is not None:
            failure_log.record(
                key, scene.id, scene.path,
                error=str(exc),
                mode=getattr(sampler, "mode", "interval"),
                hwaccel=getattr(sampler, "hwaccel", ""),
                pipeline=pipeline_name,
                model=embedder.name,
            )

    if workers > 1 and use_raw:
        return _embed_library_parallel(
            scenes, sampler, embedder, cache,
            signature=signature, batch_size=batch_size, total=total,
            workers=workers, log=log, failure_log=failure_log,
            should_stop=should_stop,
        )

    stats = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
    embed_seconds = 0.0
    processed = 0
    for scene in scenes:
        if should_stop and should_stop():
            log("  ⏹ stop requested — halting after "
                f"{stats['embedded']} embedded")
            break
        processed += 1
        key = scene_key(scene)
        if cache.has(key, embedder.name, interval=signature):
            stats["skipped"] += 1
            continue
        if not scene.path:
            log(f"  ! scene {scene.id} has no file; skipping")
            stats["failed"] += 1
            continue
        started = _time.monotonic()
        try:
            times: list[float] = []
            batch: list = []
            chunks: list[np.ndarray] = []
            if use_raw:
                frame_iter = sampler.iter_frames_raw(
                    scene.path,
                    resize_short=embedder.raw_resize,
                    crop=embedder.raw_crop,
                )
                embed_batch = lambda b: embedder.embed_array(np.stack(b))  # noqa: E731
            else:
                frame_iter = sampler.iter_frames(scene.path)
                embed_batch = embedder.embed_images
            for ts, img in frame_iter:
                times.append(ts)
                batch.append(img)
                if len(batch) >= batch_size:
                    chunks.append(embed_batch(batch))
                    batch = []
            if batch:
                chunks.append(embed_batch(batch))
            vecs = (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.zeros((0, embedder.dim), dtype=np.float32)
            )
            cache.save(
                key,
                embedder.name,
                np.asarray(times, dtype=np.float32),
                vecs,
                meta={
                    "scene_id": scene.id,
                    "path": scene.path,
                    "interval": signature,
                    "mode": getattr(sampler, "mode", "interval"),
                    "pipeline": "raw" if use_raw else "jpeg",
                    "model": embedder.name,
                    "dim": embedder.dim,
                    "n_frames": len(times),
                },
            )
            dt = _time.monotonic() - started
            embed_seconds += dt
            stats["embedded"] += 1
            stats["frames"] += len(times)
            if failure_log is not None:
                failure_log.resolve(key)  # a prior casualty that now succeeds
            progress = f"[{processed}/{total}] " if total else ""
            eta = ""
            if total and stats["embedded"]:
                rate = embed_seconds / stats["embedded"]
                eta = f", eta ~{rate * (total - processed) / 3600:.1f}h"
            log(
                f"  + {progress}scene {scene.id}: {len(times)} frames "
                f"in {dt:.1f}s{eta}"
            )
        except Exception as exc:  # keep the batch going; log the casualty
            log(f"  ! scene {scene.id} failed: {exc}")
            stats["failed"] += 1
            _record_failure(scene, key, exc)
    if stats["embedded"]:
        stats["seconds_per_scene"] = round(embed_seconds / stats["embedded"], 2)
    return stats


def _embed_library_parallel(
    scenes: Iterable[Scene],
    sampler: FrameSampler,
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    signature: float,
    batch_size: int,
    total: int | None,
    workers: int,
    log: Logger,
    failure_log=None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Raw-path embed with `workers` scenes decoding concurrently.

    Worker threads run the (I/O-latency-bound) seek+decode; the main thread
    owns the GPU: it embeds and caches each finished scene. In-flight decode
    is bounded to 2x workers so memory stays modest (a sparse scene's frames
    are only tens of MB).
    """
    import time as _time
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    stats = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
    started_all = _time.monotonic()
    processed = 0

    def _decode(scene: Scene):
        times: list[float] = []
        frames: list[np.ndarray] = []
        for ts, arr in sampler.iter_frames_raw(
            scene.path, resize_short=embedder.raw_resize, crop=embedder.raw_crop
        ):
            times.append(ts)
            frames.append(arr)
        stacked = (
            np.stack(frames)
            if frames
            else np.zeros(
                (0, embedder.raw_crop, embedder.raw_crop, 3), dtype=np.uint8
            )
        )
        return np.asarray(times, dtype=np.float32), stacked

    def _finish(entry) -> None:
        nonlocal processed
        scene, key, future = entry
        processed += 1
        try:
            times, frames = future.result()
            chunks = [
                embedder.embed_array(frames[i : i + batch_size])
                for i in range(0, len(frames), batch_size)
            ]
            vecs = (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.zeros((0, embedder.dim), dtype=np.float32)
            )
            cache.save(
                key,
                embedder.name,
                times,
                vecs,
                meta={
                    "scene_id": scene.id,
                    "path": scene.path,
                    "interval": signature,
                    "mode": getattr(sampler, "mode", "interval"),
                    "pipeline": "raw",
                    "model": embedder.name,
                    "dim": embedder.dim,
                    "n_frames": len(times),
                },
            )
            stats["embedded"] += 1
            stats["frames"] += len(times)
            if failure_log is not None:
                failure_log.resolve(key)
            elapsed = _time.monotonic() - started_all
            rate = elapsed / stats["embedded"]
            progress = f"[{processed}/{total}] " if total else ""
            eta = (
                f", eta ~{rate * (total - processed) / 3600:.1f}h" if total else ""
            )
            log(
                f"  + {progress}scene {scene.id}: {len(times)} frames "
                f"({rate:.1f}s/scene overall{eta})"
            )
        except Exception as exc:
            log(f"  ! scene {scene.id} failed: {exc}")
            stats["failed"] += 1
            if failure_log is not None:
                failure_log.record(
                    key, scene.id, scene.path,
                    error=str(exc),
                    mode=getattr(sampler, "mode", "interval"),
                    hwaccel=getattr(sampler, "hwaccel", ""),
                    pipeline="raw",
                    model=embedder.name,
                )

    inflight: deque = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for scene in scenes:
            if should_stop and should_stop():
                log(f"  ⏹ stop requested — halting after {stats['embedded']} embedded")
                break
            key = scene_key(scene)
            if cache.has(key, embedder.name, interval=signature):
                processed += 1
                stats["skipped"] += 1
                continue
            if not scene.path:
                processed += 1
                log(f"  ! scene {scene.id} has no file; skipping")
                stats["failed"] += 1
                continue
            inflight.append((scene, key, pool.submit(_decode, scene)))
            if len(inflight) >= workers * 2:  # bound decoded-but-unembedded RAM
                _finish(inflight.popleft())
        while inflight:
            _finish(inflight.popleft())

    if stats["embedded"]:
        elapsed = _time.monotonic() - started_all
        stats["seconds_per_scene"] = round(elapsed / stats["embedded"], 2)
    return stats


def sync_cache(
    scenes: Iterable[Scene],
    cache: EmbeddingCache,
    model_name: str,
    *,
    prune: bool = False,
    log: Logger = print,
) -> dict:
    """Reconcile one model's cache against the current Stash library.

    The cache is keyed by a stable file fingerprint (oshash), so a scene that
    MOVES between folders keeps its key — only the stored ``path``/``scene_id``
    go stale. This refreshes those in place (no re-embedding). A scene that was
    DELETED from Stash no longer appears in `scenes`; its cache entry is an
    orphan and, when ``prune`` is set, its ``.npz`` is removed.

    `scenes` must be the set the cache should be reconciled against — pass the
    WHOLE library (unscoped) so a scene that merely moved out of the embed
    scope isn't mistaken for a deletion. Returns counts; when ``prune`` is
    False the orphans are only reported (a dry run for the destructive half).
    """
    current: dict[str, Scene] = {}
    for scene in scenes:
        current[scene_key(scene)] = scene

    stats = {"cached": 0, "moved": 0, "orphaned": 0, "pruned": 0}
    for key in cache.keys(model_name):
        try:
            times, vecs, meta = cache.load(key, model_name)
        except Exception:
            continue  # unreadable entry: leave it for a human
        stats["cached"] += 1
        scene = current.get(key)
        if scene is None:
            stats["orphaned"] += 1
            if prune:
                cache.delete(key, model_name)
                stats["pruned"] += 1
                log(f"  - pruned {key} ({meta.get('path') or 'unknown path'})")
            else:
                log(f"  ? orphan {key} ({meta.get('path') or 'unknown path'})")
            continue
        new_path = scene.path or meta.get("path")
        path_changed = new_path and new_path != meta.get("path")
        id_changed = str(scene.id) != str(meta.get("scene_id"))
        if path_changed or id_changed:
            meta["path"] = new_path
            meta["scene_id"] = scene.id
            cache.save(key, model_name, times, vecs, meta=meta)
            stats["moved"] += 1
            if path_changed:
                log(f"  ~ moved {key}: -> {new_path}")
    return stats


def load_references(embedder: Embedder, references_dir: str | Path) -> np.ndarray:
    """Embed the top-level images in a directory into reference vectors
    (m, dim). Subfolders are other profiles' references and are ignored —
    resolve the right folder first with `resolve_references_dir`."""
    from PIL import Image as PILImage  # lazy

    files = reference_files(references_dir)
    if not files:
        raise FileNotFoundError(f"no reference images found in {references_dir}")
    images = [PILImage.open(p).convert("RGB") for p in files]
    return embedder.embed_images(images)


def score_scene(
    times: np.ndarray,
    vecs: np.ndarray,
    score_frames: ScoreFn,
    scoring,
) -> list[Segment]:
    """Pure scoring for one scene's cached embeddings (no I/O).

    `score_frames` is the Tier-agnostic scorer: similarity closure or a trained
    classifier's predict_proba.
    """
    scores = normalize_scores(score_frames(vecs), getattr(scoring, "normalize", "none"))
    scores = smooth(scores, scoring.smooth_window)
    return extract_segments(
        scores,
        times,
        high=scoring.high,
        low=scoring.low,
        min_duration=scoring.min_duration,
        merge_gap=scoring.merge_gap,
        max_duration=scoring.max_duration or None,
        pad=scoring.pad,
    )


def score_library(
    scenes: Iterable[Scene],
    cache: EmbeddingCache,
    embedder_name: str,
    score_frames: ScoreFn,
    scoring,
    *,
    client=None,
    tag_name: str = "apex",
    write: bool = False,
    log: Logger = print,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Score cached scenes into segments; optionally write Stash markers.

    write=False is a dry run: it logs the segments it *would* create, perfect
    for tuning thresholds before touching Stash.

    Writes are idempotent: a segment that overlaps an existing marker carrying
    the same tag is skipped, so re-running `--write` after a threshold tweak
    adds new finds instead of duplicating everything.
    """
    stats = {"scenes": 0, "segments": 0, "skipped": 0, "existing": 0}
    tag = None
    if write:
        if client is None:
            raise ValueError("write=True requires a client")
        tag = client.find_or_create_tag(tag_name)

    def _already_marked(scene: Scene, start: float, end: float) -> bool:
        for m in scene.markers:
            if m.primary_tag is None or m.primary_tag.name != tag_name:
                continue
            m_end = m.end_seconds if m.end_seconds is not None else m.seconds
            if start <= m_end and end >= m.seconds:
                return True
        return False

    for scene in scenes:
        if should_stop and should_stop():
            log("  ⏹ stop requested — halting scoring")
            break
        key = scene_key(scene)
        if not cache.has(key, embedder_name):
            stats["skipped"] += 1
            continue
        times, vecs, _ = cache.load(key, embedder_name)
        segs = score_scene(times, vecs, score_frames, scoring)
        stats["scenes"] += 1
        stats["segments"] += len(segs)
        for s in segs:
            if write:
                if _already_marked(scene, s.start, s.end):
                    stats["existing"] += 1
                    continue
                client.create_scene_marker(
                    scene_id=scene.id,
                    seconds=s.start,
                    primary_tag_id=tag.id,
                    # peak score rides in the title: visible in Stash, and the
                    # playlist parses it back to weight the megaboard picker
                    title=f"{tag_name} {s.peak_score:.3f}",
                    end_seconds=s.end,
                )
            else:
                log(
                    f"  ~ scene {scene.id}: {s.start:7.1f}-{s.end:7.1f}s "
                    f"peak={s.peak_score:.3f}"
                )
    return stats


# --- Tier 2: training-set assembly + candidate gathering --------------------


@dataclass
class Candidate:
    """A frame to be labeled: where it is + the current model's score for it."""

    key: str
    scene_id: str | None
    path: str | None
    time: float
    score: float


def build_training_set(
    label_store, cache: EmbeddingCache, model_name: str, profile: str,
    with_recency: bool = False,
):
    """Assemble (X, y) from labels: each label's nearest cached frame vector.

    Loads each scene's cache once. Skips labels whose scene isn't cached. With
    `with_recency`, also returns a per-row `ts` (when each rating was made), so
    training can weight recent ratings more heavily.
    """
    by_key: dict[str, list] = defaultdict(list)
    for lab in label_store.for_profile(profile):
        by_key[lab.key].append(lab)

    rows, ys, tss = [], [], []
    for key, labs in by_key.items():
        if not cache.has(key, model_name):
            continue
        times, vecs, _ = cache.load(key, model_name)
        if len(times) == 0:
            continue
        for lab in labs:
            idx = int(np.argmin(np.abs(times - lab.time)))
            rows.append(vecs[idx])
            ys.append(lab.label)
            tss.append(getattr(lab, "ts", 0.0) or 0.0)
    if not rows:
        empty = (np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=int))
        return (*empty, np.zeros((0,), dtype=np.float64)) if with_recency else empty
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(ys, dtype=int)
    if with_recency:
        return X, y, np.asarray(tss, dtype=np.float64)
    return X, y


def recency_weights(ts: np.ndarray, halflife_days: float) -> np.ndarray | None:
    """Per-sample weights that decay with age: a rating `halflife_days` old
    counts half as much as a fresh one, so the model tracks your *current*
    taste instead of averaging your whole history. Ratings with an unknown
    time (ts=0, pre-dating the field) are treated as the oldest known rating,
    and a floor keeps old history from vanishing entirely. Returns None when
    recency is off or no timestamps exist (→ uniform weighting)."""
    ts = np.asarray(ts, dtype=np.float64)
    known = ts[ts > 0]
    if halflife_days <= 0 or known.size == 0:
        return None
    from time import time as _now

    now = max(float(known.max()), _now())
    eff = np.where(ts > 0, ts, float(known.min()))
    age_days = np.clip((now - eff) / 86400.0, 0.0, None)
    w = 0.5 ** (age_days / float(halflife_days))
    return np.maximum(w, 0.05)  # old history still counts a little


def gather_candidates(
    cache: EmbeddingCache,
    model_name: str,
    score_frames: ScoreFn,
    *,
    top_per_scene: int = 3,
    random_per_scene: int = 1,
    seed: int = 0,
    limit: int | None = None,
    exclude: set[tuple[str, float]] | None = None,
) -> list[Candidate]:
    """Propose frames to label: each scene's highest-scoring frames (active
    learning) plus a few random ones (for diverse negatives).

    The two pools survive `limit` *separately* — randoms are reserved their
    proportional share rather than being sorted to the bottom and truncated,
    otherwise the classifier never sees the diverse negatives it needs.
    `exclude` skips frames already labeled: {(key, round(time, 2)), ...}.
    The result is shuffled (deterministically by `seed`) so the rater isn't
    biased by a strictly descending score order.
    """
    rng = np.random.default_rng(seed)
    exclude = exclude or set()
    top_pool: list[Candidate] = []
    rand_pool: list[Candidate] = []
    for key in cache.keys(model_name):
        times, vecs, meta = cache.load(key, model_name)
        n = len(times)
        if n == 0:
            continue
        scores = score_frames(vecs)

        def _mk(idx: int) -> Candidate:
            return Candidate(
                key=key,
                scene_id=meta.get("scene_id"),
                path=meta.get("path"),
                time=float(times[idx]),
                score=float(scores[idx]),
            )

        def _fresh(idx: int) -> bool:
            return (key, round(float(times[idx]), 2)) not in exclude

        top_idx = [int(i) for i in np.argsort(scores)[::-1][:top_per_scene] if _fresh(i)]
        top_pool.extend(_mk(i) for i in top_idx)
        remaining = [i for i in range(n) if i not in set(top_idx) and _fresh(i)]
        if remaining and random_per_scene:
            pick = rng.choice(
                remaining, size=min(random_per_scene, len(remaining)), replace=False
            )
            rand_pool.extend(_mk(int(i)) for i in pick)

    if limit and len(top_pool) + len(rand_pool) > limit:
        # reserve randoms their configured share of the budget
        frac = random_per_scene / max(1, top_per_scene + random_per_scene)
        n_rand = min(len(rand_pool), max(1, round(limit * frac)))
        n_top = limit - n_rand
        top_pool.sort(key=lambda c: c.score, reverse=True)
        rng.shuffle(rand_pool)
        top_pool, rand_pool = top_pool[:n_top], rand_pool[:n_rand]

    cands = top_pool + rand_pool
    rng.shuffle(cands)
    return cands


# "auto" taste classifier switches to the non-linear MLP once there's enough
# data for it not to overfit; below this it stays on the robust logreg.
AUTO_MLP_MIN_SAMPLES = 200


def train_profile(
    label_store, cache: EmbeddingCache, model_name: str, profile: str,
    kind: str = "logreg", recency_halflife_days: float = 0.0,
):
    """Build the training set and fit a TasteClassifier for `profile`.

    `kind="auto"` picks logreg on small data (robust) and the non-linear MLP
    once there's enough (captures multi-modal taste). `recency_halflife_days`>0
    weights recent ratings more, so the model evolves with you. When there are
    enough labels, stats include `cv_auc`: mean ROC-AUC over stratified CV folds
    — a quick "is this model any good" signal (1.0 = perfect, 0.5 = coin flip).
    """
    from .classifier import TasteClassifier

    X, y, ts = build_training_set(
        label_store, cache, model_name, profile, with_recency=True
    )
    resolved = kind
    if kind == "auto":
        resolved = "mlp" if int(X.shape[0]) >= AUTO_MLP_MIN_SAMPLES else "logreg"
    sample_weight = recency_weights(ts, recency_halflife_days)

    clf = TasteClassifier(kind=resolved, model_name=model_name, profile=profile)
    clf.train(X, y, sample_weight=sample_weight)
    stats = {
        "samples": int(X.shape[0]), "positives": int((y == 1).sum()),
        "kind": resolved, "recency": sample_weight is not None,
    }

    # CV quality signal (logreg only — cheap, and MLP CV is slow/unweighted)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    folds = min(5, n_pos, n_neg)
    if resolved == "logreg" and folds >= 2:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            clf._new_estimator(), X, y, cv=folds, scoring="roc_auc"
        )
        stats["cv_auc"] = round(float(scores.mean()), 3)
        stats["cv_folds"] = folds
    return clf, stats
