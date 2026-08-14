"""Configuration loading.

Resolution order (highest priority first):
    1. Environment variables (STASH_URL, STASH_API_KEY, PEAKS_DEVICE,
       PEAKS_HWACCEL, PEAKS_LIBRARY_PATH, PEAKS_SAMPLING_MODE,
       PEAKS_INTERVAL_SECONDS)
    2. A TOML file (default: ./config.toml)
    3. Built-in defaults

The TOML file is gitignored so your API key never gets committed.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


def _as_bool(env_val, default: bool) -> bool:
    """Coerce an env string ("1", "true", "yes", "on") to bool; fall back to
    `default` (from TOML or the dataclass) when the env var is unset."""
    if env_val is None:
        return bool(default)
    return str(env_val).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class StashConfig:
    url: str = "http://192.168.1.2:6969"
    api_key: str = ""
    timeout: int = 30


@dataclass
class SamplingConfig:
    interval_seconds: float = 8.0
    # "sparse": seek + decode one keyframe per sample — cost scales with
    #   samples, not duration. The right choice for long videos.
    # "interval": full decode, one frame kept per interval (short clips).
    # "keyframes": full-file keyframe pass (legacy fallback).
    mode: str = "sparse"
    hwaccel: str = "cuda"  # "" | "cuda" | "auto" — GPU (NVDEC) decode, default on
    pipeline: str = "raw"  # "raw" (fast: frames straight to GPU) | "jpeg"
    scene_timeout: float = 180.0  # per-scene sampling ceiling (s); 0 disables


@dataclass
class MarkersConfig:
    tag_name: str = "apex"


@dataclass
class EmbeddingConfig:
    model: str = "dino"  # "dino" | "clip" | "fake"
    cache_dir: str = "cache/embeddings"
    device: str = ""  # "" = auto (cuda if available)
    batch_size: int = 64
    # which OpenCLIP variant powers the "clip" channel. ViT-L-14 is a strong
    # default (768-dim, sharper search + tags) at ~2-3x the cost of ViT-B-32.
    # Changing it invalidates the CLIP cache (different space/dim) — the cache
    # is namespaced by variant, so re-embed CLIP after switching.
    # (Env: PEAKS_CLIP_MODEL, PEAKS_CLIP_PRETRAINED)
    clip_model: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"
    # DINOv2 backbone: "dinov2_vitl14" (large, default) or "dinov2_vitg14"
    # (giant). Bigger = more accurate structure/"find similar" at more cost.
    # Cache is namespaced per backbone, so switching means re-embedding DINO.
    # (Env: PEAKS_DINO_MODEL)
    dino_model: str = "dinov2_vitl14"
    # concurrent scene decodes (raw path). Sparse sampling is I/O-latency
    # bound, so 3-4 workers is a big speedup. (Env override: PEAKS_WORKERS)
    workers: int = 3


@dataclass
class ScoringConfig:
    reduce: str = "max"  # "max" | "mean"
    normalize: str = "none"  # "none" | "scene-z" (thresholds become std-devs)
    smooth_window: int = 3
    high: float = 0.45  # tune after Tier-1 validation
    low: float = 0.35
    min_duration: float = 3.0
    merge_gap: float = 2.0
    max_duration: float = 30.0
    pad: float = 0.5
    references_dir: str = "references"


@dataclass
class ModelingConfig:
    dir: str = "models"  # where trained classifiers are saved (gitignored)
    classifier: str = "logreg"  # "logreg" | "mlp"
    labels_path: str = "labels.json"
    # Auto-retrain the swipe-trainer's taste model in the background after this
    # many new ratings, so "Train now" is optional. 0 disables (manual only).
    # (Env override: PEAKS_AUTOTRAIN_EVERY)
    autotrain_every: int = 25
    # Taste model kind: "auto" (logreg on small data, non-linear MLP once there's
    # enough — captures multi-modal taste), or force "logreg" / "mlp". Applies to
    # the For You taste model only, not scoring. (Env: PEAKS_TASTE_CLASSIFIER)
    taste_classifier: str = "auto"
    # Weight recent ratings more when training taste: a rating this many days old
    # counts half as much, so your taste evolves with you. 0 = off (uniform).
    # (Env override: PEAKS_RECENCY_HALFLIFE_DAYS)
    recency_halflife_days: float = 90.0
    # How much the For You feed trades relevance for variety (0 = pure best-match,
    # 1 = maximally diverse). A little keeps it from collapsing into a bubble.
    # (Env override: PEAKS_FEED_DIVERSITY)
    feed_diversity: float = 0.3
    # Untrained taste is modelled as up to this many "modes" (k-means clusters of
    # your loved moments) rather than one average, so the board scores a moment by
    # its nearest mode and spans your distinct interests. (Env: PEAKS_TASTE_MODES)
    taste_modes: int = 24


@dataclass
class ScheduleConfig:
    # Recurring incremental embed (web app scheduler). 0 = off. Handy for
    # picking up newly-added scenes on CPU after the GPU goes back to the VM.
    embed_hours: float = 0.0

    # After each recurring embed pass, reconcile the cache with Stash: refresh
    # moved scenes' stored paths and (if prune) drop entries for scenes deleted
    # from Stash. Moves are always refreshed; pruning is destructive so it is
    # opt-in here. The manual `peaks sync` / GUI button prunes regardless.
    sync: bool = True
    prune: bool = False

    @property
    def embed_seconds(self) -> float:
        return self.embed_hours * 3600.0


@dataclass
class AuthConfig:
    # Password-gate the whole web app (UI + API + megaboard). Empty = no auth
    # (open, the historical behaviour). Set a password to require a login.
    # (Env: PEAKS_PASSWORD) — keep it out of the repo; config.toml is gitignored.
    password: str = ""
    # A login lasts this long (sliding: refreshed on each request). (Env:
    # PEAKS_SESSION_HOURS)
    session_hours: float = 1.0


@dataclass
class LibraryConfig:
    # Only work on scenes whose file path starts with this (empty = whole
    # library). Point it at a folder to exclude everything else — e.g. skip VR.
    path: str = ""


@dataclass
class Config:
    stash: StashConfig = field(default_factory=StashConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Load config from a TOML file (if present) then apply env overrides."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict = {}
        if path.exists():
            with path.open("rb") as fh:
                raw = tomllib.load(fh)

        stash_raw = raw.get("stash", {})
        sampling_raw = raw.get("sampling", {})
        markers_raw = raw.get("markers", {})
        embedding_raw = raw.get("embedding", {})
        scoring_raw = raw.get("scoring", {})
        modeling_raw = raw.get("modeling", {})
        library_raw = raw.get("library", {})

        stash = StashConfig(
            url=os.environ.get("STASH_URL", stash_raw.get("url", StashConfig.url)),
            api_key=os.environ.get(
                "STASH_API_KEY", stash_raw.get("api_key", StashConfig.api_key)
            ),
            timeout=int(stash_raw.get("timeout", StashConfig.timeout)),
        )
        sampling = SamplingConfig(
            interval_seconds=float(
                os.environ.get(
                    "PEAKS_INTERVAL_SECONDS",
                    sampling_raw.get(
                        "interval_seconds", SamplingConfig.interval_seconds
                    ),
                )
            ),
            mode=os.environ.get(
                "PEAKS_SAMPLING_MODE", sampling_raw.get("mode", SamplingConfig.mode)
            ),
            hwaccel=os.environ.get(
                "PEAKS_HWACCEL", sampling_raw.get("hwaccel", SamplingConfig.hwaccel)
            ),
            pipeline=os.environ.get(
                "PEAKS_PIPELINE",
                sampling_raw.get("pipeline", SamplingConfig.pipeline),
            ),
            scene_timeout=float(
                os.environ.get(
                    "PEAKS_SCENE_TIMEOUT",
                    sampling_raw.get("scene_timeout", SamplingConfig.scene_timeout),
                )
            ),
        )
        markers = MarkersConfig(
            tag_name=markers_raw.get("tag_name", MarkersConfig.tag_name)
        )
        embedding = EmbeddingConfig(
            model=os.environ.get(
                "PEAKS_MODEL", embedding_raw.get("model", EmbeddingConfig.model)
            ),
            cache_dir=embedding_raw.get("cache_dir", EmbeddingConfig.cache_dir),
            device=os.environ.get(
                "PEAKS_DEVICE", embedding_raw.get("device", EmbeddingConfig.device)
            ),
            batch_size=int(embedding_raw.get("batch_size", EmbeddingConfig.batch_size)),
            workers=int(
                os.environ.get(
                    "PEAKS_WORKERS",
                    embedding_raw.get("workers", EmbeddingConfig.workers),
                )
            ),
            clip_model=os.environ.get(
                "PEAKS_CLIP_MODEL", embedding_raw.get("clip_model", EmbeddingConfig.clip_model)
            ),
            clip_pretrained=os.environ.get(
                "PEAKS_CLIP_PRETRAINED",
                embedding_raw.get("clip_pretrained", EmbeddingConfig.clip_pretrained),
            ),
            dino_model=os.environ.get(
                "PEAKS_DINO_MODEL", embedding_raw.get("dino_model", EmbeddingConfig.dino_model)
            ),
        )
        scoring = ScoringConfig(
            reduce=scoring_raw.get("reduce", ScoringConfig.reduce),
            normalize=scoring_raw.get("normalize", ScoringConfig.normalize),
            smooth_window=int(
                scoring_raw.get("smooth_window", ScoringConfig.smooth_window)
            ),
            high=float(scoring_raw.get("high", ScoringConfig.high)),
            low=float(scoring_raw.get("low", ScoringConfig.low)),
            min_duration=float(
                scoring_raw.get("min_duration", ScoringConfig.min_duration)
            ),
            merge_gap=float(scoring_raw.get("merge_gap", ScoringConfig.merge_gap)),
            max_duration=float(
                scoring_raw.get("max_duration", ScoringConfig.max_duration)
            ),
            pad=float(scoring_raw.get("pad", ScoringConfig.pad)),
            references_dir=scoring_raw.get(
                "references_dir", ScoringConfig.references_dir
            ),
        )
        modeling = ModelingConfig(
            dir=modeling_raw.get("dir", ModelingConfig.dir),
            classifier=modeling_raw.get("classifier", ModelingConfig.classifier),
            labels_path=modeling_raw.get("labels_path", ModelingConfig.labels_path),
            autotrain_every=int(
                os.environ.get(
                    "PEAKS_AUTOTRAIN_EVERY",
                    modeling_raw.get("autotrain_every", ModelingConfig.autotrain_every),
                )
            ),
            taste_classifier=os.environ.get(
                "PEAKS_TASTE_CLASSIFIER",
                modeling_raw.get("taste_classifier", ModelingConfig.taste_classifier),
            ),
            recency_halflife_days=float(
                os.environ.get(
                    "PEAKS_RECENCY_HALFLIFE_DAYS",
                    modeling_raw.get(
                        "recency_halflife_days", ModelingConfig.recency_halflife_days
                    ),
                )
            ),
            feed_diversity=float(
                os.environ.get(
                    "PEAKS_FEED_DIVERSITY",
                    modeling_raw.get("feed_diversity", ModelingConfig.feed_diversity),
                )
            ),
            taste_modes=int(
                os.environ.get(
                    "PEAKS_TASTE_MODES",
                    modeling_raw.get("taste_modes", ModelingConfig.taste_modes),
                )
            ),
        )
        library = LibraryConfig(
            path=os.environ.get(
                "PEAKS_LIBRARY_PATH", library_raw.get("path", LibraryConfig.path)
            ),
        )
        schedule_raw = raw.get("schedule", {})
        schedule = ScheduleConfig(
            embed_hours=float(
                os.environ.get(
                    "PEAKS_EMBED_EVERY_HOURS",
                    schedule_raw.get("embed_hours", ScheduleConfig.embed_hours),
                )
            ),
            sync=_as_bool(
                os.environ.get("PEAKS_SYNC"),
                schedule_raw.get("sync", ScheduleConfig.sync),
            ),
            prune=_as_bool(
                os.environ.get("PEAKS_PRUNE"),
                schedule_raw.get("prune", ScheduleConfig.prune),
            ),
        )
        auth_raw = raw.get("auth", {})
        auth = AuthConfig(
            password=os.environ.get(
                "PEAKS_PASSWORD", auth_raw.get("password", AuthConfig.password)
            ),
            session_hours=float(
                os.environ.get(
                    "PEAKS_SESSION_HOURS",
                    auth_raw.get("session_hours", AuthConfig.session_hours),
                )
            ),
        )
        return cls(
            stash=stash,
            sampling=sampling,
            markers=markers,
            embedding=embedding,
            scoring=scoring,
            modeling=modeling,
            library=library,
            schedule=schedule,
            auth=auth,
        )
