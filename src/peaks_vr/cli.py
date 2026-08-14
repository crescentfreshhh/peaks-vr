"""peaks-vr command line — run the taste loop over local video files.

Two subcommands cover the end-to-end path the README describes, minus the parts
that still need Phase-0 research (HereSphere) and a GPU:

  peaks-vr embed VIDEO...            sample -> (optional VR de-warp) -> embed ->
                                     cache. Resumable; skips already-cached files.
  peaks-vr score VIDEO... -r STILLS  score cached scenes against reference stills
                                     and print the moments it would mark (dry run).

Defaults to the offline ``FakeEmbedder`` so the whole pipeline runs with no
torch, no GPU, and no model download — install the ``[ml]`` extra and pass
``--model dino``/``--model clip`` for the real thing. Sampling uses sparse mode
(PyAV, bundled codecs) so a plain ``pip install -e .[ml]`` needs no system
ffmpeg for the flat path; ``--vr`` de-warp uses the ffmpeg ``v360`` filter and
therefore needs a real ffmpeg on PATH.

This operates on files on disk directly (no Stash), which is the quickest way to
exercise the plumbing; a Stash-backed path can be added later by reusing
``peaks_vr.stash_client`` and ``pipeline.score_library``'s ``write=True`` branch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cache import EmbeddingCache
from .config import ScoringConfig
from .embedding import get_embedder
from .models import Scene, SceneFile
from .pipeline import embed_library, load_references, score_library, score_scene
from .sampling import FrameSampler
from .scoring import make_similarity_scorer

DEFAULT_CACHE = "cache/embeddings"


def scene_from_path(path: str) -> Scene:
    """Build a minimal :class:`Scene` from a local file path.

    No Stash involved: the file's own path is its identity, and the cache keys
    off a hash of it (see ``pipeline.scene_key`` → ``cache.path_key``). Enough
    for the local-file embed/score loop; a Stash-sourced Scene carries more.
    """
    p = Path(path)
    return Scene(id=str(p), title=p.stem, files=[SceneFile(
        path=str(p), duration=None, width=None, height=None,
        frame_rate=None, video_codec=None, size=None,
    )])


def cmd_embed(args) -> int:
    embedder = get_embedder(args.model)
    cache = EmbeddingCache(args.cache)
    if args.vr:
        from .reprojection import Reprojector
        from .vr_format import detect

    total = len(args.videos)
    stats = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
    for path in args.videos:
        # A fresh sampler per file so the VR de-warp can be file-specific
        # (each scene has its own projection).
        reproject = None
        mode = "sparse"
        if args.vr:
            mode = "interval"
            fmt = detect(Path(path).name)
            if not fmt.is_known:
                print(f"  ! {Path(path).name}: VR format not recognized "
                      f"(confidence {fmt.confidence:.0%}); skipping de-warp",
                      file=sys.stderr)
            else:
                reproject = Reprojector.for_format(fmt)
        sampler = FrameSampler(interval_seconds=args.interval, mode=mode,
                               reproject=reproject)
        s = embed_library([scene_from_path(path)], sampler, embedder, cache,
                          total=total)
        for k in stats:
            stats[k] += s.get(k, 0)
    print(f"\nembed: {stats['embedded']} embedded, {stats['skipped']} skipped, "
          f"{stats['failed']} failed, {stats['frames']} frames "
          f"→ cache '{args.cache}' (model '{embedder.name}')")
    return 0 if stats["failed"] == 0 else 1


def cmd_score(args) -> int:
    embedder = get_embedder(args.model)
    cache = EmbeddingCache(args.cache)
    references = load_references(embedder, args.references)
    score_frames = make_similarity_scorer(references, reduce=args.reduce)
    scoring = ScoringConfig(
        high=args.high, low=args.low, min_duration=args.min_duration,
        merge_gap=args.merge_gap, max_duration=args.max_duration, pad=args.pad,
    )

    scenes = [scene_from_path(p) for p in args.videos]
    # Dry run: print the moments, don't write anywhere (write=True needs Stash).
    n_scenes = n_segs = 0
    for scene in scenes:
        from .pipeline import scene_key

        key = scene_key(scene)
        if not cache.has(key, embedder.name):
            print(f"  ? {scene.title}: not embedded yet — run 'embed' first",
                  file=sys.stderr)
            continue
        times, vecs, _ = cache.load(key, embedder.name)
        segs = score_scene(times, vecs, score_frames, scoring)
        n_scenes += 1
        n_segs += len(segs)
        if not segs:
            print(f"  · {scene.title}: no moments above threshold")
        for s in segs:
            print(f"  ♥ {scene.title}: {s.start:7.1f}–{s.end:7.1f}s "
                  f"(peak {s.peak_score:.3f}, {s.duration:.1f}s)")
    print(f"\nscore: {n_segs} moments across {n_scenes} scene(s) "
          f"[dry run — no markers written]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="peaks-vr", description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=DEFAULT_CACHE,
                   help=f"embedding cache directory (default: {DEFAULT_CACHE})")
    p.add_argument("--model", default="fake", choices=["fake", "dino", "clip"],
                   help="embedder (default: fake — offline, no GPU)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("embed", help="sample + embed video(s) into the cache")
    e.add_argument("videos", nargs="+", help="video file paths")
    e.add_argument("--interval", type=float, default=8.0,
                   help="seconds between samples (default: 8)")
    e.add_argument("--vr", action="store_true",
                   help="VR de-warp: detect format + reproject one eye to a flat "
                        "viewport before embedding (needs ffmpeg on PATH)")
    e.set_defaults(func=cmd_embed)

    s = sub.add_parser("score", help="score cached video(s) against reference stills")
    s.add_argument("videos", nargs="+", help="video file paths (must be embedded)")
    s.add_argument("-r", "--references", required=True,
                   help="directory of reference still images (the taste examples)")
    s.add_argument("--reduce", default="max", choices=["max", "mean"],
                   help="how to reduce similarity across references (default: max)")
    s.add_argument("--high", type=float, default=ScoringConfig.high)
    s.add_argument("--low", type=float, default=ScoringConfig.low)
    s.add_argument("--min-duration", type=float, default=ScoringConfig.min_duration)
    s.add_argument("--merge-gap", type=float, default=ScoringConfig.merge_gap)
    s.add_argument("--max-duration", type=float, default=ScoringConfig.max_duration)
    s.add_argument("--pad", type=float, default=ScoringConfig.pad)
    s.set_defaults(func=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
