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
import time
from pathlib import Path

from .cache import EmbeddingCache
from .config import ScoringConfig
from .embedding import get_embedder
from .models import Scene, SceneFile
from .pipeline import embed_library, load_references, score_library, score_scene
from .sampling import FrameSampler
from .scoring import make_similarity_scorer

DEFAULT_CACHE = "cache/embeddings"

# Model name → cache subdirectory (the embedder's canonical name). Lets score/
# recommend resolve the cache without instantiating a torch-heavy embedder.
_CANONICAL_MODEL = {"fake": "fake", "dino": "dinov2", "clip": "clip"}


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


def cmd_probe(args) -> int:
    """Phase-0 on-device check: connect to HereSphere's remote and print live
    playback state, optionally testing a seek. This is the single command the
    user runs against a real headset to confirm the read + control surface."""
    from .heresphere import RemoteClient, RemoteError

    print(f"connecting to {args.host}:{args.port} "
          f"({args.byteorder}-endian framing)…", file=sys.stderr)
    client = RemoteClient(args.host, args.port, byteorder=args.byteorder)
    try:
        client.connect()
    except RemoteError as exc:
        print(f"  ✗ {exc}", file=sys.stderr)
        print("  Is HereSphere/DeoVR playing with remote control enabled, and "
              "reachable at that IP?", file=sys.stderr)
        return 1

    seen = 0
    deadline = time.monotonic() + args.seconds
    tested_seek = False
    try:
        for st in client.monitor():
            seen += 1
            state = "▶ playing" if st.playing else "⏸ paused"
            dur = f"/{st.duration:.1f}s" if st.duration else ""
            print(f"  [{seen:>3}] {state}  t={st.current_time:7.2f}s{dur}  "
                  f"path={st.path!r}")
            if args.test_seek is not None and not tested_seek and seen >= 2:
                print(f"  → sending test seek to {args.test_seek:g}s…",
                      file=sys.stderr)
                client.seek(args.test_seek)
                tested_seek = True
            if time.monotonic() >= deadline:
                break
    except RemoteError as exc:
        print(f"  ✗ {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    if seen == 0:
        print("  ✗ connected but received no status packets. If the connection "
              "held, the length-prefix endianness may be flipped — retry with "
              f"--byteorder {'little' if args.byteorder == 'big' else 'big'}.",
              file=sys.stderr)
        return 1
    print(f"\nprobe: read {seen} status packet(s) — the READ surface works. "
          f"{'A test seek was sent — confirm the headset jumped (CONTROL surface).' if tested_seek else 'Add --test-seek 30 to also test control.'}")
    return 0


def cmd_recommend(args) -> int:
    """Turn ❤️ marks into a ranked playlist of similar moments (#3)."""
    from .labels import LabelStore
    from .recommend import recommend_from_labels

    embedder_name = _CANONICAL_MODEL.get(args.model, args.model)
    cache = EmbeddingCache(args.cache)
    store = LabelStore(args.labels)
    pos, _ = store.counts(args.profile)
    if pos == 0:
        print(f"  ✗ no ❤️ marks for profile '{args.profile}' in {args.labels} — "
              f"flag some moments first (peaks-vr flag)", file=sys.stderr)
        return 1
    scoring = ScoringConfig(
        high=args.high, low=args.low, min_duration=args.min_duration,
        merge_gap=args.merge_gap, max_duration=args.max_duration, pad=args.pad,
    )
    playlist = recommend_from_labels(
        cache, store, embedder_name, args.profile, scoring,
        limit=args.limit, exclude_seed_scenes=args.exclude_seeds,
    )
    if not playlist.moments:
        print("  · no moments cleared the similarity threshold — try lowering "
              "--high/--low, or embed more scenes", file=sys.stderr)
    for i, m in enumerate(playlist.moments, 1):
        name = (m.path or m.key).split("/")[-1].split("\\")[-1]
        print(f"  {i:>3}. {name}  {m.start:7.1f}–{m.end:7.1f}s  "
              f"(score {m.score:.3f})")
    print(f"\nrecommend: {len(playlist)} moments from {pos} liked "
          f"(profile '{args.profile}')")
    if args.out:
        playlist.save(args.out)
        print(f"  → wrote {args.out}")
    return 0


def cmd_dj(args) -> int:
    """Play a playlist back-to-back in the headset (#4)."""
    from .dj import DJ, play_playlist
    from .recommend import Playlist

    playlist = Playlist.load(args.playlist)
    if args.dry_run or not args.host:
        DJ(player=None).dry_run(playlist)  # player unused by dry_run
        if not args.host and not args.dry_run:
            print("\n  (no --host given — showed the set without playing. Add "
                  "--host <headset-ip> to play it.)", file=sys.stderr)
        return 0
    print(f"DJ: playing {len(playlist)} moments on {args.host}:{args.port}"
          f"{' (looping)' if args.loop else ''}…")
    played = play_playlist(playlist, args.host, args.port,
                           byteorder=args.byteorder, loop=args.loop)
    print(f"\nDJ: played {played} moment(s)")
    return 0


def cmd_flag(args) -> int:
    """Launch the real-time flagging web UI (README feature #2)."""
    from .web.flagging import run, run_demo

    if args.demo:
        run_demo(web_host=args.web_host, web_port=args.web_port,
                 labels_path=args.labels, profile=args.profile)
        return 0
    if not args.host:
        print("  ✗ --host <headset-ip> is required (or use --demo)",
              file=sys.stderr)
        return 2
    sampler = None
    if args.preview:
        # Optional live de-warped preview — needs ffmpeg on PATH.
        sampler = FrameSampler(interval_seconds=args.interval, mode="interval")
    run(args.host, args.port, web_host=args.web_host, web_port=args.web_port,
        labels_path=args.labels, profile=args.profile, byteorder=args.byteorder,
        sampler=sampler)
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

    pr = sub.add_parser("probe", help="Phase-0: check the HereSphere remote "
                        "control surface on a real headset")
    pr.add_argument("--host", required=True,
                    help="headset / player IP address (from its WiFi settings)")
    pr.add_argument("--port", type=int, default=23554,
                    help="remote-control TCP port (default: 23554)")
    pr.add_argument("--seconds", type=float, default=10.0,
                    help="how long to listen for status packets (default: 10)")
    pr.add_argument("--byteorder", default="big", choices=["big", "little"],
                    help="length-prefix endianness (default: big)")
    pr.add_argument("--test-seek", type=float, default=None, metavar="T",
                    help="also send one seek to T seconds, to test control")
    pr.set_defaults(func=cmd_probe)

    fl = sub.add_parser("flag", help="real-time moment flagging web UI (#2) — "
                        "mirror the headset and ❤️-mark moments")
    fl.add_argument("--host", help="headset / player IP (omit with --demo)")
    fl.add_argument("--port", type=int, default=23554,
                    help="HereSphere remote port (default: 23554)")
    fl.add_argument("--web-host", default="0.0.0.0",
                    help="address to serve the UI on (default: 0.0.0.0)")
    fl.add_argument("--web-port", type=int, default=8760,
                    help="port to serve the UI on (default: 8760)")
    fl.add_argument("--profile", default="apex",
                    help="taste profile the ❤️ marks belong to (default: apex)")
    fl.add_argument("--labels", default="labels.json",
                    help="labels JSON file to append marks to")
    fl.add_argument("--byteorder", default="big", choices=["big", "little"],
                    help="remote length-prefix endianness (default: big)")
    fl.add_argument("--preview", action="store_true",
                    help="enable the live de-warped frame preview (needs ffmpeg)")
    fl.add_argument("--interval", type=float, default=2.0,
                    help="preview sampler interval seconds (default: 2)")
    fl.add_argument("--demo", action="store_true",
                    help="run with a synthetic feed — no headset required")
    fl.set_defaults(func=cmd_flag)

    rc = sub.add_parser("recommend", help="turn ❤️ marks into a ranked playlist "
                        "of similar moments (#3)")
    rc.add_argument("--labels", default="labels.json",
                    help="labels JSON with your ❤️ marks (default: labels.json)")
    rc.add_argument("--profile", default="apex",
                    help="taste profile to recommend for (default: apex)")
    rc.add_argument("--limit", type=int, default=50,
                    help="max moments in the playlist (default: 50)")
    rc.add_argument("--exclude-seeds", action="store_true",
                    help="leave out the scenes your likes came from (new only)")
    rc.add_argument("--out", help="write the playlist to this JSON file")
    rc.add_argument("--high", type=float, default=ScoringConfig.high)
    rc.add_argument("--low", type=float, default=ScoringConfig.low)
    rc.add_argument("--min-duration", type=float, default=ScoringConfig.min_duration)
    rc.add_argument("--merge-gap", type=float, default=ScoringConfig.merge_gap)
    rc.add_argument("--max-duration", type=float, default=ScoringConfig.max_duration)
    rc.add_argument("--pad", type=float, default=ScoringConfig.pad)
    rc.set_defaults(func=cmd_recommend)

    dj = sub.add_parser("dj", help="play a playlist back-to-back in the headset (#4)")
    dj.add_argument("playlist", help="playlist JSON from `peaks-vr recommend --out`")
    dj.add_argument("--host", help="headset IP (omit for a dry-run preview)")
    dj.add_argument("--port", type=int, default=23554,
                    help="HereSphere remote port (default: 23554)")
    dj.add_argument("--byteorder", default="big", choices=["big", "little"])
    dj.add_argument("--loop", action="store_true", help="repeat until interrupted")
    dj.add_argument("--dry-run", action="store_true",
                    help="print the set without playing")
    dj.set_defaults(func=cmd_dj)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
