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


VIDEO_EXTS = {".mp4", ".mkv", ".m4v", ".mov", ".wmv", ".avi", ".ts", ".webm"}


def iter_video_files(paths) -> list[str]:
    """Expand the given paths into a sorted list of video files: a directory is
    walked recursively for known video extensions; a file is kept as-is. Lets
    `embed /data` take a whole library."""
    out: list[str] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            out += [str(f) for f in pth.rglob("*")
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
        else:
            out.append(str(pth))
    return sorted(dict.fromkeys(out))  # dedupe, stable order


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
    from .failures import FailureLog

    embedder = get_embedder(args.model)
    cache = EmbeddingCache(args.cache)
    failures = FailureLog(Path(args.cache).parent / "failures.json")
    hwaccel = "" if args.hwaccel == "none" else args.hwaccel
    if args.vr:
        from .reprojection import Reprojector
        from .vr_format import detect

    if args.retry_failed:
        videos = [e["path"] for e in failures.entries() if e.get("path")]
        if not videos:
            print("  · no failed files to retry", file=sys.stderr)
            return 0
    else:
        videos = iter_video_files(args.videos)
    if not videos:
        print("  ✗ no video files found", file=sys.stderr)
        return 1
    print(f"embedding {len(videos)} file(s) (model '{embedder.name}', "
          f"interval {args.interval:g}s, hwaccel {hwaccel or 'off'}"
          f"{', VR de-warp' if args.vr else ''}"
          f"{', retry' if args.retry_failed else ''})…", file=sys.stderr)

    total = len(videos)
    stats = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
    for path in videos:
        # A fresh sampler per file so the VR de-warp can be file-specific
        # (each scene has its own projection). VR uses sparse mode — one
        # seek+de-warp per sample, not a full-file decode.
        reproject = None
        if args.vr:
            dims = FrameSampler(hwaccel=hwaccel).probe_dimensions(path)
            aspect = (dims[0] / dims[1]) if dims else None
            fmt = detect(Path(path).name, aspect_ratio=aspect, assume=args.assume)
            if not fmt.is_known:
                print(f"  ! {Path(path).name}: VR format not recognized "
                      f"(no hint/assume); skipping de-warp", file=sys.stderr)
            else:
                reproject = Reprojector.for_format(fmt)
        sampler = FrameSampler(interval_seconds=args.interval, mode="sparse",
                               hwaccel=hwaccel, reproject=reproject)
        s = embed_library([scene_from_path(path)], sampler, embedder, cache,
                          total=total, failure_log=failures)
        for k in stats:
            stats[k] += s.get(k, 0)
    print(f"\nembed: {stats['embedded']} embedded, {stats['skipped']} skipped, "
          f"{stats['failed']} failed, {stats['frames']} frames "
          f"→ cache '{args.cache}' (model '{embedder.name}')")
    if len(failures):
        print(f"  {len(failures)} file(s) in the failure log "
              f"(retry with --retry-failed)", file=sys.stderr)
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


def _probe_listen(args) -> int:
    """Listen for HereSphere's timestamp-server push (headset connects to us)."""
    import socket as _socket

    from .heresphere import RemoteError, TimestampReceiver

    def lan_ip() -> str:
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
            return ip
        except OSError:
            return "<this-computer-ip>"

    rx = TimestampReceiver(port=args.ts_port, byteorder=args.byteorder)
    try:
        rx.bind()
    except RemoteError as exc:
        print(f"  ✗ {exc}", file=sys.stderr)
        return 1
    print(f"listening on {lan_ip()}:{args.ts_port} — set HereSphere's timestamp "
          f"server to this address, then play a video.", file=sys.stderr)
    try:
        ip = rx.accept(timeout=args.seconds)
    except RemoteError as exc:
        print(f"  ✗ {exc} (nothing connected within {args.seconds:g}s)",
              file=sys.stderr)
        rx.close()
        return 1
    print(f"  ✓ HereSphere connected from {ip}", file=sys.stderr)

    seen = 0
    raw_shown = False

    def on_raw(chunk: bytes) -> None:
        nonlocal raw_shown
        if raw_shown:
            return
        raw_shown = True
        print(f"  ? couldn't decode the stream — first bytes (hex):\n    "
              f"{chunk[:64].hex(' ')}\n    ascii: {chunk[:64]!r}\n    "
              f"(paste this back — it tells us the exact format to parse)",
              file=sys.stderr)

    deadline = time.monotonic() + args.seconds
    try:
        for st in rx.monitor(on_raw=on_raw):
            seen += 1
            state = "▶ playing" if st.playing else "⏸ paused"
            dur = f"/{st.duration:.1f}s" if st.duration else ""
            print(f"  [{seen:>3}] {state}  t={st.current_time:7.2f}s{dur}  "
                  f"path={st.path!r}")
            if time.monotonic() >= deadline:
                break
    except RemoteError as exc:
        print(f"  ✗ {exc}", file=sys.stderr)
    finally:
        rx.close()

    if seen == 0:
        print("  ✗ connected but decoded no playback packets — see the hex above.",
              file=sys.stderr)
        return 1
    print(f"\nprobe: received {seen} timestamp packet(s) — the READ surface works "
          f"(flagging is good to go). Control/DJ still needs the DeoVR remote: "
          f"try `peaks-vr probe --host <headset-ip>`.")
    return 0


def cmd_probe(args) -> int:
    """On-device check of the HereSphere link. Two modes: --listen receives the
    timestamp-server push (headset → us, read); --host dials the DeoVR remote
    (us → headset, read + control)."""
    from .heresphere import RemoteClient, RemoteError

    if args.listen:
        return _probe_listen(args)
    if not args.host:
        print("  ✗ give --host <headset-ip> (DeoVR remote) or --listen "
              "(timestamp server)", file=sys.stderr)
        return 2

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


def cmd_preview(args) -> int:
    """Render ONE de-warped frame so you can eyeball the projection before
    embedding a whole library through it."""
    from .reprojection import Reprojector
    from .vr_format import detect

    hwaccel = "" if args.hwaccel == "none" else args.hwaccel
    name = Path(args.video).name
    dims = FrameSampler(hwaccel=hwaccel).probe_dimensions(args.video)
    aspect = (dims[0] / dims[1]) if dims else None
    fmt = detect(name, aspect_ratio=aspect, assume=args.assume)
    print(f"detected: projection={fmt.projection.value} layout={fmt.layout.value} "
          f"fov={fmt.fov_deg} confidence={fmt.confidence:.0%} (source: {fmt.source})",
          file=sys.stderr)
    if not fmt.is_known:
        print("  ✗ couldn't determine the VR format. Add a filename hint "
              "(e.g. _180_sbs, _MKX200_tb) or pass --assume 180_sbs.",
              file=sys.stderr)
        return 1

    reproject = Reprojector.for_format(
        fmt, viewport_fov_deg=args.fov, yaw=args.yaw, pitch=args.pitch,
    )
    sampler = FrameSampler(hwaccel=hwaccel, reproject=reproject)
    try:
        img = sampler.grab_frame(args.video, args.time)
    except Exception as exc:
        print(f"  ✗ de-warp failed: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="JPEG", quality=90)
    print(f"wrote {out} — open it: it should look like a normal forward-facing "
          f"view (not fisheye/stretched, no seam). Tune with --fov/--yaw/--pitch "
          f"and re-run if it's off.")
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
    """Launch the peaks-vr control panel + flagging web UI."""
    from .web.flagging import run, run_demo

    if args.demo:
        run_demo(web_host=args.web_host, web_port=args.web_port,
                 labels_path=args.labels, profile=args.profile,
                 media_root=args.media, cache_root=args.cache, model=args.model,
                 assume_default=args.assume)
        return 0
    if not args.listen and not args.host:
        print("  ✗ need --listen (HereSphere timestamp server) or --host "
              "<headset-ip> (DeoVR remote), or --demo", file=sys.stderr)
        return 2
    run(args.host, args.port, listen=args.listen, ts_port=args.ts_port,
        web_host=args.web_host, web_port=args.web_port, labels_path=args.labels,
        profile=args.profile, byteorder=args.byteorder,
        media_root=args.media, cache_root=args.cache, model=args.model,
        assume_default=args.assume)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="peaks-vr", description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=DEFAULT_CACHE,
                   help=f"embedding cache directory (default: {DEFAULT_CACHE})")
    p.add_argument("--model", default="fake", choices=["fake", "dino", "clip"],
                   help="embedder (default: fake — offline, no GPU)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("embed", help="sample + embed video(s) into the cache")
    e.add_argument("videos", nargs="+",
                   help="video files or directories (dirs are scanned recursively)")
    e.add_argument("--interval", type=float, default=8.0,
                   help="seconds between samples (default: 8)")
    e.add_argument("--vr", action="store_true",
                   help="VR de-warp: detect format + reproject one eye to a flat "
                        "viewport before embedding (needs ffmpeg on PATH)")
    e.add_argument("--hwaccel", default="none", choices=["none", "auto", "cuda"],
                   help="hardware decode for the flat/interval path (default none; "
                        "the VR de-warp path always decodes per-frame on CPU)")
    e.add_argument("--assume", default="180_sbs",
                   help="format to assume for files with no filename hint "
                        "(e.g. 180_sbs, 180_tb, mkx200_sbs; '' to skip them)")
    e.add_argument("--retry-failed", action="store_true",
                   help="re-embed only the files in the failure log")
    e.set_defaults(func=cmd_embed)

    pv = sub.add_parser("preview", help="render one de-warped frame to eyeball "
                        "the VR reprojection before embedding")
    pv.add_argument("video", help="a VR video file")
    pv.add_argument("--time", type=float, default=60.0,
                    help="timestamp to sample, seconds (default: 60)")
    pv.add_argument("--out", default="preview.jpg",
                    help="output JPEG path (default: preview.jpg)")
    pv.add_argument("--fov", type=float, default=100.0,
                    help="output viewport horizontal FOV, degrees (default: 100)")
    pv.add_argument("--yaw", type=float, default=0.0,
                    help="viewport yaw, degrees (default: 0 = forward)")
    pv.add_argument("--pitch", type=float, default=0.0,
                    help="viewport pitch, degrees (default: 0; negative looks down)")
    pv.add_argument("--hwaccel", default="none", choices=["none", "auto", "cuda"],
                    help="GPU-assisted decode for the grab (default: none)")
    pv.add_argument("--assume", default="180_sbs",
                    help="format to assume if the filename has no hint "
                         "(default: 180_sbs; '' to require a hint)")
    pv.set_defaults(func=cmd_preview)

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

    pr = sub.add_parser("probe", help="check the HereSphere link — --listen for "
                        "the timestamp server, or --host for the DeoVR remote")
    pr.add_argument("--listen", action="store_true",
                    help="receive HereSphere's timestamp-server push (headset "
                         "connects to us — read surface)")
    pr.add_argument("--ts-port", type=int, default=23573,
                    help="port to listen on for --listen (default: 23573)")
    pr.add_argument("--host",
                    help="headset IP for the DeoVR remote (read + control)")
    pr.add_argument("--port", type=int, default=23554,
                    help="DeoVR remote TCP port (default: 23554)")
    pr.add_argument("--seconds", type=float, default=10.0,
                    help="how long to wait/listen (default: 10)")
    pr.add_argument("--byteorder", default="big", choices=["big", "little"],
                    help="length-prefix endianness (default: big)")
    pr.add_argument("--test-seek", type=float, default=None, metavar="T",
                    help="(--host only) also send one seek to T seconds")
    pr.set_defaults(func=cmd_probe)

    fl = sub.add_parser("flag", help="real-time moment flagging web UI (#2) — "
                        "mirror the headset and ❤️-mark moments")
    fl.add_argument("--listen", action="store_true",
                    help="receive HereSphere's timestamp-server push (headset "
                         "connects to us) instead of dialing the DeoVR remote")
    fl.add_argument("--ts-port", type=int, default=23573,
                    help="port HereSphere's timestamp server pushes to "
                         "(default: 23573; used with --listen)")
    fl.add_argument("--host", help="headset / player IP for the DeoVR remote "
                    "(omit with --listen or --demo)")
    fl.add_argument("--port", type=int, default=23554,
                    help="HereSphere DeoVR remote port (default: 23554)")
    fl.add_argument("--web-host", default="0.0.0.0",
                    help="address to serve the UI on (default: 0.0.0.0)")
    fl.add_argument("--web-port", type=int, default=8801,
                    help="port to serve the UI on (default: 8801)")
    fl.add_argument("--profile", default="apex",
                    help="taste profile the ❤️ marks belong to (default: apex)")
    fl.add_argument("--labels", default="labels.json",
                    help="labels JSON file to append marks to")
    fl.add_argument("--byteorder", default="big", choices=["big", "little"],
                    help="remote length-prefix endianness (default: big)")
    fl.add_argument("--media", default=None,
                    help="library directory to enable the Embed tab (e.g. /data)")
    fl.add_argument("--assume", default="180_sbs",
                    help="default format for files with no filename hint "
                         "(default: 180_sbs)")
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
