"""Real-time moment flagging (README feature #2).

While you watch in HereSphere, this app mirrors playback on a nearby computer
and turns a ❤️ tap into a training label at (scene, timecode):

    RemoteClient.monitor()  ->  PlaybackMirror  ->  GET /api/state  ->  browser
                                                    POST /api/mark   ->  LabelStore

The design keeps the network feed and the HTTP app decoupled through
:class:`PlaybackMirror`, so the whole thing runs and is testable with **no
headset**: production feeds the mirror from a real :class:`RemoteClient`, while
``--demo`` (and the test suite) feed it a synthetic moving timecode.

Reuses, rather than reinventing: :class:`peaks_vr.labels.LabelStore` (the ❤️
store), :class:`peaks_vr.heresphere.RemoteClient` (the live feed),
:func:`peaks_vr.cache.path_key` (the scene key a mark attaches to), and — for the
optional de-warped preview — :class:`peaks_vr.sampling.FrameSampler` +
:class:`peaks_vr.reprojection.Reprojector`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..cache import path_key
from ..heresphere import PlaybackState
from ..labels import LabelStore

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PROFILE = "apex"
DEFAULT_WEB_PORT = 8801

# The request body must be a MODULE-LEVEL pydantic model or FastAPI resolves it
# as query params instead of a body (a locally-defined class fails hint
# resolution). Guarded so importing this module doesn't require the [web] extra.
try:
    from pydantic import BaseModel

    class MarkBody(BaseModel):
        time: float
        in_time: float | None = None
        out_time: float | None = None

    class EmbedBody(BaseModel):
        interval: float = 8.0
        vr: bool = True
        hwaccel: str = "auto"   # affects the flat path; VR decodes in-process
        assume: str | None = None
except ImportError:  # pragma: no cover - only when fastapi/pydantic absent
    MarkBody = None  # type: ignore
    EmbedBody = None  # type: ignore


# --- the playback mirror ----------------------------------------------------

class PlaybackMirror:
    """Thread-safe holder of the latest :class:`PlaybackState`.

    A feeder thread calls :meth:`update` as packets arrive; the HTTP handlers
    read :meth:`snapshot`. ``connected`` reflects whether the feed is live.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: PlaybackState | None = None
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, state: PlaybackState) -> None:
        with self._lock:
            self._state = state
            self._connected = True

    def set_connected(self, value: bool) -> None:
        with self._lock:
            self._connected = value

    def snapshot(self) -> tuple[bool, PlaybackState | None]:
        with self._lock:
            return self._connected, self._state

    # feeder lifecycle -------------------------------------------------------

    def start_feeder(self, feeder: Callable[["PlaybackMirror", threading.Event], None]) -> None:
        """Run ``feeder(self, stop_event)`` on a daemon thread."""
        self._thread = threading.Thread(target=feeder, args=(self, self._stop),
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def remote_feeder(host: str, port: int, *, byteorder: str = "big"):
    """A feeder that streams a real :class:`RemoteClient` into the mirror,
    reconnecting on drop. Returns a callable for :meth:`PlaybackMirror.start_feeder`."""
    from ..heresphere import RemoteClient, RemoteError

    def feed(mirror: PlaybackMirror, stop: threading.Event) -> None:
        while not stop.is_set():
            client = RemoteClient(host, port, byteorder=byteorder)
            try:
                client.connect()
                for state in client.monitor():
                    if stop.is_set():
                        break
                    mirror.update(state)
            except RemoteError:
                mirror.set_connected(False)
                stop.wait(2.0)  # brief backoff before reconnect
            finally:
                client.close()
            mirror.set_connected(False)

    return feed


def timestamp_feeder(port: int, *, host: str = "0.0.0.0", byteorder: str = "big"):
    """A feeder for HereSphere's **timestamp server**: the headset connects to
    us and pushes playback, so we listen (inverse of :func:`remote_feeder`).
    Re-arms after a disconnect so the next play session reconnects cleanly."""
    from ..heresphere import RemoteError, TimestampReceiver

    def feed(mirror: PlaybackMirror, stop: threading.Event) -> None:
        rx = TimestampReceiver(host=host, port=port, byteorder=byteorder)
        try:
            rx.bind()
        except RemoteError as exc:
            print(f"timestamp server: {exc}")
            return
        while not stop.is_set():
            try:
                rx.accept()
                mirror.set_connected(True)
                for state in rx.monitor():
                    if stop.is_set():
                        break
                    mirror.update(state)
            except RemoteError:
                pass
            mirror.set_connected(False)
            rx._conn = None  # drop the finished connection, wait for the next
        rx.close()

    return feed


def demo_feeder(path: str = "D:/vr/demo_scene_180_sbs.mp4", duration: float = 600.0):
    """A headset-free feeder: a timecode that advances in real time and loops,
    so the UI is fully demonstrable with no hardware."""

    def feed(mirror: PlaybackMirror, stop: threading.Event) -> None:
        start = time.monotonic()
        while not stop.wait(0.5):
            t = (time.monotonic() - start) % duration
            mirror.update(PlaybackState(
                path=path, current_time=round(t, 2), playing=True,
                duration=duration,
            ))

    return feed


# --- the app ----------------------------------------------------------------

@dataclass
class MarkResult:
    key: str
    times: list[float]
    positives: int
    total: int


# model name → cache subdirectory (matches embedding._CANONICAL)
_CANONICAL_MODEL = {"fake": "fake", "dino": "dinov2", "clip": "clip"}


def _reprojector_for(path: str, *, assume: str | None = None, sampler=None):
    """Build a VR de-warp for a file. Filename hints win; otherwise the frame's
    aspect ratio (probed via ``sampler``) sets the stereo layout and ``assume``
    fills in the rest — so unhinted files still de-warp. Returns (Reprojector or
    None, VRFormat)."""
    from ..reprojection import Reprojector
    from ..vr_format import detect

    aspect = None
    if sampler is not None:
        dims = sampler.probe_dimensions(path)
        if dims:
            aspect = dims[0] / dims[1]
    fmt = detect(Path(path).name, aspect_ratio=aspect, assume=assume)
    return (Reprojector.for_format(fmt) if fmt.is_known else None), fmt


def failure_log_for(cache_root: str):
    """The failure log beside the cache (persists in the /config volume)."""
    from ..failures import FailureLog
    return FailureLog(Path(cache_root).parent / "failures.json")


def embed_job(job, mgr, *, media_root: str, cache_root: str, model: str,
              interval: float, vr: bool, hwaccel: str,
              assume: str | None = None, paths: list[str] | None = None) -> None:
    """Background task: embed videos (resumable). Scans ``media_root`` unless an
    explicit ``paths`` list is given (used for retrying just the failures).
    Each casualty is recorded in the failure log; a later success clears it."""
    from ..cache import EmbeddingCache
    from ..cli import iter_video_files, scene_from_path
    from ..embedding import get_embedder
    from ..pipeline import embed_library
    from ..sampling import FrameSampler

    videos = paths if paths is not None else iter_video_files([media_root])
    job.total = len(videos)
    where = "retry set" if paths is not None else media_root
    job.log(f"found {len(videos)} file(s) ({where})")
    if not videos:
        return
    embedder = get_embedder(model)
    cache = EmbeddingCache(cache_root)
    failures = failure_log_for(cache_root)
    hw = "" if hwaccel in ("none", "cpu") else hwaccel
    for path in videos:
        if mgr.should_stop():
            job.log("stop requested — halting")
            break
        job.set_current(Path(path).name)
        reproject = None
        if vr:
            probe = FrameSampler(hwaccel=hw)  # for probe_dimensions only
            reproject, fmt = _reprojector_for(path, assume=assume, sampler=probe)
            if reproject is None:
                job.log(f"! {job.current}: VR format unknown (no hint, no assume) "
                        f"— skipping de-warp")
        sampler = FrameSampler(interval_seconds=interval, mode="sparse",
                               hwaccel=hw, reproject=reproject)
        s = embed_library([scene_from_path(path)], sampler, embedder, cache,
                          log=job.log, failure_log=failures)
        for k in ("embedded", "skipped", "failed", "frames"):
            job.stats[k] = job.stats.get(k, 0) + s.get(k, 0)
        job.done += 1
    job.stats["failed_total"] = len(failures)


def create_app(mirror: PlaybackMirror, store: LabelStore, *,
               profile: str = DEFAULT_PROFILE, sampler=None,
               media_root: str | None = None, cache_root: str = "cache",
               model: str = "dino", jobs=None, assume_default: str = "180_sbs"):
    """Build the FastAPI app. ``sampler`` (a configured
    :class:`peaks_vr.sampling.FrameSampler`, optionally with a ``reproject``)
    enables the live de-warped preview; without it ``/api/frame`` returns 503."""
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import FileResponse

    from .jobs import JobManager

    app = FastAPI(title="peaks-vr")
    active_profile = profile
    jobs = jobs or JobManager()
    model_dir = _CANONICAL_MODEL.get(model, model)

    def _state_dict() -> dict:
        connected, st = mirror.snapshot()
        if st is None:
            return {"connected": connected, "path": None, "key": None,
                    "current_time": 0.0, "playing": False, "duration": None}
        return {
            "connected": connected,
            "path": st.path,
            "key": path_key(st.path) if st.path else None,
            "current_time": st.current_time,
            "playing": st.playing,
            "duration": st.duration,
        }

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    def state():
        return _state_dict()

    @app.post("/api/mark")
    def mark(body: MarkBody):
        _, st = mirror.snapshot()
        if st is None or not st.path:
            raise HTTPException(409, "no scene is playing yet")
        key = path_key(st.path)
        # An in/out window records both endpoints (precise apex clip); otherwise
        # a single point. All are positive labels for `profile`.
        times = sorted({t for t in (body.in_time, body.time, body.out_time)
                        if t is not None})
        for t in times:
            store.add(key, float(t), 1, profile, scene_id=st.path)
        store.save()
        pos, neg = store.counts(profile)
        return MarkResult(key=key, times=times, positives=pos,
                          total=pos + neg).__dict__

    @app.get("/api/marks")
    def marks(profile: str | None = None, limit: int = 50):
        prof = profile or active_profile
        labs = [l for l in store.for_profile(prof) if l.label == 1]
        labs.sort(key=lambda l: l.ts, reverse=True)
        return [{"key": l.key, "time": l.time, "scene_id": l.scene_id, "ts": l.ts}
                for l in labs[:limit]]

    @app.get("/api/frame")
    def frame(time: float):
        _, st = mirror.snapshot()
        if sampler is None or st is None or not st.path:
            raise HTTPException(503, "frame preview unavailable "
                                "(no sampler configured or nothing playing)")
        try:
            img = sampler.grab_frame(st.path, time)
        except Exception as exc:  # ffmpeg missing / file unreadable
            raise HTTPException(503, f"could not render frame: {exc}")
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    # --- control panel: library + embedding ---------------------------------

    def _cache_count() -> int:
        from ..cache import EmbeddingCache
        try:
            return len(list(EmbeddingCache(cache_root).keys(model_dir)))
        except Exception:
            return 0

    def _safe_under_media(path: str) -> str:
        """Resolve ``path`` and ensure it stays within media_root."""
        base = Path(media_root or ".").resolve()
        p = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if media_root and base not in p.parents and p != base:
            raise HTTPException(400, "path outside the media directory")
        return str(p)

    @app.get("/api/config")
    def config():
        return {"media_root": media_root, "model": model,
                "assume_default": assume_default,
                "has_media": bool(media_root and Path(media_root).is_dir())}

    @app.get("/api/library")
    def library(limit: int = 2000):
        from ..cli import iter_video_files
        if not media_root:
            return {"count": 0, "files": []}
        files = iter_video_files([media_root])
        return {"count": len(files),
                "files": [{"path": f, "name": Path(f).name} for f in files[:limit]]}

    @app.get("/api/stats")
    def stats():
        return {"embedded": _cache_count(), "model": model,
                "running": jobs.running}

    @app.get("/api/preview")
    def preview(path: str, time: float = 60.0, fov: float = 100.0,
                pitch: float = 0.0, yaw: float = 0.0, hwaccel: str = "none",
                assume: str | None = None):
        from ..reprojection import Reprojector
        from ..sampling import FrameSampler
        from ..vr_format import detect

        real = _safe_under_media(path)
        hw = "" if hwaccel == "none" else hwaccel
        probe = FrameSampler(hwaccel=hw)
        dims = probe.probe_dimensions(real)
        aspect = (dims[0] / dims[1]) if dims else None
        fmt = detect(Path(real).name, aspect_ratio=aspect,
                     assume=assume if assume is not None else assume_default)
        if not fmt.is_known:
            raise HTTPException(422, f"VR format not recognized for "
                                f"{Path(real).name} — set an 'assume' format")
        rep = Reprojector.for_format(fmt, viewport_fov_deg=fov, pitch=pitch, yaw=yaw)
        s = FrameSampler(hwaccel=hw, reproject=rep)
        try:
            img = s.grab_frame(real, time)
        except Exception as exc:
            raise HTTPException(503, f"could not render preview: {exc}")
        from io import BytesIO
        buf = BytesIO(); img.save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"X-VR-Format":
                                 f"{fmt.projection.value}/{fmt.layout.value}/"
                                 f"{fmt.fov_deg} ({fmt.source})"})

    @app.post("/api/embed/start")
    def embed_start(body: EmbedBody):
        if not media_root:
            raise HTTPException(400, "no media directory configured (mount /data)")
        try:
            jobs.start("embed", lambda job, mgr: embed_job(
                job, mgr, media_root=media_root, cache_root=cache_root,
                model=model, interval=body.interval, vr=body.vr,
                hwaccel=body.hwaccel,
                assume=body.assume if body.assume is not None else assume_default))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"started": True}

    @app.post("/api/embed/stop")
    def embed_stop():
        jobs.stop()
        return {"stopping": True}

    @app.get("/api/embed/status")
    def embed_status():
        snap = jobs.status()
        return {"job": snap, "embedded": _cache_count(), "running": jobs.running}

    @app.get("/api/failures")
    def failures():
        log = failure_log_for(cache_root)
        return {"count": len(log),
                "entries": [{"name": Path(e.get("path") or e["key"]).name,
                             "path": e.get("path"), "error": e.get("error", ""),
                             "hwaccel": e.get("hwaccel", ""), "ts": e.get("ts")}
                            for e in log.entries()]}

    @app.post("/api/embed/retry")
    def embed_retry():
        log = failure_log_for(cache_root)
        paths = [e["path"] for e in log.entries() if e.get("path")]
        if not paths:
            return {"started": False, "reason": "no failed files to retry"}
        try:
            jobs.start("retry", lambda job, mgr: embed_job(
                job, mgr, media_root=media_root or "", cache_root=cache_root,
                model=model, interval=8.0, vr=True, hwaccel="auto",
                assume=assume_default, paths=paths))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"started": True, "count": len(paths)}

    @app.post("/api/failures/clear")
    def failures_clear():
        return {"cleared": failure_log_for(cache_root).clear()}

    return app


# --- entry points -----------------------------------------------------------

def _serve(app, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def run(hs_host: str | None = None, hs_port: int = 23554, *,
        listen: bool = False, ts_port: int = 23573,
        web_host: str = "0.0.0.0", web_port: int = DEFAULT_WEB_PORT,
        labels_path: str = "labels.json", profile: str = DEFAULT_PROFILE,
        byteorder: str = "big", sampler=None,
        media_root: str | None = None, cache_root: str = "cache",
        model: str = "dino", assume_default: str = "180_sbs") -> None:
    """Run the peaks-vr control panel + flagging UI.

    Two ways to source playback: ``listen=True`` receives HereSphere's
    **timestamp server** push on ``ts_port`` (the headset connects to us — the
    container default); otherwise we dial the headset's DeoVR remote at
    ``hs_host:hs_port``. ``media_root`` (e.g. ``/data``) enables the Embed tab.
    """
    mirror = PlaybackMirror()
    if listen:
        mirror.start_feeder(timestamp_feeder(ts_port, byteorder=byteorder))
        source = f"timestamp server :{ts_port} (HereSphere connects in)"
    else:
        mirror.start_feeder(remote_feeder(hs_host, hs_port, byteorder=byteorder))
        source = f"remote {hs_host}:{hs_port}"
    store = LabelStore(labels_path)
    app = create_app(mirror, store, profile=profile, sampler=sampler,
                     media_root=media_root, cache_root=cache_root, model=model,
                     assume_default=assume_default)
    print(f"peaks-vr → http://{web_host}:{web_port}  "
          f"(source: {source}, media: {media_root or 'none'}, model '{model}')")
    _serve(app, web_host, web_port)


def run_demo(*, web_host: str = "127.0.0.1", web_port: int = DEFAULT_WEB_PORT,
             labels_path: str = "labels.demo.json",
             profile: str = DEFAULT_PROFILE, media_root: str | None = None,
             cache_root: str = "cache", model: str = "fake",
             assume_default: str = "180_sbs") -> None:
    """Run the control panel with a synthetic playback feed — no headset."""
    mirror = PlaybackMirror()
    mirror.start_feeder(demo_feeder())
    store = LabelStore(labels_path)
    app = create_app(mirror, store, profile=profile, media_root=media_root,
                     cache_root=cache_root, model=model,
                     assume_default=assume_default)
    print(f"peaks-vr DEMO → http://{web_host}:{web_port}  "
          f"(synthetic playback, marks → {labels_path})")
    _serve(app, web_host, web_port)
