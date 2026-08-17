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
        hwaccel: str = "auto"   # "" | "cuda" | "auto" — NVDEC decode for VR/flat
        assume: str | None = None
        scene_timeout: float | None = None  # per-scene ceiling (s); None = env/default

    class ReembedBody(BaseModel):
        path: str
        format: str = "auto"    # "auto" = re-detect (clear override) | a token | "flat"
        fov: float = 100.0
        pitch: float = 0.0
        flat: bool = False

    class LoginBody(BaseModel):
        password: str = ""

    class TasteLikeBody(BaseModel):
        key: str
        time: float
        path: str | None = None
        category: str | None = None
        label: int = 1          # 1 = thumbs-up, 0 = thumbs-down

    class ProfileBody(BaseModel):
        profile: str = ""
except ImportError:  # pragma: no cover - only when fastapi/pydantic absent
    MarkBody = None  # type: ignore
    EmbedBody = None  # type: ignore
    ReembedBody = None  # type: ignore
    LoginBody = None  # type: ignore
    TasteLikeBody = None  # type: ignore
    ProfileBody = None  # type: ignore


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


def _forced_reprojector(token: str, *, fov: float = 100.0, pitch: float = 0.0,
                        flat: bool = False):
    """Build a de-warp from an *authoritative* format token — a QC correction, not
    a guess. ``flat`` means embed the raw centre crop (no de-warp). Returns
    (Reprojector or None, VRFormat); the reprojector is None for flat or an
    unparseable token."""
    from ..reprojection import Reprojector
    from ..vr_format import (Projection, StereoLayout, VRFormat,
                             detect_from_filename)

    if flat:
        return None, VRFormat(Projection.UNKNOWN, StereoLayout.MONO, None,
                              source="override:flat")
    fmt = detect_from_filename(token)
    if not fmt.is_known:
        return None, fmt
    return Reprojector.for_format(fmt, viewport_fov_deg=fov, pitch=pitch), fmt


def _reprojector_for(path: str, *, assume: str | None = None, sampler=None,
                     override: dict | None = None):
    """Build a VR de-warp for a file. A stored QC ``override`` wins outright;
    otherwise filename hints win, then the frame's aspect ratio (probed via
    ``sampler``) sets the stereo layout and ``assume`` fills the rest — so
    unhinted files still de-warp. Returns (Reprojector or None, VRFormat)."""
    from ..reprojection import Reprojector
    from ..vr_format import detect

    if override is not None:
        rep, fmt = _forced_reprojector(override.get("format", ""),
                                       fov=override.get("fov", 100.0),
                                       pitch=override.get("pitch", 0.0),
                                       flat=override.get("flat", False))
        if rep is not None or override.get("flat"):
            return rep, fmt
        # an unparseable stored token falls through to normal detection

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
              assume: str | None = None, paths: list[str] | None = None,
              scene_timeout: float | None = None, invalidate: bool = False,
              watchdog=None) -> None:
    """Background task: embed videos (resumable). Scans ``media_root`` unless an
    explicit ``paths`` list is given (used for retrying just the failures, or a
    single-file QC re-embed). Each casualty is recorded in the failure log; a
    later success clears it.

    A stored QC :mod:`~peaks_vr.overrides` correction for a file wins over
    auto-detection (so fixes stick across runs). ``invalidate`` deletes each
    path's cache entry first, forcing a genuine re-embed instead of the resumable
    skip — set by the QC re-embed after the format is changed.

    ``scene_timeout`` is the per-scene sampling ceiling in seconds (0 disables).
    When None it comes from ``PEAKS_SCENE_TIMEOUT``, falling back to the
    :class:`FrameSampler` default — heavy 8K scenes need well over the old 180s."""
    import os

    from ..cache import EmbeddingCache, path_key
    from ..cli import iter_video_files, scene_from_path
    from ..embedding import get_embedder
    from ..overrides import overrides_for
    from ..pipeline import embed_library
    from ..sampling import FrameSampler

    if scene_timeout is None:
        env = os.environ.get("PEAKS_SCENE_TIMEOUT")
        scene_timeout = float(env) if env else FrameSampler().scene_timeout
    job.log(f"per-scene timeout: {scene_timeout:.0f}s"
            + (" (disabled)" if not scene_timeout else ""))

    videos = paths if paths is not None else iter_video_files([media_root])
    job.total = len(videos)
    where = "retry set" if paths is not None else media_root
    job.log(f"found {len(videos)} file(s) ({where})")
    if not videos:
        return
    embedder = get_embedder(model)
    cache = EmbeddingCache(cache_root)
    failures = failure_log_for(cache_root)
    overrides = overrides_for(cache_root)
    hw = "" if hwaccel in ("none", "cpu") else hwaccel
    for path in videos:
        if mgr.should_stop():
            job.log("stop requested — halting")
            break
        # RAM backpressure: never start a scene while at the memory cap. A hard
        # breach halts the run cleanly (resumable) instead of risking an OOM kill.
        if watchdog is not None and not watchdog.gate(job.log, mgr.should_stop):
            job.log("halted by RAM watchdog — resume after freeing headroom")
            break
        job.set_current(Path(path).name)
        if invalidate:  # drop the old vector so this isn't skipped as resumable
            cache.delete(path_key(path), embedder.name)
        ov = overrides.get(path)
        reproject = None
        if vr and ov and ov.get("flat"):
            job.log(f"  {job.current}: flat (de-warp off) per QC override")
        elif vr:
            probe = FrameSampler(hwaccel=hw)  # for probe_dimensions only
            reproject, fmt = _reprojector_for(path, assume=assume, sampler=probe,
                                              override=ov)
            if reproject is None:
                job.log(f"! {job.current}: VR format unknown (no hint, no assume) "
                        f"— skipping de-warp")
            elif ov:
                job.log(f"  {job.current}: format override → "
                        f"{fmt.projection.value}/{fmt.layout.value}/{fmt.fov_deg}")
        sampler = FrameSampler(interval_seconds=interval, mode="sparse",
                               hwaccel=hw, reproject=reproject,
                               scene_timeout=scene_timeout)
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
    import os

    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

    from . import auth as _auth
    from .jobs import JobManager

    from ..memwatch import MemoryWatchdog, limit_from_env

    app = FastAPI(title="peaks-vr")

    # --- active taste profile (switchable from the UI, persisted to /config) --
    # `profile` (from PEAKS_VR_PROFILE / default) is only the initial value; the
    # UI picker overrides it and the choice is saved beside the cache so it sticks
    # across restarts without touching env vars. Categories still nest under it as
    # <profile>:<category>.
    from ..pipeline import safe_tag
    _profile_path = Path(cache_root).parent / "active_profile"

    def _load_active() -> str:
        try:
            val = _profile_path.read_text().strip()
            return val or profile
        except OSError:
            return profile

    _active = {"name": _load_active()}

    def _profile() -> str:
        return _active["name"]

    def _set_active(name: str) -> str:
        name = safe_tag((name or "").strip()) or profile
        _active["name"] = name
        try:
            _profile_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = _profile_path.with_name(_profile_path.name + ".tmp")
            tmp.write_text(name)
            tmp.replace(_profile_path)
        except OSError:
            pass
        return name

    # --- optional password gate (opt-in via PEAKS_VR_PASSWORD) ---------------
    _password = os.environ.get("PEAKS_VR_PASSWORD") or ""
    _auth_on = bool(_password)
    _secret = _auth.load_secret(cache_root) if _auth_on else b""
    try:
        _timeout = int(os.environ.get("PEAKS_VR_SESSION_TIMEOUT", "3600"))
    except ValueError:
        _timeout = 3600
    # Endpoints the browser polls in the background: they require a valid session
    # but must NOT renew it, so "idle" means no *user* interaction (a left-open
    # tab still lapses). Everything else — and an explicit /api/ping on activity —
    # slides the expiry forward.
    _POLL_PATHS = {"/api/state", "/api/mem", "/api/embed/status"}
    _OPEN_PATHS = {"/login", "/api/login"}

    def _set_session(resp) -> None:
        value, max_age = _auth.make_cookie(_secret, _timeout)
        resp.set_cookie(_auth.COOKIE, value, max_age=max_age, httponly=True,
                        samesite="lax", path="/")

    if _auth_on:
        @app.middleware("http")
        async def _gate(request: Request, call_next):
            path = request.url.path
            if path in _OPEN_PATHS:
                return await call_next(request)
            if not _auth.verify(_secret, request.cookies.get(_auth.COOKIE)):
                if path.startswith("/api/"):
                    return Response('{"detail":"authentication required"}',
                                    status_code=401, media_type="application/json")
                return RedirectResponse("/login", status_code=303)
            resp = await call_next(request)
            if path not in _POLL_PATHS:   # slide the idle timeout on real use
                _set_session(resp)
            return resp

    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return HTMLResponse(_auth.LOGIN_PAGE)

    @app.post("/api/login")
    def login(body: LoginBody):
        if not _auth_on:
            return {"ok": True}                       # gate disabled — always in
        import time as _t
        if not _auth.check_password(_password, body.password):
            _t.sleep(0.5)                             # blunt brute-force guessing
            raise HTTPException(401, "incorrect password")
        resp = Response('{"ok":true}', media_type="application/json")
        _set_session(resp)
        return resp

    @app.post("/api/logout")
    def logout():
        resp = Response('{"ok":true}', media_type="application/json")
        resp.delete_cookie(_auth.COOKIE, path="/")
        return resp

    @app.post("/api/ping")
    def ping():
        # A valid session reached here (past the gate); the middleware renews it.
        return {"ok": True, "auth": _auth_on}
    # Persist the embed run's log/status beside the cache (same /config volume as
    # failures.json / overrides.json) so it survives a restart and shows on any
    # device's fresh load.
    jobs = jobs or JobManager(
        persist_path=Path(cache_root).parent / "embed_status.json")
    model_dir = _CANONICAL_MODEL.get(model, model)

    # Self-regulating RAM watchdog (default cap 24 GB, PEAKS_VR_MAX_RAM_GB). A
    # hard breach stops the running job cleanly (resumable) before the kernel
    # OOM-killer can crash the container mid-embed. Its log lines land in the
    # active job's log (visible in the UI) and on stderr (docker logs).
    def _wlog(msg: str) -> None:
        import sys
        print(f"[peaks-vr][mem] {msg}", file=sys.stderr)
        j = jobs.current_job
        if j is not None and jobs.running:
            j.log(msg)

    watchdog = MemoryWatchdog(limit_from_env(), on_trip=jobs.stop,
                              log=_wlog).start()

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
            store.add(key, float(t), 1, _profile(), scene_id=st.path)
        store.save()
        pos, neg = store.counts(_profile())
        return MarkResult(key=key, times=times, positives=pos,
                          total=pos + neg).__dict__

    @app.get("/api/marks")
    def marks(profile: str | None = None, limit: int = 50):
        prof = profile or _profile()
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
                "assume_default": assume_default, "profile": _profile(),
                "has_media": bool(media_root and Path(media_root).is_dir())}

    @app.get("/api/profiles")
    def profiles():
        """Pickable taste profiles: the base names that already have labels, plus
        the active one and the built-in default. Categories (``base:cat``) are
        collapsed to their base — they're managed inside the DJ taste tab."""
        bases = {p.split(":", 1)[0] for p in store.profiles()}
        bases |= {_profile(), profile}
        return {"active": _profile(), "profiles": sorted(bases)}

    @app.post("/api/profile")
    def set_profile(body: ProfileBody):
        """Switch the active taste profile (creating it if new). Persisted so it
        survives a restart. All marks / DJ-taste likes go to this profile from
        now on, and the recommender/DJ read from it."""
        return {"active": _set_active(body.profile)}

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

    @app.get("/api/qc")
    def qc(limit: int = 5000):
        """Every video with its QC status — embedded / failed / neither — so the
        UI can render a whole-library contact sheet. `embedded` reflects a cache
        hit under this model (same key the embedder writes: ``path_key``)."""
        from ..cache import EmbeddingCache
        from ..cli import iter_video_files
        from ..overrides import overrides_for
        if not media_root:
            return {"count": 0, "files": []}
        files = iter_video_files([media_root])
        try:
            embedded = set(EmbeddingCache(cache_root).keys(model_dir))
        except Exception:
            embedded = set()
        fails = {e.get("path"): e.get("error", "")
                 for e in failure_log_for(cache_root).entries() if e.get("path")}
        ovs = overrides_for(cache_root).all()
        out = []
        for f in files[:limit]:
            out.append({"name": Path(f).name, "path": f,
                        "embedded": path_key(f) in embedded,
                        "failed": f in fails, "error": fails.get(f, ""),
                        "override": ovs.get(f)})
        return {"count": len(files), "files": out}

    @app.get("/api/preview")
    def preview(path: str, time: float = 60.0, frac: float | None = None,
                fov: float = 100.0, pitch: float = 0.0, yaw: float = 0.0,
                hwaccel: str = "none", assume: str | None = None,
                format: str | None = None, flat: bool = False):
        from ..overrides import overrides_for
        from ..reprojection import Reprojector
        from ..sampling import FrameSampler
        from ..vr_format import detect

        real = _safe_under_media(path)
        hw = "" if hwaccel == "none" else hwaccel
        probe = FrameSampler(hwaccel=hw)

        # De-warp precedence: an explicit `format`/`flat` param (previewing a QC
        # candidate) > a stored override for this file > auto-detection. So QC
        # thumbnails (which pass neither) reflect saved corrections, and the fix
        # modal can preview a candidate before it's committed.
        rep = None
        if format is not None or flat:
            rep, fmt = _forced_reprojector(format or "", fov=fov, pitch=pitch,
                                           flat=flat)
            if rep is None and not flat:
                raise HTTPException(422, f"format {format!r} not recognized")
        else:
            ov = overrides_for(cache_root).get(path)
            if ov is not None:
                rep, fmt = _reprojector_for(real, sampler=probe, override=ov)
                flat = bool(ov.get("flat"))
            else:
                dims = probe.probe_dimensions(real)
                aspect = (dims[0] / dims[1]) if dims else None
                fmt = detect(Path(real).name, aspect_ratio=aspect,
                             assume=assume if assume is not None else assume_default)
                if not fmt.is_known:
                    raise HTTPException(422, f"VR format not recognized for "
                                        f"{Path(real).name} — set an 'assume' format")
                rep = Reprojector.for_format(fmt, viewport_fov_deg=fov,
                                             pitch=pitch, yaw=yaw)

        # A fraction of the file's duration lands a QC thumbnail mid-file
        # regardless of clip length (a fixed time can overrun a short clip).
        if frac is not None:
            try:
                time = max(0.0, min(float(frac), 0.999)) * probe.probe_duration(real)
            except Exception:
                pass  # unknown duration — fall back to the absolute `time`
        s = FrameSampler(hwaccel=hw, reproject=rep)  # rep None → raw (flat) crop
        try:
            img = s.grab_frame(real, time)
        except Exception as exc:
            raise HTTPException(503, f"could not render preview: {exc}")
        from io import BytesIO
        buf = BytesIO(); img.save(buf, format="JPEG", quality=90)
        tag = "flat/mono/—" if rep is None else (
            f"{fmt.projection.value}/{fmt.layout.value}/{fmt.fov_deg}")
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"X-VR-Format": f"{tag} ({fmt.source})"})

    @app.post("/api/embed/start")
    def embed_start(body: EmbedBody):
        if not media_root:
            raise HTTPException(400, "no media directory configured (mount /data)")
        try:
            jobs.start("embed", lambda job, mgr: embed_job(
                job, mgr, media_root=media_root, cache_root=cache_root,
                model=model, interval=body.interval, vr=body.vr,
                hwaccel=body.hwaccel, scene_timeout=body.scene_timeout,
                assume=body.assume if body.assume is not None else assume_default,
                watchdog=watchdog))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"started": True}

    @app.post("/api/embed/reembed")
    def embed_reembed(body: ReembedBody):
        """Re-embed one file with a corrected (or reset) de-warp. Stores the
        sticky override (or clears it for ``format == "auto"``), then runs a
        single-file embed with ``invalidate`` so the old vector is dropped and the
        file is genuinely re-processed."""
        if not media_root:
            raise HTTPException(400, "no media directory configured (mount /data)")
        from ..overrides import overrides_for
        real = _safe_under_media(body.path)  # validate it's inside the library
        ov = overrides_for(cache_root)
        flat = body.flat or body.format == "flat"
        if body.format == "auto" and not flat:
            ov.remove(body.path)
            note = "auto (re-detect)"
        else:
            if not flat:
                rep, _ = _forced_reprojector(body.format, fov=body.fov,
                                             pitch=body.pitch)
                if rep is None:
                    raise HTTPException(422, f"format {body.format!r} not recognized")
            ov.set(body.path, format="" if flat else body.format,
                   fov=body.fov, pitch=body.pitch, flat=flat)
            note = "flat (no de-warp)" if flat else body.format
        try:
            jobs.start("reembed", lambda job, mgr: embed_job(
                job, mgr, media_root=media_root, cache_root=cache_root,
                model=model, interval=8.0, vr=not flat, hwaccel="auto",
                assume=assume_default, paths=[body.path], invalidate=True,
                watchdog=watchdog))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"started": True, "format": note}

    @app.post("/api/embed/stop")
    def embed_stop():
        jobs.stop()
        return {"stopping": True}

    @app.get("/api/embed/status")
    def embed_status():
        snap = jobs.status()
        return {"job": snap, "embedded": _cache_count(), "running": jobs.running,
                "mem": watchdog.snapshot()}

    @app.get("/api/mem")
    def mem():
        return watchdog.snapshot()

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
                assume=assume_default, paths=paths, watchdog=watchdog))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"started": True, "count": len(paths)}

    @app.post("/api/failures/clear")
    def failures_clear():
        return {"cleared": failure_log_for(cache_root).clear()}

    # --- DJ taste profile: bulk-curate taste from the embedded library -------

    @app.get("/api/taste/suggest")
    def taste_suggest(count: int = 60, seed: int | None = None,
                      random: bool = False):
        """A batch of frames sampled across all embedded scenes to thumb-up.
        ``random=true`` forces pure random exploration (untagged "Load more");
        otherwise warm profiles get active learning. Excludes already-liked
        frames. Render each via /api/preview?path=&time=."""
        from ..cache import EmbeddingCache
        from .. import taste as _taste
        cache = EmbeddingCache(cache_root)
        frames = _taste.suggest_frames(store, cache, model_dir, _profile(),
                                       count=count, seed=seed, random=random)
        return {"count": len(frames), "base": _profile(), "frames": frames}

    @app.get("/api/taste/similar")
    def taste_similar(key: str, time: float, count: int = 60):
        """Frames across the library most similar to one anchor frame — the
        "more like this" query (query-by-example against the frame's vector)."""
        from ..cache import EmbeddingCache
        from .. import taste as _taste
        cache = EmbeddingCache(cache_root)
        frames = _taste.similar_frames(store, cache, model_dir, _profile(),
                                       key, time, count=count)
        return {"count": len(frames), "base": _profile(), "frames": frames}

    @app.post("/api/taste/like")
    def taste_like(body: TasteLikeBody):
        from .. import taste as _taste
        prof = _taste.like_frame(store, body.key, body.time, body.path,
                                 _profile(), body.category, body.label)
        store.save()
        pos, _ = store.counts(prof)
        return {"ok": True, "profile": prof, "count": pos}

    @app.post("/api/taste/unlike")
    def taste_unlike(body: TasteLikeBody):
        from .. import taste as _taste
        dropped = _taste.unlike_frame(store, body.key, body.time,
                                      _profile(), body.category)
        if dropped:
            store.save()
        return {"ok": True, "dropped": dropped}

    @app.get("/api/taste/summary")
    def taste_summary():
        from .. import taste as _taste
        return _taste.taste_summary(store, _profile())

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
