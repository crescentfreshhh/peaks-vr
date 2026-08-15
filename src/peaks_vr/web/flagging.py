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
DEFAULT_WEB_PORT = 8760

# The request body must be a MODULE-LEVEL pydantic model or FastAPI resolves it
# as query params instead of a body (a locally-defined class fails hint
# resolution). Guarded so importing this module doesn't require the [web] extra.
try:
    from pydantic import BaseModel

    class MarkBody(BaseModel):
        time: float
        in_time: float | None = None
        out_time: float | None = None
except ImportError:  # pragma: no cover - only when fastapi/pydantic absent
    MarkBody = None  # type: ignore


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


def create_app(mirror: PlaybackMirror, store: LabelStore, *,
               profile: str = DEFAULT_PROFILE, sampler=None):
    """Build the FastAPI app. ``sampler`` (a configured
    :class:`peaks_vr.sampling.FrameSampler`, optionally with a ``reproject``)
    enables the live de-warped preview; without it ``/api/frame`` returns 503."""
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import FileResponse

    app = FastAPI(title="peaks-vr flagging")
    active_profile = profile

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

    return app


# --- entry points -----------------------------------------------------------

def _serve(app, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def run(hs_host: str | None = None, hs_port: int = 23554, *,
        listen: bool = False, ts_port: int = 23573,
        web_host: str = "0.0.0.0", web_port: int = DEFAULT_WEB_PORT,
        labels_path: str = "labels.json", profile: str = DEFAULT_PROFILE,
        byteorder: str = "big", sampler=None) -> None:
    """Run the flagging UI against a real HereSphere headset.

    Two ways to source playback: ``listen=True`` receives HereSphere's
    **timestamp server** push on ``ts_port`` (the headset connects to us — the
    container default); otherwise we dial the headset's DeoVR remote at
    ``hs_host:hs_port``.
    """
    mirror = PlaybackMirror()
    if listen:
        mirror.start_feeder(timestamp_feeder(ts_port, byteorder=byteorder))
        source = f"timestamp server :{ts_port} (HereSphere connects in)"
    else:
        mirror.start_feeder(remote_feeder(hs_host, hs_port, byteorder=byteorder))
        source = f"remote {hs_host}:{hs_port}"
    store = LabelStore(labels_path)
    app = create_app(mirror, store, profile=profile, sampler=sampler)
    print(f"peaks-vr flagging UI → http://{web_host}:{web_port}  "
          f"(source: {source}, profile '{profile}')")
    _serve(app, web_host, web_port)


def run_demo(*, web_host: str = "127.0.0.1", web_port: int = DEFAULT_WEB_PORT,
             labels_path: str = "labels.demo.json",
             profile: str = DEFAULT_PROFILE) -> None:
    """Run the flagging UI with a synthetic feed — no headset required."""
    mirror = PlaybackMirror()
    mirror.start_feeder(demo_feeder())
    store = LabelStore(labels_path)
    app = create_app(mirror, store, profile=profile)
    print(f"peaks-vr flagging DEMO → http://{web_host}:{web_port}  "
          f"(synthetic playback, marks → {labels_path})")
    _serve(app, web_host, web_port)
