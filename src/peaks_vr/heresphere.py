"""HereSphere remote control — the linchpin (README, "The linchpin").

Almost everything novel in peaks-vr depends on what an external program can
**read from** and **command on** the HereSphere headset player in real time:

  * Report the **currently-playing file + timecode** live → gates real-time
    moment flagging (feature #2).
  * Accept **seek / load-next** commands → gates the "VR DJ" (feature #4).

Both are answered **yes** by the protocol HereSphere speaks: it is
**DeoVR remote-control compatible** (see ``docs/HERESPHERE_API.md`` for the
Phase-0 research and sources). The wire protocol, confirmed across the DeoVR
docs and several open-source clients:

  * **Transport:** a plain **TCP** socket on port ``23554`` (DeoVR default;
    confirm HereSphere's port on-device with ``peaks-vr probe``).
  * **Framing:** every message is a **4-byte big-endian length** prefix followed
    by that many bytes of **UTF-8 JSON**. A length of ``0`` is a keep-alive ping
    with no payload.
  * **Player → client, ~1 Hz:**
    ``{"path": "...", "duration": 123.45, "currentTime": 10.5,
       "playbackSpeed": 1.0, "playerState": 0}`` — ``playerState`` 0 = playing,
    1 = paused.
  * **Client → player:** send the *same* framed JSON; ``path`` loads a file,
    ``currentTime`` seeks, ``playbackSpeed`` sets speed, ``playerState``
    plays/pauses.
  * **Keep-alive:** the client must send a packet at least every ~1 s or the
    player drops the connection after ~3 s of silence.

The framing helpers (:func:`encode_frame`, :class:`FrameDecoder`) are pure and
unit-tested offline; only :class:`RemoteClient` touches the network.

Two knobs stay configurable because they can only be pinned down on real
hardware: the ``port`` and the length-prefix ``byteorder`` (big-endian is the
documented default; ``peaks-vr probe`` flags a mismatch).
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterator

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 23554  # DeoVR remote-control default; confirm for HereSphere
PING_INTERVAL = 1.0   # seconds; player drops us after ~3s of silence

# playerState enum (DeoVR remote protocol)
STATE_PLAYING = 0
STATE_PAUSED = 1


# --- framing (pure, offline-testable) ---------------------------------------

def encode_frame(payload: dict | None, *, byteorder: str = "big") -> bytes:
    """Serialize one protocol message: 4-byte length prefix + UTF-8 JSON.

    ``payload=None`` (or ``{}``-less falsy) produces a **keep-alive ping**: a
    bare ``length=0`` prefix with no body, exactly what the player expects once
    a second to hold the connection open.
    """
    if not payload:
        return struct.pack(_len_fmt(byteorder), 0)
    body = json.dumps(payload).encode("utf-8")
    return struct.pack(_len_fmt(byteorder), len(body)) + body


def _len_fmt(byteorder: str) -> str:
    """struct format for the 4-byte unsigned length prefix."""
    if byteorder == "big":
        return ">I"
    if byteorder == "little":
        return "<I"
    raise ValueError(f"byteorder must be 'big' or 'little', got {byteorder!r}")


class FrameDecoder:
    """Accumulates raw TCP bytes and yields complete decoded messages.

    TCP is a byte stream: one ``recv`` can split a frame or glue several
    together. Feed whatever arrives to :meth:`feed`; it yields one item per
    complete frame — a parsed ``dict`` for a status packet, or ``None`` for a
    ``length=0`` keep-alive ping — and buffers any partial remainder.
    """

    def __init__(self, *, byteorder: str = "big"):
        self._buf = bytearray()
        self._fmt = _len_fmt(byteorder)

    def feed(self, data: bytes) -> Iterator[dict | None]:
        self._buf.extend(data)
        while True:
            if len(self._buf) < 4:
                return
            (length,) = struct.unpack(self._fmt, self._buf[:4])
            if length == 0:  # keep-alive ping
                del self._buf[:4]
                yield None
                continue
            if len(self._buf) < 4 + length:
                return  # payload not fully arrived yet
            body = bytes(self._buf[4 : 4 + length])
            del self._buf[: 4 + length]
            yield json.loads(body.decode("utf-8"))


# --- state ------------------------------------------------------------------

@dataclass(frozen=True)
class PlaybackState:
    """A snapshot of what the headset is playing right now.

    ``path`` is the source file HereSphere reports (used to look up the cached,
    de-warped scene); ``current_time`` is the playhead in seconds (the anchor
    for a ❤️ mark); ``playing`` distinguishes play from pause.
    """

    path: str | None
    current_time: float
    playing: bool
    duration: float | None = None
    playback_speed: float = 1.0

    @classmethod
    def from_packet(cls, d: dict) -> "PlaybackState":
        return cls(
            path=d.get("path") or None,
            current_time=float(d.get("currentTime") or 0.0),
            playing=int(d.get("playerState", STATE_PLAYING)) == STATE_PLAYING,
            duration=(float(d["duration"]) if d.get("duration") is not None else None),
            playback_speed=float(d.get("playbackSpeed") or 1.0),
        )


class RemoteError(RuntimeError):
    """Raised when the remote channel can't be reached or speaks unexpectedly."""


# --- client -----------------------------------------------------------------

class RemoteClient:
    """DeoVR/HereSphere remote-control client over TCP.

    Read side: :meth:`read_state` returns the latest snapshot; :meth:`monitor`
    yields a :class:`PlaybackState` per status packet (what the flagging UI
    consumes). Control side: :meth:`seek`, :meth:`load`, :meth:`play`,
    :meth:`pause`. A background thread sends a keep-alive ping every second so
    the player doesn't drop the connection.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        byteorder: str = "big",
        timeout: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.byteorder = byteorder
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._decoder = FrameDecoder(byteorder=byteorder)
        self._latest: PlaybackState | None = None
        self._send_lock = threading.Lock()
        self._pinger: threading.Thread | None = None
        self._stop = threading.Event()

    # --- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        """Open the TCP channel and start the keep-alive pinger."""
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
        except OSError as exc:
            raise RemoteError(
                f"could not connect to HereSphere/DeoVR remote at "
                f"{self.host}:{self.port} — is the player running with remote "
                f"control enabled? ({exc})"
            ) from exc
        self._stop.clear()
        self._pinger = threading.Thread(target=self._ping_loop, daemon=True)
        self._pinger.start()

    def close(self) -> None:
        """Stop the pinger and close the socket. Safe to call more than once."""
        self._stop.set()
        if self._pinger is not None:
            self._pinger.join(timeout=PING_INTERVAL * 2)
            self._pinger = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _ping_loop(self) -> None:
        ping = encode_frame(None, byteorder=self.byteorder)
        while not self._stop.wait(PING_INTERVAL):
            try:
                self._send_raw(ping)
            except OSError:
                return  # socket gone; monitor()/read_state will surface it

    # --- send ---------------------------------------------------------------

    def _send_raw(self, data: bytes) -> None:
        if self._sock is None:
            raise RemoteError("not connected — call connect() first")
        with self._send_lock:
            self._sock.sendall(data)

    def _send(self, payload: dict) -> None:
        self._send_raw(encode_frame(payload, byteorder=self.byteorder))

    # --- read (gates real-time flagging) ------------------------------------

    def monitor(self) -> Iterator[PlaybackState]:
        """Yield a :class:`PlaybackState` for each status packet as it arrives.

        Blocks between packets (the player emits ~1/s). Keep-alive pings from
        the player are consumed silently. Ends when the connection closes.
        """
        if self._sock is None:
            raise RemoteError("not connected — call connect() first")
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                raise RemoteError(f"remote connection lost: {exc}") from exc
            if not chunk:
                return  # peer closed
            for msg in self._decoder.feed(chunk):
                if msg is None:
                    continue  # keep-alive ping
                state = PlaybackState.from_packet(msg)
                self._latest = state
                yield state

    def read_state(self, *, wait: bool = True) -> PlaybackState:
        """Return the current :class:`PlaybackState`.

        With ``wait`` (default), blocks for the next status packet if none has
        arrived yet; otherwise returns the last-seen snapshot or raises if there
        is none.
        """
        if self._latest is not None and not wait:
            return self._latest
        for state in self.monitor():
            return state
        raise RemoteError("connection closed before any status packet arrived")

    # --- control (gates DJ playback) ----------------------------------------

    def seek(self, seconds: float) -> None:
        """Seek the current file to ``seconds`` (the DJ jumping to a moment)."""
        self._send({"currentTime": float(seconds)})

    def load(self, path: str, *, start: float = 0.0) -> None:
        """Load ``path`` and begin at ``start`` seconds. Each moment plays its
        own source file in its native projection, so HereSphere re-detects
        format per clip — which dissolves the mixed-format problem (README
        #4/#6)."""
        payload: dict = {"path": path}
        if start:
            payload["currentTime"] = float(start)
        self._send(payload)

    def play(self) -> None:
        self._send({"playerState": STATE_PLAYING})

    def pause(self) -> None:
        self._send({"playerState": STATE_PAUSED})

    def set_speed(self, speed: float) -> None:
        self._send({"playbackSpeed": float(speed)})

    # --- context-manager sugar ---------------------------------------------

    def __enter__(self) -> "RemoteClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
