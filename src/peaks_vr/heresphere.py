"""HereSphere remote control — the linchpin (README, "The linchpin").

Almost everything novel in peaks-vr depends on what an external program can
**read from** and **command on** the HereSphere headset player in real time:

  * Can it report the **currently-playing file + timecode** live?
    → gates real-time moment flagging (feature #2).
  * Can it accept **seek / load-next** commands?
    → gates sequential moment playback, the "VR DJ" (feature #4).

HereSphere exposes an HTTP API (it's how it ingests video libraries) and is
broadly **DeoVR remote-control compatible** — DeoVR's remote is a documented
WebSocket that streams ``currentTime`` / ``path`` / ``playing`` and accepts
seek/play commands. **Confirming this read + control surface is Phase 0** (see
``docs/HERESPHERE_API.md``); it is the load-bearing wall, and the exact wire
format is what the research phase pins down before this client is finished.

If live current-time turns out not to be externally readable, real-time
flagging needs a fallback — e.g. drop a HereSphere bookmark and read it back.
:meth:`RemoteClient.read_state` is where that fallback would be selected.

Status: scaffold. The interface below reflects the DeoVR remote's shape so the
flagging UI and DJ can be built against it; the transport is wired once Phase 0
confirms the protocol. Requires the ``[web]`` extra (``websockets``) at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 23554  # DeoVR remote-control default; confirm for HereSphere


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


class RemoteError(RuntimeError):
    """Raised when the remote channel can't be reached or speaks unexpectedly."""


class RemoteClient:
    """Client for HereSphere's DeoVR-compatible remote-control channel.

    The surface mirrors what the features need: a live read (:meth:`read_state`)
    and two commands (:meth:`seek`, :meth:`load`). Implementations land after
    Phase 0 confirms the transport; until then every I/O method raises
    :class:`NotImplementedError` so callers fail loudly rather than silently.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port

    # --- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        """Open the remote-control channel (WebSocket, per DeoVR). TODO: wire the
        ``websockets`` transport once Phase 0 confirms the handshake."""
        raise NotImplementedError("HereSphere transport pending Phase 0 research")

    def close(self) -> None:
        """Close the channel. Safe to call when never connected."""
        # No transport yet; nothing to release.
        return None

    # --- read (gates real-time flagging) ------------------------------------

    def read_state(self) -> PlaybackState:
        """Return the current :class:`PlaybackState`.

        Primary path: read the streamed ``currentTime``/``path``/``playing``
        fields. Fallback (if live time isn't externally readable): drop and
        re-read a HereSphere bookmark. Which path is used is decided here once
        Phase 0 resolves the capability."""
        raise NotImplementedError("read_state pending Phase 0 research")

    # --- control (gates DJ playback) ----------------------------------------

    def seek(self, seconds: float) -> None:
        """Seek the current file to ``seconds`` (the DJ jumping to a moment)."""
        raise NotImplementedError("seek pending Phase 0 research")

    def load(self, path: str, *, start: float = 0.0) -> None:
        """Load ``path`` and begin at ``start`` seconds. Each moment plays its
        own source file in its native projection, so HereSphere re-detects
        format per clip — which is what dissolves the mixed-format problem
        (README #4/#6)."""
        raise NotImplementedError("load pending Phase 0 research")

    # --- context-manager sugar ---------------------------------------------

    def __enter__(self) -> "RemoteClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
