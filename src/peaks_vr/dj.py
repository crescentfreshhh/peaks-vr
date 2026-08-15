"""The VR DJ (README feature #4).

Plays a :class:`peaks_vr.recommend.Playlist` back-to-back in the headset by
driving HereSphere over the remote API (feature 0's :class:`RemoteClient`): for
each moment it **loads that moment's own source file** at the clip's start, lets
it play for the clip window, then commands the next one.

The key idea from the README: because each moment plays its native file,
HereSphere **re-detects projection per clip**, so any mix of 180/360, SBS/TB,
fisheye/equirect can sit back-to-back with only a brief reload between clips —
no transcoding, no format unification (#6). Consecutive moments from the *same*
file are a cheap seek instead of a reload.

Advancing is driven by wall-clock time over the clip window. The client and the
sleeper are injectable, so the whole sequencer is unit-tested with a fake player
and instant time — no headset needed.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from .recommend import Moment, Playlist

# small gap so HereSphere has a beat to load/re-detect the next file's format
RELOAD_PAD = 0.5


class _Player(Protocol):
    """The slice of RemoteClient the DJ needs (so tests can fake it)."""
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def load(self, path: str, *, start: float = 0.0) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def play(self) -> None: ...


class DJ:
    """Sequences a playlist through a headset player."""

    def __init__(self, player: _Player, *,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print):
        self.player = player
        self._sleep = sleep
        self._log = log

    def _cue(self, m: Moment, prev: Moment | None) -> None:
        """Start one moment: seek if it's the same file as the last, else load."""
        same_file = prev is not None and m.path == prev.path and m.path
        if same_file:
            self.player.seek(m.start)
        else:
            self.player.load(m.path, start=m.start)
        self.player.play()

    def play(self, playlist: Playlist, *, loop: bool = False,
             min_clip: float = 2.0) -> int:
        """Play through ``playlist``. Returns how many moments were played.

        Each moment holds for at least ``min_clip`` seconds (so a tiny segment
        isn't a blink), plus a small pad for the format reload between files.
        ``loop`` repeats until interrupted.
        """
        if not playlist.moments:
            self._log("  (empty playlist — nothing to play)")
            return 0
        played = 0
        prev: Moment | None = None
        try:
            while True:
                for m in playlist.moments:
                    name = (m.path or m.key).split("/")[-1].split("\\")[-1]
                    self._log(f"  ▶ {name}  {m.start:.1f}–{m.end:.1f}s "
                              f"(score {m.score:.3f})")
                    self._cue(m, prev)
                    hold = max(m.duration, min_clip)
                    if prev is None or m.path != prev.path:
                        hold += RELOAD_PAD
                    self._sleep(hold)
                    prev = m
                    played += 1
                if not loop:
                    break
        except KeyboardInterrupt:
            self._log(f"\n  ⏹ stopped after {played} moment(s)")
        return played

    def dry_run(self, playlist: Playlist) -> None:
        """Print the sequence without touching a player — a preview of the set."""
        total = sum(max(m.duration, 0.0) for m in playlist.moments)
        self._log(f"playlist '{playlist.profile}': {len(playlist.moments)} "
                  f"moments, ~{total:.0f}s")
        for i, m in enumerate(playlist.moments, 1):
            name = (m.path or m.key).split("/")[-1].split("\\")[-1]
            self._log(f"  {i:>3}. {name}  {m.start:7.1f}–{m.end:7.1f}s "
                      f"({m.duration:4.1f}s, score {m.score:.3f})")


def play_playlist(playlist: Playlist, host: str, port: int = 23554, *,
                  byteorder: str = "big", loop: bool = False,
                  log: Callable[[str], None] = print) -> int:
    """Connect to a real headset and play the playlist. Convenience wrapper."""
    from .heresphere import RemoteClient

    client = RemoteClient(host, port, byteorder=byteorder)
    client.connect()
    try:
        return DJ(client, log=log).play(playlist, loop=loop)
    finally:
        client.close()
