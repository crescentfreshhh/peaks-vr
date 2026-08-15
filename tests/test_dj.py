"""Offline tests for the VR DJ (#4): drive a fake player through a playlist and
assert the command sequence — same-file moments seek, new files load. No headset,
instant time."""

from peaks_vr.dj import DJ
from peaks_vr.recommend import Moment, Playlist


class FakePlayer:
    """Records the control calls the DJ makes, in order."""

    def __init__(self):
        self.calls = []

    def connect(self):
        self.calls.append(("connect",))

    def close(self):
        self.calls.append(("close",))

    def load(self, path, *, start=0.0):
        self.calls.append(("load", path, start))

    def seek(self, seconds):
        self.calls.append(("seek", seconds))

    def play(self):
        self.calls.append(("play",))


def _playlist():
    return Playlist(profile="apex", moments=[
        Moment(path="D:/vr/A.mp4", key="kA", start=0.0, end=8.0, score=0.9),
        Moment(path="D:/vr/A.mp4", key="kA", start=20.0, end=28.0, score=0.8),
        Moment(path="D:/vr/B.mp4", key="kB", start=5.0, end=11.0, score=0.7),
    ])


def test_dj_loads_new_files_and_seeks_within_a_file():
    player = FakePlayer()
    dj = DJ(player, sleep=lambda s: None)   # instant
    played = dj.play(_playlist())

    assert played == 3
    assert player.calls == [
        ("load", "D:/vr/A.mp4", 0.0), ("play",),   # first moment: load A
        ("seek", 20.0), ("play",),                 # same file → seek
        ("load", "D:/vr/B.mp4", 5.0), ("play",),   # new file → load B
    ]


def test_dj_hold_time_respects_window_and_min_clip():
    player = FakePlayer()
    slept = []
    dj = DJ(player, sleep=slept.append)
    dj.play(Playlist(profile="apex", moments=[
        Moment(path="A", key="kA", start=0.0, end=1.0, score=1.0),   # 1s < min
        Moment(path="A", key="kA", start=5.0, end=15.0, score=1.0),  # 10s window
    ]), min_clip=2.0)
    # first clip padded up to min_clip (2s) + reload pad; second is its 10s window
    assert slept[0] >= 2.0
    assert slept[1] >= 10.0


def test_empty_playlist_plays_nothing():
    player = FakePlayer()
    assert DJ(player, sleep=lambda s: None).play(Playlist(profile="apex")) == 0
    assert player.calls == []


def test_dry_run_prints_without_touching_player(capsys):
    DJ(player=None).dry_run(_playlist())
    out = capsys.readouterr().out
    assert "3 moments" in out
    assert "A.mp4" in out and "B.mp4" in out
