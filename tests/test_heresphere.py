"""Offline tests for the HereSphere/DeoVR remote framing and state parsing.

Pure protocol logic — no sockets, no headset. Covers the load-bearing bits:
the 4-byte length-prefixed JSON framing (including partial reads and keep-alive
pings) and mapping a status packet onto PlaybackState."""

import json
import struct

import pytest

from peaks_vr.heresphere import (
    FrameDecoder,
    PlaybackState,
    encode_frame,
)

SAMPLE = {
    "path": "D:/vr/scene_180_sbs.mp4",
    "duration": 123.45,
    "currentTime": 10.5,
    "playbackSpeed": 1.0,
    "playerState": 0,
}


# --- framing ----------------------------------------------------------------

def test_encode_frame_is_length_prefixed_json():
    frame = encode_frame(SAMPLE)
    (length,) = struct.unpack(">I", frame[:4])
    assert length == len(frame) - 4
    assert json.loads(frame[4:].decode("utf-8")) == SAMPLE


def test_encode_none_is_zero_length_ping():
    assert encode_frame(None) == struct.pack(">I", 0)
    assert encode_frame({}) == struct.pack(">I", 0)  # falsy → ping too


def test_roundtrip_single_frame():
    dec = FrameDecoder()
    out = list(dec.feed(encode_frame(SAMPLE)))
    assert out == [SAMPLE]


def test_multiple_frames_in_one_chunk():
    dec = FrameDecoder()
    blob = encode_frame(SAMPLE) + encode_frame({"currentTime": 5.0})
    assert list(dec.feed(blob)) == [SAMPLE, {"currentTime": 5.0}]


def test_partial_reads_are_buffered():
    dec = FrameDecoder()
    frame = encode_frame(SAMPLE)
    # split mid-payload across two recv()s, plus a split inside the length prefix
    assert list(dec.feed(frame[:2])) == []          # length not even complete
    assert list(dec.feed(frame[2:10])) == []         # payload not complete
    assert list(dec.feed(frame[10:])) == [SAMPLE]    # now it completes


def test_zero_length_ping_decodes_to_none():
    dec = FrameDecoder()
    stream = encode_frame(None) + encode_frame(SAMPLE)
    assert list(dec.feed(stream)) == [None, SAMPLE]


def test_little_endian_roundtrip():
    dec = FrameDecoder(byteorder="little")
    assert list(dec.feed(encode_frame(SAMPLE, byteorder="little"))) == [SAMPLE]


def test_bad_byteorder_rejected():
    with pytest.raises(ValueError):
        encode_frame(SAMPLE, byteorder="middle")


# --- state parsing ----------------------------------------------------------

def test_playbackstate_from_playing_packet():
    st = PlaybackState.from_packet(SAMPLE)
    assert st.path == "D:/vr/scene_180_sbs.mp4"
    assert st.current_time == 10.5
    assert st.duration == 123.45
    assert st.playing is True


def test_playbackstate_paused_and_missing_fields():
    st = PlaybackState.from_packet({"playerState": 1})
    assert st.playing is False        # 1 == paused
    assert st.path is None            # missing → None
    assert st.current_time == 0.0     # missing → 0
    assert st.duration is None


def test_playbackstate_reads_resource_field():
    # HereSphere's timestamp server reports the file as "resource", not "path"
    st = PlaybackState.from_packet({"resource": "Z:/vr/x.mp4", "currentTime": 3.0})
    assert st.path == "Z:/vr/x.mp4" and st.current_time == 3.0


# --- timestamp receiver (HereSphere connects to us) -------------------------

def _push(port, payloads, *, framed=True):
    import socket
    import time as _t

    _t.sleep(0.15)
    c = socket.create_connection(("127.0.0.1", port))
    for p in payloads:
        c.sendall(encode_frame(p) if framed else (json.dumps(p) + "\n").encode())
        _t.sleep(0.03)
    _t.sleep(0.2)
    c.close()


def test_timestamp_receiver_decodes_framed_push():
    import threading

    from peaks_vr.heresphere import TimestampReceiver

    rx = TimestampReceiver(port=0)
    rx.bind()
    port = rx._server.getsockname()[1]
    threading.Thread(target=_push, args=(port, [
        {"resource": "Z:/a_180_sbs.mp4", "currentTime": 10.0, "playerState": 0},
        {"resource": "Z:/a_180_sbs.mp4", "currentTime": 12.0, "playerState": 1},
    ]), daemon=True).start()

    ip = rx.accept(timeout=3)
    assert ip  # HereSphere's IP
    states = []
    for st in rx.monitor():
        states.append(st)
        if len(states) >= 2:
            break
    rx.close()
    assert states[0].path == "Z:/a_180_sbs.mp4"
    assert states[0].current_time == 10.0 and states[0].playing is True
    assert states[1].playing is False


def test_timestamp_receiver_decodes_raw_json_push():
    import threading

    from peaks_vr.heresphere import TimestampReceiver

    rx = TimestampReceiver(port=0)
    rx.bind()
    port = rx._server.getsockname()[1]
    threading.Thread(target=_push, args=(port, [
        {"resource": "Z:/b_MKX200.mp4", "currentTime": 5.5, "playerState": 0},
    ]), kwargs={"framed": False}, daemon=True).start()

    rx.accept(timeout=3)
    got = next(iter(rx.monitor()))
    rx.close()
    assert got.path == "Z:/b_MKX200.mp4" and got.current_time == 5.5
