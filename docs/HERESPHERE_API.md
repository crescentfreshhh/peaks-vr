# HereSphere API — Phase 0 research

> **Status: researched from documentation + open-source clients; awaiting one
> on-device confirmation run.** This is the README's "linchpin": everything
> novel in peaks-vr depends on reading the currently-playing file + timecode and
> sending seek / load-next commands to HereSphere. Both are **supported**.

The two load-bearing questions and their answers:

1. **Can an external program read the currently-playing file + timecode, live?**
   → **Yes.** The player streams a status packet ~once per second containing
   `path` and `currentTime`. Gates real-time moment flagging (#2).
2. **Can it accept seek / load-next commands?**
   → **Yes.** The client sends the same packet shape back with `currentTime`
   (seek) or `path` (load). Gates sequential moment playback, the "VR DJ" (#4).

Because live current-time **is** externally readable, the bookmark fallback the
README worried about is **not needed**.

HereSphere is **DeoVR remote-control compatible**, so it speaks the DeoVR remote
protocol below. The implementation of this spec lives in
[`src/peaks_vr/heresphere.py`](../src/peaks_vr/heresphere.py); the on-device
check is [`peaks-vr probe`](../src/peaks_vr/cli.py).

---

## The remote-control protocol (DeoVR-compatible)

### Transport
- **Plain TCP** to the player, default port **`23554`**.
- The player IP is shown in its network/WiFi settings.

### Framing
Every message is a **4-byte unsigned length prefix** (big-endian — the
documented convention) followed by that many bytes of **UTF-8 JSON**.

```
[ 4 bytes: length N ][ N bytes: UTF-8 JSON ]
```

A prefix of **`0`** (no body) is a **keep-alive ping**.

### Keep-alive
Both sides send a packet at least **every ~1 second**. If the player receives
nothing for **~3 seconds**, it closes the connection. A `length=0` ping counts.
(`RemoteClient` runs a background pinger every 1s.)

### Player → client: status packet (~1 Hz)
```json
{
  "path": "D:/vr/scene_180_sbs.mp4",
  "duration": 123.45,
  "currentTime": 10.5,
  "playbackSpeed": 1.0,
  "playerState": 0
}
```
| field           | meaning                                             |
| --------------- | --------------------------------------------------- |
| `path`          | file the headset is playing → look up cached scene  |
| `currentTime`   | playhead in seconds → the ❤️ mark anchor            |
| `duration`      | total length in seconds                             |
| `playbackSpeed` | playback rate (1.0 = normal)                        |
| `playerState`   | **0 = playing, 1 = paused**                         |

### Client → player: commands
Send the **same framed JSON**; include only the fields you want to change:
| goal                | payload                          |
| ------------------- | -------------------------------- |
| seek                | `{"currentTime": 42.0}`          |
| load a file (DJ)    | `{"path": "...", "currentTime": 0}` |
| play / pause        | `{"playerState": 0}` / `{"playerState": 1}` |
| change speed        | `{"playbackSpeed": 1.5}`         |

Loading a new `path` is what makes the DJ work: **each moment plays its own
source file in its native projection**, so HereSphere re-detects format per clip
and any mix of projections can play back-to-back (README #4/#6).

---

## Headset setup (one-time)
1. In HereSphere, enable **remote control** (the DeoVR-compatible remote server).
2. Note the headset's **IP address** (network settings).
3. Make sure the headset and the computer running peaks-vr are on the **same
   network** and the port isn't firewalled.

---

## Confirm on device (one command)
With HereSphere **playing a video** and remote control enabled:

```bash
peaks-vr probe --host <headset-ip>
```
Live `path` + `currentTime` lines scrolling by confirm the **read** surface.
Add a control test:
```bash
peaks-vr probe --host <headset-ip> --test-seek 30
```
and watch the headset jump to 30s to confirm the **control** surface.

### Checklist — confirmed vs. still to verify on device
- [x] Transport = TCP, length-prefixed UTF-8 JSON (from docs + clients)
- [x] Status packet streams `path` + `currentTime` live (read surface)
- [x] `currentTime` / `path` commands seek / load (control surface)
- [ ] **Exact port** on HereSphere (23554 assumed) — confirm via `probe`
- [ ] **Length-prefix endianness** (big-endian assumed) — if `probe` sees no
      packets, retry `--byteorder little`
- [ ] Remote-control toggle location in the current HereSphere build

Paste the `probe` output back and these get checked off.

---

## Sources
- [DeoVR adds remote control API](https://deovr.com/blog/7-deovr-adds-remote-control-api)
  (protocol announcement) — full spec at `https://deovr.com/app/doc#remote-control`.
- [xbvr issue #288 — Support DeoVR remote control protocol](https://github.com/xbapps/xbvr/issues/288)
  (port 23554, length-prefixed JSON, keep-alive timing).
- [philpw99/DeoVR-Remote](https://github.com/philpw99/DeoVR-Remote) — a working
  TCP remote client.
- [ecal-mid/DEOVR-Remote-Control](https://github.com/ecal-mid/DEOVR-Remote-Control)
  — status fields + command set.
