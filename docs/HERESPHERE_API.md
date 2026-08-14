# HereSphere API — Phase 0 research

> **Status: to be filled in.** This is the Phase-0 (no-code) research template.
> Everything novel in peaks-vr branches on the answers here, so pin them down
> before building the flagging UI (#2) or the DJ (#4). See the README's
> "The linchpin" section.

The load-bearing questions:

1. **Can an external program read the currently-playing file + timecode, live?**
   → gates real-time moment flagging.
2. **Can it accept seek / load-next commands?**
   → gates sequential moment playback (the "VR DJ").

HereSphere exposes an HTTP API (how it ingests libraries) and is broadly
**DeoVR remote-control compatible** — DeoVR's remote is a documented WebSocket
that streams `currentTime` / `path` / `playing` and accepts seek/play commands.
The goal of this doc is to confirm the exact surface against the real app.

The scaffold client that these findings complete lives in
[`src/peaks_vr/heresphere.py`](../src/peaks_vr/heresphere.py).

---

## Read surface (gates real-time flagging)

- [ ] Transport (WebSocket? HTTP polling?) and endpoint / port
      _(DeoVR default is `ws://<host>:23554`; confirm for HereSphere)_
- [ ] How is the **current file path** reported? Field name, format, absolute vs
      library-relative?
- [ ] How is the **current timecode** reported? Field name, units, update cadence?
- [ ] Is `playing` / paused state exposed?
- [ ] Latency / update frequency of the stream (matters for reaction-lag scrub).

**Findings:**

```
(paste observed messages / request+response here)
```

## Control surface (gates DJ playback)

- [ ] **Seek** within the current file — command shape, units.
- [ ] **Load** a specific file (and start offset) — command shape. Confirms each
      moment can play its own source in its native projection (README #4/#6).
- [ ] Play / pause commands.
- [ ] Any acknowledgement / error responses to commands?

**Findings:**

```
(paste observed commands / responses here)
```

## Fallback: if live current-time is NOT externally readable

Real-time flagging needs a way to anchor a ❤️ mark to (scene, time). If the
playhead can't be read live:

- [ ] Can peaks-vr **drop a HereSphere bookmark** at "now" and **read it back**
      (with its timecode) shortly after?
- [ ] Bookmark write path, read path, and the lag involved.

This is the fallback `RemoteClient.read_state` would select. Document it here
so the client can implement the right path.

**Findings:**

```
(notes)
```

---

## References

- DeoVR remote-control protocol (WebSocket `currentTime`/`path`/`playing`).
- HereSphere's HTTP library/API documentation.
- `stash-vr` — for how it talks to the same ecosystem.
