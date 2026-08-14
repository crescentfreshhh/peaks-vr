# peaks-vr

**Taste-driven moment discovery for VR video.** A standalone sibling to
[peaks](../peaks) (the 2D taste engine for Stash): peaks-vr learns which *moments* inside your VR
library you actually like, finds more like them, and plays them back in the headset — without you
ever leaving VR to hunt.

> Status: **design draft.** This document seeds the repo; nothing here is built yet.

---

## Why a separate program

VR is different enough from flat video that entangling it with 2D-peaks would make both worse:

- VR frames are **stereo + fisheye/equirectangular** — they must be de-warped before any vision model
  can understand them (see [The core problem](#the-core-problem-vr-frames-break-vision-models)).
- Playback lives in a **headset player** (HereSphere), not a browser grid.
- Storage, trimming, and format handling have VR-specific concerns.
- The taste model should be **VR-only** so flat recommendations never leak into the headset feed.

So: **separate repo, separate program, VR-only media directory, separate taste model.** peaks-vr
*reuses the engine* (the copied core below) but keeps the streams uncrossed.

### Relationship to peaks (2D)

peaks-vr copies-and-diverges from peaks' proven core rather than sharing a package (zero cross-repo
coupling for a solo project). Lifted, then evolved independently:

- **Sampling framework** (ffmpeg-based frame extraction) → extended with a VR de-warp stage.
- **Embedding cache** (per-scene frame vectors + timestamps) → same format, VR-preprocessed input.
- **Taste model** (centroid / multi-mode / nearest-neighbor over embeddings) → unchanged math.
- **Similarity search** → unchanged.

New, VR-specific: format detection + reprojection, the HereSphere remote integration, the real-time
flagging UI, the moment-sequencer ("DJ"), and (later) trimming.

---

## The core problem: VR frames break vision models

peaks embeds a center-cropped square frame with DINOv2/CLIP. A raw VR frame is a **side-by-side (or
over-under) stereo pair** of **fisheye/equirectangular-warped** images. Center-cropping SBS lands on
the *seam between the two eyes*; the whole image is geometrically distorted. Embeddings of that are
meaningless — the taste model would be blind to VR content.

**The fix — a VR-aware sampling stage before embedding:**

1. **Detect the format** per scene (projection, stereo layout, FOV) from filename conventions
   (`_180_sbs`, `_MKX200`, `_FISHEYE190`, `_TB`, `_oculus`, …), Stash tags, and aspect ratio.
   (stash-vr already does this classification — study/borrow its logic.)
2. **Take one eye** — left half of SBS, top half of TB.
3. **Reproject to a flat perspective viewport** (ffmpeg `v360` filter) at a central action FOV
   (≈ forward-facing, ~90–110°). This yields a "normal" image the vision model understands.
4. **Embed** the viewport, exactly as 2D-peaks does.

The same de-warped viewport is used for **thumbnails** (raw fisheye previews look terrible).

This stage is the highest-risk, highest-value piece — build and validate it first. It is also
**GPU-hungry** (4K–8K decode + reprojection + embed), so NVDEC/hardware decode matters a lot.

### The projection zoo

VR libraries are a mess of formats: 180 vs 360, SBS vs TB (over-under), and fisheye variants
(MKX200, RF52, VRCA220, fisheye190…), at 4K–8K, with varied FOV and POV framing. Start with the most
common case — **180° SBS** (equirect and fisheye) — and expand coverage incrementally. Each
projection needs its own `v360` parameters.

---

## The linchpin: HereSphere's API (research before building)

Almost everything novel depends on **what an external app can read from and command on HereSphere in
real time**:

- Can it **report the currently-playing file + timecode** live? → gates real-time flagging.
- Can it **accept seek / load-next commands**? → gates sequential moment playback.

HereSphere exposes an HTTP API (it's how it consumes video libraries) and is broadly **DeoVR
remote-control compatible** (DeoVR's remote is a documented WebSocket streaming `currentTime` / `path`
/ `playing` and accepting seek/play commands). **Confirm the read + control surface first** — it's the
load-bearing wall. If current-time can't be read externally, real-time flagging needs a fallback
(e.g., drop a HereSphere bookmark and read it back).

---

## Features / design

### 1. Cache the VR scenes
The cache is embedding **vectors** (tiny on disk) — the cost is *compute*, not storage. Each scene is
format-detected → de-warped (one eye → flat viewport) → embedded at sampled intervals, keyed by a
stable file fingerprint. Reuses peaks' embedding cache; the new work is the reprojection stage and
robust format detection.

### 2. Real-time moment flagging (the novel interaction)
While you watch in HereSphere, a **peaks-vr web UI on a nearby computer** mirrors playback:

- Connect to HereSphere's live playback channel → read current **file + timecode**.
- Render a **de-warped frame** at that timecode (peaks-vr has the scene cached) so the UI shows
  roughly what you're seeing.
- A **❤️ mark** control writes an apex/label at (scene, time).

Refinements:
- **Reaction-lag scrubber** — when you hit like, the good bit was ~1–2 s ago; let the UI scrub a few
  seconds around "now" so you nail the exact frame.
- **In/out window** — capturing start+end (not a single point) yields precise apex clips for free.

### 3. Generate similar moments
Once liked moments are embedded (de-warped viewport), this is a **direct port of 2D-peaks' taste /
similarity engine** onto the VR embedding space — centroid / multi-mode / nearest-neighbor, unchanged.
The reprojection in #1 is what makes the vectors meaningful; the ranking itself is the proven part.

### 4. Play liked + recommended moments sequentially in the headset
**peaks-vr as a "VR DJ" over the remote API (recommended).** It seeks HereSphere to a moment and, when
the clip window elapses, commands it to load the next. Dynamic, and — crucially — **each moment plays
its own source file in its native projection**, so HereSphere re-detects format per clip. This
dissolves the mixed-format problem (see #6).

Alternative — a **pre-stitched single-file reel** — is simpler to "play" but only works within one
format group (see #5/#6).

### 5. Trim videos to reclaim space (long-term)
Cutting clips doesn't touch projection (it's metadata/tags, not pixels), so trimmed VR files still
play correctly. Caveats:
- **Keyframe alignment** — precise 8K cuts without re-encoding require cutting on keyframes
  (stream-copy) or an expensive re-encode; default to padding to nearest keyframes + stream-copy.
- **Destructive** — operate on copies, keep originals until reviewed, and preserve VR tags/filename
  conventions so HereSphere still detects format.

Savings are huge (e.g. three 30 s moments kept out of a 30-min 8K scene).

### 6. Mixed formats / POV / size
The library is a zoo (180/360, SBS/TB, MKX200/RF52/fisheye190, 4K–8K, varied POV). Key insight:
**don't unify formats for playback — let each moment play its own file.** The DJ playlist (#4) means
HereSphere re-detects projection on every clip, so any mix can sit back-to-back with only a brief
reload between clips — no transcoding. **Stitching (#5) is the only place format matters**, so the
rule is: *stitch only within a format group (same projection + stereo + resolution); across groups,
use the DJ playlist.* Jarring POV/height jumps between clips are a mild VR-comfort issue (HereSphere's
re-center helps); optionally order a reel to group similar POV.

---

## Roadmap

- **Phase 0 — Research (no code):** nail down HereSphere's API surface (read current file+time;
  send seek/load-next). Everything else branches on this.
- **Phase 1 — The taste loop, end to end:** VR-aware embedding (#1) → real-time flagging UI (#2) →
  similar-moment recommendations (#3) → DJ playback (#4). Proves the whole thesis; needs no
  stitching or trimming.
- **Phase 2 — Archival / space:** trimming (#5), same-format reel stitching, retention policies.

---

## Open questions / risks

- **HereSphere API** capabilities (the linchpin above).
- **Which viewport(s) to embed** — center-forward is the obvious default, but VR action often sits
  lower; consider a second downward viewport or a small viewport ensemble per sample.
- **Reprojection coverage** — how many projection types to support before it's "enough."
- **Compute budget** — de-warping + embedding an 8K VR library is heavy; needs GPU decode.
- **Format-detection accuracy** — misclassification → wrong de-warp → bad embeddings; needs a
  confident classifier and a manual override.

---

## Getting started (once the repo exists)

1. Copy peaks' core modules (sampling, embedding cache, taste model, search) as the starting point.
2. Phase 0: probe HereSphere's API from a script; document the exact read/control endpoints.
3. Phase 1: build the VR sampling/reprojection stage first (highest risk), validate embeddings on a
   handful of scenes, then wire the flagging UI → recs → DJ playback.

GPU (NVDEC) strongly recommended for embedding; the reprojection + high-res decode is the bottleneck.
