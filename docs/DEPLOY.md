# Deploying peaks-vr (Docker)

peaks-vr runs as a **pulled container** — no Python, pip, or ffmpeg on your
machine. The image ships CUDA torch + an NVDEC ffmpeg, so the GPU-heavy `embed`
pass works with the NVIDIA runtime; the flagging UI and the HereSphere intake run
CPU-only.

The container runs the **flagging web UI** (port `8801`) plus a **HereSphere
timestamp-server listener** (port `23573`). `embed` / `recommend` / `dj` run from
the container console.

Image: `ghcr.io/crescentfreshhh/peaks-vr:latest` (built + pushed by CI on every
push to `main`).

---

## Two ways HereSphere talks to peaks-vr

HereSphere exposes playback two different ways; peaks-vr supports both.

| | Timestamp server (default) | DeoVR remote |
|---|---|---|
| Direction | **HereSphere → peaks-vr** (headset connects in) | **peaks-vr → HereSphere** (we dial the headset) |
| peaks-vr role | listens on `23573` | connects to headset `:23554` |
| Gives us | live file + timecode (**read**) | read **+ control** (seek/load) |
| Powers | flagging (#2) | flagging **and** the DJ (#4) |
| Setup | enter peaks-vr's IP:port in HereSphere | enter the headset IP in peaks-vr |

Most HereSphere builds expose the **timestamp server** (it asks you for an IP +
port). That fully covers flagging. The **DJ** needs control, which is the DeoVR
remote — test whether your build offers it separately (below).

---

## Unraid (Community Applications template)

1. Copy `unraid/peaks-vr.xml` into `/boot/config/plugins/dockerMan/templates-user/`
   (or add this repo as a template source), then **Add Container → peaks-vr**.
2. Set the two paths:
   - **Config / data** → `/mnt/user/appdata/peaks-vr` (cache, labels, models).
   - **VR library** → the share holding your VR videos, mounted at `/data` (ro).
3. Leave the ports at `8801` / `23573`. Keep `--runtime=nvidia` (needs the
   **Nvidia Driver** plugin) for embedding.
4. Apply — it pulls the image and starts.

## docker compose (anything else)

```bash
git clone https://github.com/crescentfreshhh/peaks-vr.git
cd peaks-vr
# edit docker-compose.yml: point the /data volume at your VR library
docker compose up -d
```

(No GPU? Delete the `deploy:` block — the flag UI still runs; only `embed` needs
the GPU.)

---

## Everything runs from the WebGUI

Open **http://\<server-ip\>:8801**. Four tabs — no console needed.

### Password protection (optional)

Set **`PEAKS_VR_PASSWORD`** to require a password for the WebGUI. Unset (the
default) leaves it open on your LAN. When set:

- Opening `:8801` shows a login page; the panel is unreachable until you sign in.
- The session **locks after 1 hour of inactivity** (`PEAKS_VR_SESSION_TIMEOUT`
  seconds) — "inactivity" means no interaction, so a tab left open still locks;
  clicking/typing keeps it alive. A **Lock** link (top-right) signs out on demand.
- It guards the browser app only. HereSphere's timestamp server / DeoVR remote
  can't send a password, so playback is unaffected.

The login cookie is signed with a key auto-generated at `/config/session.key`
(persists across restarts; override with `PEAKS_VR_SECRET`).

### ① Embed library (do this first, no headset required)

This is the GPU pass that measures every scene (sample a frame every N seconds →
de-warp one eye to a flat viewport → vision-model fingerprint → cache). It's a
one-time cost per file, **resumable**, and independent of HereSphere.

1. The **Library** card shows how many videos were found under `/data` and how
   many are already embedded.
2. **Preview the de-warp first.** In the Preview card, pick a file and click
   **Preview frame**. The image should look like a **normal forward-facing view**
   — not fisheye, not stretched, no seam down the middle. If it's off, the VR
   format wasn't recognized from the filename (names need a hint like `_180_sbs`,
   `_MKX200_tb`, `_FISHEYE190`), or nudge **FOV/Pitch** (VR action often sits low,
   so try Pitch `-15`).
3. Click **Start embedding**. A progress bar, current file, ETA, and a live log
   appear. You can **Stop** and resume anytime — done scenes are skipped.
4. **Failed embeds** appear in their own card with the ffmpeg error. **Retry
   failed** re-attempts only those (faster than a full rescan); a fixed file
   drops off the list automatically. Some errors mean the file itself is broken
   — e.g. `moov atom not found` = a truncated/incomplete mp4; fix or re-download
   it before retrying. The card now shows the **real reason** for each failure
   (not just an exit code). VR de-warp is resilient: if the GPU (NVDEC) decode of
   a scene crashes on a tricky encode, it **auto-retries that scene on CPU
   decode** before marking it failed — so a GPU-hostile file still embeds. (NVDEC
   is optional overall: without a GPU/driver, decode is CPU throughout — slower,
   but it won't fail the file. For GPU decode, install the unraid **Nvidia
   Driver** plugin and run with `--runtime=nvidia`.)

**How long should embedding take?** VR sampling opens each file **once**, decodes
**keyframes only** in a single process, and de-warps them in-process (the same
one-decoder-per-scene approach that makes 2D peaks fast) — then the **AI model
runs on the GPU**. With **Decode = auto/cuda** the heavy 8K HEVC keyframe decode
runs on the GPU (NVDEC), one decoder context per scene, and only the light v360
de-warp stays on CPU — expect roughly **1–2 min** per 20–40 min 8K scene. On CPU
decode (no GPU, or Decode = cpu) the same 8K scene is slower — several minutes —
but still completes. If NVDEC can't handle a particular file (e.g. a 10-bit or
alpha profile), decode **falls back to CPU automatically** and logs one line to
the container log (`docker logs peaks-vr` → `NVDEC decode unavailable … using CPU
decode`); the file is never failed just because the GPU path didn't take.

Levers: raise the **Interval** (fewer samples) or use a lighter `PEAKS_VR_MODEL`.
The Embed tab shows the current file's elapsed seconds so you can watch it
progress.

**RAM watchdog (cap 24 GB).** Embedding 8K is memory-hungry — decode buffers, a
whole scene's frames on the GPU, the torch allocator, and allocator fragmentation
across a long run. peaks-vr watches its own resident memory and **self-regulates**
to stay under a cap (default **24 GB**, set `PEAKS_VR_MAX_RAM_GB`): as usage rises
it reclaims memory (GC + CUDA cache + return arenas to the OS), and if it reaches
the cap it **stops the run cleanly** — before the kernel OOM-killer can crash the
container — logging why. Since embedding is resumable, you lose nothing: free
headroom (raise the cap, raise the **Interval**, or use a lighter model) and start
again. Set `PEAKS_VR_MAX_RAM_GB=0` to disable.

The Embed tab shows `RAM 1.4 / 24 GB · container 8.4 GB`. Two numbers, because
they measure different things — and this is why the peaks-vr figure looks lower
than `docker stats`:

- **`RAM` (working set)** — anonymous memory (heap, tensors, decode buffers). This
  is what actually causes an OOM kill, so **the 24 GB cap is enforced on this
  number**.
- **`container`** — the total `docker stats` reports: the working set **plus**
  reclaimable file page cache and the multi-GB CUDA/torch library mappings (and
  page cache from reading 8K files). It's larger, but that extra memory is
  reclaimable and doesn't OOM you, so it isn't what the cap watches.

So a gap between the two (e.g. peaks-vr says 1.4 GB while `docker stats` says
8.4 GB) is normal and healthy — most of the difference is CUDA/torch libraries
and file cache.

The embed **log and last result are persisted** to `/config/embed_status.json`,
so the running (or last) run's progress and log show on a page refresh, from any
device, and even after a container restart.

**Per-scene timeout.** Each scene has a ceiling (default **900 s** = 15 min);
past it the scene is marked failed and the run moves on, so one pathological file
can't wedge the batch. Genuinely heavy 8K scenes can take a few minutes, so the
old 180 s cap was too low and could kill legitimate files — 900 s is the new
default. Set it per run in the Embed tab's **Per-scene timeout (s)** field (0 =
no limit), or via `PEAKS_SCENE_TIMEOUT` for CLI embeds.

### ② QC the embeds (no headset required)

After a run, spot-check the whole library at a glance. The **QC embeds** tab
shows a contact sheet — for every video, **two frames** (early + late), de-warped
with the *exact same transform the embedder used* — plus a status badge:

- **embedded** (green) — a cache entry exists for this file under the current
  model.
- **failed** (red) — hover for the ffmpeg error; also actionable from the
  Failed-embeds card on the Embed tab.
- **not embedded** (grey) — discovered but not yet processed.

Read the thumbnails: each should look like a **normal forward-facing view**. If
one is **fisheye, split down the middle (two frames), or stretched**, that file's
format was detected wrong — the embeddings for it are meaningless. Use the filter
chips (All / Embedded / Failed / Not embedded) to narrow the grid, and click any
thumbnail to enlarge it. Thumbnails load lazily as you scroll, so a large library
doesn't render every frame at once.

**Fix a wrong de-warp (⟳ fix).** Every card has a **⟳ fix** button. It opens a
panel where you force the correct **methodology** for that one file — flip
SBS↔TB, switch equirect↔fisheye (MKX200/220, Fisheye190), nudge FOV/pitch, or
choose **Flat (no de-warp)** for already-flat content. **Preview** renders one
de-warped eye so you can confirm it before committing; **Re-embed this file** then
drops the old vector and re-embeds just that file with the chosen format. The
correction is **sticky** — saved per file (a green **override** badge marks it)
and reused by every future embed, even after a cache clear, so you never have to
re-fix the same file. Picking **Auto (re-detect)** clears the override.

### ③ Build your DJ taste (no headset required)

The **DJ taste** tab bootstraps your taste from the already-embedded library, so
you can build a profile before/without HereSphere. It shows a contact sheet of
frames sampled across every scene; **👍** the ones that match your taste and each
becomes a positive example the DJ learns from (stored in `/config/labels.json`,
same profile the ❤️ flag marks feed).

- **Categories.** Type a **category** (cowgirl, blowjob…) before liking and those
  👍s are tagged (stored as `dj:cowgirl` etc.). Set a category, thumb-up all the
  frames that fit, then switch. Matching still spans **all** your likes
  (nearest-neighbour — every like is its own reference, so distinct acts don't blur
  together); categories are for organising and, later, balancing a DJ set across
  acts. Leave the field blank to like without a tag.
- **Category dropdown.** Pick an existing category or **＋ New category…** to add
  one; the choice tags every 👍 until you change it. **Untagged** = no tag.
- **Load more** reshuffles a fresh batch (already-liked frames are excluded). With
  **Untagged** selected it shows **truly random** frames across the library (broad
  exploration); with a category selected and enough likes, it shifts toward "more
  like what you like" so the profile sharpens.
- **⌕ more like this** (next to each 👍) refills the grid with frames **similar to
  that one** — query-by-example against the frame's vector, ranked by similarity,
  capped per scene for variety. A "show random ↺" link returns to exploration.
- Click a thumbnail to enlarge it. The header shows your total likes and the
  per-category breakdown.

**Taste profiles (the picker at the top).** Every like/mark files into a named
**taste profile**. The **Taste profile** dropdown (top of the page) switches which
one is active, and **+ new** creates one (e.g. `dj`, `date_night`) — the choice is
saved to `/config/active_profile` and sticks across restarts, so you don't need to
set `PEAKS_VR_PROFILE` (that env var is now just the *initial* default). Marks, the
DJ-taste 👍, and the recommender/DJ all read from the active profile; categories
nest under it as `<profile>:<category>`. Keep everything in one profile, or run
several (a DJ set profile, a flagging profile, different vibes).

### ④ Flag moments (needs HereSphere playback)

1. Find **this server's LAN IP** (unraid shows it; else `ip addr`/`ipconfig`),
   e.g. `192.168.1.50`.
2. In HereSphere → settings → **timestamp server**, enter `192.168.1.50` and port
   **`23573`**, enable it.
3. On the **Flag** tab, put the headset on and play a video — the page mirrors
   playback; tap **❤ MARK** on moments you like (marks → `/config/labels.json`).

If the Flag tab connects but shows no playback, set `PEAKS_VR_BYTEORDER=little`
and restart. Still nothing? Diagnose from the console:
`docker exec -it peaks-vr peaks-vr probe --listen --ts-port 23573` (prints the raw
bytes if it can't decode — send those).

(Turning the library + your taste into a ranked DJ playlist is the next step,
once scenes are embedded.)

---

## Environment variables

| var | default | meaning |
|---|---|---|
| `PEAKS_VR_PROFILE` | `apex` | taste profile the ❤️ marks belong to |
| `PEAKS_VR_TS_PORT` | `23573` | timestamp-server intake port |
| `PEAKS_VR_WEB_PORT` | `8801` | flagging UI port |
| `PEAKS_VR_BYTEORDER` | `big` | stream length-prefix endianness (`big`/`little`) |
| `PEAKS_VR_REMOTE_HOST` | _(unset)_ | set to the headset IP to **dial the DeoVR remote** instead of listening |
| `PEAKS_VR_REMOTE_PORT` | `23554` | DeoVR remote port |
| `PEAKS_VR_PREVIEW` | _(off)_ | `1` = live de-warped preview (needs ffmpeg + an embedded scene) |
| `PEAKS_SCENE_TIMEOUT` | `900` | per-scene sampling ceiling in seconds (`0` = off); fallback for CLI embeds — the Embed tab's field overrides it per run |
| `PEAKS_VR_MAX_RAM_GB` | `24` | RAM watchdog cap in GB (`0` = disable). Under pressure it reclaims memory; at the cap it stops the run cleanly (resumable) before an OOM kill |
| `PEAKS_VR_PASSWORD` | _(unset)_ | set to require a password for the WebGUI; unset = open. Session locks after the idle timeout |
| `PEAKS_VR_SESSION_TIMEOUT` | `3600` | WebGUI idle-session timeout in seconds (1 hour). Only used when `PEAKS_VR_PASSWORD` is set |
| `PEAKS_VR_SECRET` | _(auto)_ | cookie signing key. Auto-generated + persisted to `/config/session.key`; set to pin it |

## Path mapping note

HereSphere reports the file it's playing by its own path. For `recommend` to line
up a ❤️ mark with an embedded scene, embed your library through the container's
`/data` mount. If HereSphere's reported paths differ from `/data/...`, a future
path-remap setting will bridge them; for now, matching the mount to how
HereSphere sees the library keeps keys consistent.
