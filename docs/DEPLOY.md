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

Open **http://\<server-ip\>:8801**. Two tabs — no console needed.

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
   it before retrying. (NVDEC is optional: if the GPU/driver isn't available in
   the container, decode falls back to CPU automatically — slower, but it won't
   fail the file. To actually get GPU decode, ensure the unraid **Nvidia Driver**
   plugin is installed and the container has `--runtime=nvidia`.)

**How long should embedding take?** With the **GPU working** (NVDEC decode +
CUDA model), a typical 20–40 min 8K scene is ~**1–3 minutes**. On **CPU only**
(no GPU reaching the container), expect ~**3–6 minutes** per 8K scene — and if you
see 20+ minutes, the GPU almost certainly isn't active. Levers if you're
CPU-bound: get the NVIDIA runtime working (the real fix — check `nvidia-smi` in
the container console), raise the **Interval** (fewer samples), or use a lighter
`PEAKS_VR_MODEL`. The Embed tab shows the current file's elapsed seconds so you
can see a scene is progressing, not stuck.

### ② Flag moments (needs HereSphere playback)

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

## Path mapping note

HereSphere reports the file it's playing by its own path. For `recommend` to line
up a ❤️ mark with an embedded scene, embed your library through the container's
`/data` mount. If HereSphere's reported paths differ from `/data/...`, a future
path-remap setting will bridge them; for now, matching the mount to how
HereSphere sees the library keeps keys consistent.
