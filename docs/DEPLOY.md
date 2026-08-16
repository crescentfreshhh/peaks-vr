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

## First run — connect HereSphere

1. Find **this server's LAN IP** (the machine running the container): unraid
   shows it; otherwise `ip addr` / `ipconfig`. Say it's `192.168.1.50`.
2. In HereSphere → settings → **timestamp server**, enter
   `192.168.1.50` and port **`23573`**, and enable it.
3. Open the UI at **http://192.168.1.50:8801**, put the headset on, play a video.
   The page should mirror playback; tap **❤ MARK** on the moments you like.
   Marks persist to `/config/labels.json`.

If the UI connects but shows no playback, the stream's byte order may be flipped
— set `PEAKS_VR_BYTEORDER=little` and restart. If it still shows nothing, run the
diagnostic in the console (below) and send the hex it prints.

## Embed your library

Embedding measures every scene — sample a frame every N seconds → de-warp (one
eye → flat viewport) → run it through the vision model → cache the vectors. It's
a one-time GPU cost per file, **resumable** (re-running skips what's done), and
independent of HereSphere. It's the prerequisite for finding worthy moments later.

**1. Preview the de-warp first** — the VR reprojection depends on the file's
format being detected from its name, so confirm it looks right on one file before
committing the whole library:

```bash
docker exec -it peaks-vr peaks-vr preview "/data/<clip>_180_sbs.mp4" --out /config/preview.jpg
```

Open `/config/preview.jpg` off the share. It should look like a **normal
forward-facing view** — not fisheye, not stretched, no seam down the middle. If
it's off, the format probably wasn't detected from the filename (make sure names
carry a hint like `_180_sbs`, `_MKX200_tb`, `_FISHEYE190`), or tune the viewport
with `--fov` / `--pitch` (VR action often sits a bit low, so e.g. `--pitch -15`).

**2. Embed the whole library** (point it at the mount; directories are scanned
recursively; GPU decode via NVDEC):

```bash
docker exec -it peaks-vr peaks-vr --model dino embed /data --vr --hwaccel cuda
```

It prints per-scene progress and an ETA, and you can stop/restart it any time —
already-embedded scenes are skipped. Interval defaults to 8s (`--interval` to
change).

### Other console commands

```bash
# diagnose the timestamp stream (prints raw bytes if it can't decode)
docker exec -it peaks-vr peaks-vr probe --listen --ts-port 23573

# check whether the DeoVR remote (control, for the DJ) is reachable
docker exec -it peaks-vr peaks-vr probe --host <headset-ip>
```

(Finding worthy moments — from reference stills or ❤️ marks — and playing them
with the DJ is a later step, once the library is embedded.)

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
