"""Frame sampling.

Three modes:

  "sparse"    — THE FAST PATH for long videos: seek to each sample time and
                decode exactly one keyframe (PyAV, in-process). Cost scales
                with the number of SAMPLES, not the video's duration — unlike
                every other mode, which must decode/discard the whole file.
                Timestamps are the decoded keyframe's exact pts, so samples
                snap to keyframes (typically within a couple of seconds of the
                grid). Requires the `av` package (bundled ffmpeg libs).

  "interval"  — one frame every `interval_seconds`, streamed out of ffmpeg
                over a pipe through a bounded queue. Decodes the ENTIRE video
                to keep a fraction of it — fine for short clips, brutal for
                hour-long files. Frame i comes from source time ~i*interval.

  "keyframes" — decode only keyframes (`-skip_frame nokey`): far less decode
                than interval, but still reads/parses the whole file and
                yields every keyframe (dense-GOP files gain little).

Optional `hwaccel` ("cuda"/"auto") offloads interval-mode decode to the GPU
(NVDEC). Sparse mode doesn't need it: it decodes ~one I-frame per sample.

The pure helpers (`plan_timestamps`, `iter_jpegs`, `iter_fixed_chunks`,
`parse_showinfo_times`) are unit-tested offline; sparse mode is tested against
real generated video (PyAV bundles codecs); the ffmpeg-binary paths need a
real box.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from typing import IO, TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

    from .reprojection import Reprojector

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def plan_timestamps(
    duration: float, interval: float, *, offset: float = 0.0
) -> list[float]:
    """Expected sample timestamps for a clip: `offset, offset+interval, ...`.

    Matches how ffmpeg's `fps=1/interval` filter emits frames: the first output
    frame comes from the very start of the clip, then one per interval. Frame i
    therefore represents source time ~`i * interval` — timestamps must NOT be
    shifted, or every downstream marker/label lands offset from its frame.
    """
    if duration <= 0 or interval <= 0:
        return []
    times: list[float] = []
    t = offset
    while t < duration:
        times.append(round(t, 3))
        t += interval
    return times


def iter_jpegs(stream: IO[bytes], chunk_size: int = 65536) -> Iterator[bytes]:
    """Split a concatenated-MJPEG byte stream into individual JPEG blobs.

    Scans for SOI (FFD8) / EOI (FFD9) markers across chunk boundaries. Safe for
    ffmpeg mjpeg output: 0xFF bytes inside entropy-coded data are always
    followed by 0x00 stuffing, and there are no embedded EXIF thumbnails, so a
    raw FFD9 only ever terminates a frame.
    """
    buf = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        buf += chunk
        while True:
            soi = buf.find(_JPEG_SOI)
            if soi < 0:
                buf = buf[-1:]  # keep a possible split FF
                break
            eoi = buf.find(_JPEG_EOI, soi + 2)
            if eoi < 0:
                buf = buf[soi:]  # incomplete frame; wait for more bytes
                break
            yield buf[soi : eoi + 2]
            buf = buf[eoi + 2 :]


def iter_fixed_chunks(stream: IO[bytes], chunk_size: int) -> Iterator[bytes]:
    """Yield exact `chunk_size` slices from a stream (rawvideo framing).

    Handles short reads; a trailing partial chunk (truncated stream) is
    dropped rather than yielded corrupt.
    """
    buf = bytearray()
    while True:
        data = stream.read(chunk_size - len(buf) if len(buf) < chunk_size else chunk_size)
        if not data:
            break
        buf.extend(data)
        while len(buf) >= chunk_size:
            yield bytes(buf[:chunk_size])
            del buf[:chunk_size]


_SHOWINFO_PTS = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_showinfo_times(stderr_text: str) -> list[float]:
    """Extract per-frame pts_time values from ffmpeg showinfo stderr output."""
    return [round(float(m), 3) for m in _SHOWINFO_PTS.findall(stderr_text)]


class SamplerError(RuntimeError):
    pass


def _sparse_extract_worker(
    path: str, interval: float, resize_short: int, crop: int, out_path: str
) -> None:
    """Runs in a CHILD process: sparse-decode `path` and save (times, frames)
    to `out_path` as an .npz. Kept at module level so it's importable by the
    `spawn` start method. A non-zero exit / no output tells the parent the
    scene failed; a hang is handled by the parent's kill-timeout.

    In-worker guards (total-error cap, implausible-duration check) make it exit
    fast on obviously-broken files without waiting for the parent timeout."""
    import numpy as np
    import av

    try:
        av.logging.set_level(av.logging.ERROR)  # mute per-seek decoder chatter
    except Exception:
        pass

    max_total_errors = 60
    max_scene_hours = 12.0

    times: list[float] = []
    frames: list = []
    with av.open(path) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        try:  # decode keyframes only — big speedup, and seeks land on them
            stream.codec_context.skip_frame = "NONKEY"
        except (AttributeError, ValueError):
            pass

        tb = stream.time_base
        if stream.duration and tb:
            duration = float(stream.duration * tb)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            raise RuntimeError(f"could not determine duration of {path}")
        if duration <= 0 or duration > max_scene_hours * 3600:
            raise RuntimeError(f"implausible duration {duration:.0f}s for {path}")

        interp_kw: dict = {"interpolation": "BICUBIC"}
        last_pts = None
        total_errors = 0
        t = 0.0
        while t < duration:
            target = t
            t += interval
            try:
                container.seek(int(target / tb), stream=stream)
                frame = next(container.decode(stream), None)
            except StopIteration:
                break
            except Exception:
                total_errors += 1
                if total_errors >= max_total_errors:
                    raise RuntimeError(f"{total_errors} decode errors in {path}")
                continue
            if frame is None:
                break
            if frame.pts is not None and last_pts is not None and frame.pts <= last_pts:
                continue  # same keyframe as the previous sample
            last_pts = frame.pts
            ts = frame.time if frame.time is not None else target

            scale = resize_short / min(frame.width, frame.height)
            nw = max(crop, int(round(frame.width * scale)))
            nh = max(crop, int(round(frame.height * scale)))
            try:
                out = frame.reformat(width=nw, height=nh, format="rgb24", **interp_kw)
            except TypeError:  # older PyAV without the interpolation kwarg
                interp_kw = {}
                out = frame.reformat(width=nw, height=nh, format="rgb24")
            arr = out.to_ndarray()
            y0 = (nh - crop) // 2
            x0 = (nw - crop) // 2
            frames.append(np.ascontiguousarray(arr[y0 : y0 + crop, x0 : x0 + crop]))
            times.append(round(float(ts), 3))

    if not times:
        raise RuntimeError(f"no frames decoded for {path}")
    np.savez(
        out_path,
        times=np.asarray(times, dtype=np.float32),
        frames=np.stack(frames),
    )


def _open_container_for_decode(path: str, hwaccel: str):
    """Open ``path`` with NVDEC hardware decode when ``hwaccel`` is cuda/auto,
    else plain CPU. Returns the container. On any hardware-init failure (no
    libcuda in the container, unsupported codec/profile — e.g. 10-bit or alpha),
    logs one line to stderr and falls back to CPU decode, so a file is never
    failed just because the GPU path didn't take. One context per scene (this is
    called once per file), so there's no per-sample NVDEC-init storm."""
    import sys

    import av

    if hwaccel in ("cuda", "auto"):
        try:
            from av.codec.hwaccel import HWAccel

            hw = HWAccel(device_type="cuda", allow_software_fallback=True)
            return av.open(path, hwaccel=hw)
        except Exception as exc:  # no CUDA, old PyAV, unsupported profile, …
            sys.stderr.write(
                f"[peaks-vr] NVDEC decode unavailable for {path} "
                f"({type(exc).__name__}: {exc}); using CPU decode\n"
            )
    return av.open(path)


def _sparse_dewarp_worker(
    path: str, interval: float, crop: int, out_path: str, filter_chain: str,
    hwaccel: str = "",
) -> None:
    """Runs in a CHILD process: like `_sparse_extract_worker`, but applies the VR
    de-warp (one-eye crop → v360 → model resize/crop) to each decoded keyframe
    via a SINGLE in-process libavfilter graph. The file is opened ONCE and only
    keyframes are decoded, so cost scales with samples — the fast path that makes
    2D peaks quick, now with the reprojection. `filter_chain` is a comma-joined
    ffmpeg filtergraph ending at the model's crop×crop input geometry.

    ``hwaccel`` cuda/auto decodes the (expensive, 8K) HEVC keyframes on the GPU
    via NVDEC; the light v360 de-warp stays on CPU. Frames are transferred to
    system memory automatically, so the filtergraph is fed the same way as the
    CPU path. Any GPU-init failure falls back to CPU decode (see
    :func:`_open_container_for_decode`). Running in a spawned child isolates CUDA
    init from the parent."""
    import av
    import numpy as np

    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass

    # Record the real reason on failure so the parent can surface it (the child's
    # traceback otherwise only reaches stderr). Cleared on success.
    try:
        _sparse_dewarp_extract(path, interval, crop, out_path, filter_chain,
                               hwaccel, av, np)
    except BaseException as exc:  # incl. the RuntimeErrors below
        try:
            with open(out_path + ".err", "w") as fh:
                fh.write(f"{type(exc).__name__}: {exc}")
        except OSError:
            pass
        raise


def _sparse_dewarp_extract(path, interval, crop, out_path, filter_chain, hwaccel,
                           av, np) -> None:
    max_total_errors = 60
    max_scene_hours = 12.0
    times: list[float] = []
    frames: list = []

    with _open_container_for_decode(path, hwaccel) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        try:  # decode keyframes only — the big speedup; seeks land on them
            stream.codec_context.skip_frame = "NONKEY"
        except (AttributeError, ValueError):
            pass

        tb = stream.time_base
        if stream.duration and tb:
            duration = float(stream.duration * tb)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            raise RuntimeError(f"could not determine duration of {path}")
        if duration <= 0 or duration > max_scene_hours * 3600:
            raise RuntimeError(f"implausible duration {duration:.0f}s for {path}")

        graph = None  # built lazily once we have a frame for the buffer template
        last_pts = None
        total_errors = 0
        t = 0.0
        while t < duration:
            target = t
            t += interval
            try:
                container.seek(int(target / tb), stream=stream)
                frame = next(container.decode(stream), None)
            except StopIteration:
                break
            except Exception:
                total_errors += 1
                if total_errors >= max_total_errors:
                    raise RuntimeError(f"{total_errors} decode errors in {path}")
                continue
            if frame is None:
                break
            if frame.pts is not None and last_pts is not None and frame.pts <= last_pts:
                continue  # same keyframe as the previous sample
            last_pts = frame.pts
            ts = frame.time if frame.time is not None else target

            if graph is None:
                graph = av.filter.Graph()
                # Build the buffer from the ACTUAL decoded frame's geometry/format
                # so an NVDEC-downloaded frame (nv12) is described correctly, not
                # the stream's coded format. Falls back to the stream template on
                # older PyAV that lacks the explicit kwargs (the CPU path).
                try:
                    node = graph.add_buffer(
                        width=frame.width, height=frame.height,
                        format=frame.format.name, time_base=stream.time_base,
                    )
                except (TypeError, ValueError):
                    node = graph.add_buffer(template=stream)
                for spec in filter_chain.split(","):
                    name, _, args = spec.partition("=")
                    filt = graph.add(name.strip(), args.strip())
                    node.link_to(filt)
                    node = filt
                sink = graph.add("buffersink")
                node.link_to(sink)
                graph.configure()

            graph.push(frame)
            out = graph.pull()
            arr = out.to_ndarray(format="rgb24")  # ends at crop×crop from the chain
            frames.append(np.ascontiguousarray(arr))
            times.append(round(float(ts), 3))

    if not times:
        raise RuntimeError(f"no frames decoded for {path}")
    np.savez(
        out_path,
        times=np.asarray(times, dtype=np.float32),
        frames=np.stack(frames),
    )


class FrameSampler:
    def __init__(
        self,
        interval_seconds: float = 2.0,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        frame_size: int = 288,
        mode: str = "interval",
        hwaccel: str = "",
        queue_frames: int = 256,
        pipeline: str = "raw",
        scene_timeout: float = 900.0,
        reproject: "Reprojector | None" = None,
    ):
        """`frame_size`: short-side pixels for the JPEG pipeline (0 = original
        size). `queue_frames` bounds how far decode runs ahead of the consumer.
        `pipeline`: "raw" pipes raw RGB frames pre-sized to the model's input
        straight to the embedder — no JPEG encode, no PIL decode, no CPU
        resize (the fast path). "jpeg" is the legacy/escape-hatch path.

        `reproject`: an optional VR de-warp stage (:class:`peaks_vr.reprojection.
        Reprojector`). When set, its ffmpeg filter (one-eye crop → v360 flat
        viewport) is applied to every decoded frame *before* the model-input
        resize/crop, so the cache stores de-warped-viewport vectors — the whole
        point of peaks-vr. When None (the default) this sampler behaves exactly
        like 2D-peaks', which is what keeps the ported flat-video tests valid."""
        if mode not in ("sparse", "interval", "keyframes"):
            raise ValueError(f"unknown sampling mode: {mode!r}")
        if pipeline not in ("raw", "jpeg"):
            raise ValueError(f"unknown pipeline: {pipeline!r}")
        self.interval = interval_seconds
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.frame_size = frame_size
        self.mode = mode
        self.hwaccel = hwaccel
        self.queue_frames = queue_frames
        self.pipeline = pipeline
        # hard ceiling on how long a single scene may spend sampling, so one
        # corrupt/pathological file can never stall the whole run. 0 disables.
        self.scene_timeout = scene_timeout
        # Optional VR de-warp prepended to every filtergraph (None == flat 2D).
        self.reproject = reproject

    @property
    def interval_signature(self) -> float:
        """Value stored/checked in the embedding cache so a sampling-config
        change invalidates old entries. Keyframe mode uses -1.0 (its spacing
        is encode-dependent, not an interval); sparse encodes as -(100 +
        interval) so each sparse grid is distinct from every interval grid."""
        if self.mode == "keyframes":
            return -1.0
        if self.mode == "sparse":
            return -(100.0 + self.interval)
        return self.interval

    @property
    def wants_raw(self) -> bool:
        """True when this sampler produces numpy frames for embed_array.
        Sparse mode is always raw (it decodes straight to arrays); interval
        mode honours the pipeline setting; keyframes stays on the PIL path."""
        return self.mode == "sparse" or (
            self.mode == "interval" and self.pipeline == "raw"
        )

    # --- probing ---------------------------------------------------------------

    def probe_duration(self, path: str) -> float:
        """Return duration in seconds via ffprobe."""
        cmd = [
            self.ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffprobe} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise SamplerError(f"ffprobe failed for {path}: {exc.stderr[:300]}") from exc
        try:
            return float(json.loads(out.stdout)["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SamplerError(f"could not parse duration for {path}") from exc

    def probe_dimensions(self, path: str) -> tuple[int, int] | None:
        """Return (width, height) of the first video stream via ffprobe, or None
        on any failure — a best-effort signal for aspect-ratio format detection."""
        cmd = [
            self.ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", path,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            s = json.loads(out.stdout)["streams"][0]
            w, h = int(s["width"]), int(s["height"])
            return (w, h) if w > 0 and h > 0 else None
        except Exception:
            return None

    # --- command assembly -------------------------------------------------------

    def _scale_filter(self) -> str:
        if not self.frame_size:
            return ""
        # short side -> frame_size, other side scales up to keep aspect
        return (
            f"scale=w={self.frame_size}:h={self.frame_size}"
            ":force_original_aspect_ratio=increase:force_divisible_by=2"
        )

    def _input_args(self) -> list[str]:
        args = []
        if self.hwaccel:
            args += ["-hwaccel", self.hwaccel]
        return args

    def _vf(self) -> str:
        """The ffmpeg filtergraph for the current mode."""
        if self.mode == "keyframes":
            parts = ["showinfo"]  # exact pts_time per emitted frame
        else:
            parts = [f"fps=1/{self.interval:g}"]
        if self.reproject is not None:
            # VR de-warp runs on the sampled frame, before any scaling.
            parts.append(self.reproject.ffmpeg_filter())
        scale = self._scale_filter()
        if scale:
            parts.append(scale)
        return ",".join(parts)

    def _raw_vf(self, resize_short: int, crop: int) -> str:
        """Filtergraph for the raw pipeline: sample, then reproduce the
        model's preprocessing in ffmpeg (bicubic short-side resize + center
        crop), so frames arrive at exactly the network's input size."""
        dewarp = f"{self.reproject.ffmpeg_filter()}," if self.reproject else ""
        return (
            f"fps=1/{self.interval:g},"
            f"{dewarp}"
            f"scale=w={resize_short}:h={resize_short}"
            ":force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={crop}:{crop}"
        )

    # --- single frame (labeler) ---------------------------------------------------

    def grab_frame(self, path: str, time: float) -> "Image":
        """Decode a single frame at `time` seconds (used by the labeler).

        `-ss` before `-i` with re-encoding is frame-accurate in modern ffmpeg
        (it seeks to the prior keyframe, then decodes forward to the target).
        """
        from PIL import Image as PILImage  # lazy

        # When a reprojector is set, de-warp the grabbed frame too (one eye →
        # flat viewport) — so previews and the flag UI show what the model sees.
        vf = [] if self.reproject is None else ["-vf", self.reproject.ffmpeg_filter()]

        def _cmd(hwaccel: str) -> list[str]:
            args = ["-hwaccel", hwaccel] if hwaccel else []
            return [
                self.ffmpeg, "-v", "error", *args,
                "-ss", f"{time:g}", "-i", path, *vf,
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
            ]

        try:
            out = subprocess.run(_cmd(self.hwaccel), capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            if self.hwaccel:  # NVDEC unavailable → retry on CPU
                try:
                    out = subprocess.run(_cmd(""), capture_output=True, check=True)
                except subprocess.CalledProcessError as exc2:
                    raise SamplerError(
                        f"ffmpeg frame grab failed for {path}@{time}s: "
                        f"{exc2.stderr[:200].decode('utf-8', 'replace')}"
                    ) from exc2
            else:
                raise SamplerError(
                    f"ffmpeg frame grab failed for {path}@{time}s: "
                    f"{exc.stderr[:200].decode('utf-8', 'replace')}"
                ) from exc
        if not out.stdout:
            raise SamplerError(f"no frame decoded for {path}@{time}s")
        return PILImage.open(BytesIO(out.stdout)).convert("RGB")

    # --- bulk sampling --------------------------------------------------------------

    def iter_frames(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Yield (timestamp_seconds, PIL.Image) samples for a video."""
        if self.mode == "keyframes":
            yield from self._iter_frames_keyframes(path)
        else:
            yield from self._iter_frames_interval(path)

    def iter_frames_raw(self, path: str, *, resize_short: int, crop: int):
        """Yield (timestamp, HxWx3 uint8 numpy frame) at the model's input
        geometry — the raw path. Dispatches on mode: sparse seeks per sample;
        interval streams a full decode over a pipe."""
        if self.mode == "sparse":
            if self.reproject is not None:
                # VR: open the file ONCE, decode keyframes, and de-warp each in an
                # in-process libavfilter graph (PyAV). One decoder per scene — no
                # per-sample process spawn — so it's ~as fast as the flat 2D path.
                # Falls back to the per-sample ffmpeg path only if `av` is absent.
                try:
                    import av  # noqa: F401
                except ImportError:
                    yield from self._iter_frames_dewarp_seek(
                        path, resize_short=resize_short, crop=crop
                    )
                    return
                yield from self._iter_frames_dewarp_inproc(
                    path, resize_short=resize_short, crop=crop
                )
                return
            yield from self._iter_frames_sparse(
                path, resize_short=resize_short, crop=crop
            )
        else:
            yield from self._iter_frames_raw_interval(
                path, resize_short=resize_short, crop=crop
            )

    def _dewarp_raw_vf(self, resize_short: int, crop: int) -> str:
        """Filtergraph for a single de-warped sample at the model's input size:
        one-eye crop + v360 reprojection, then the model's resize/crop."""
        return (
            f"{self.reproject.ffmpeg_filter()},"
            f"scale=w={resize_short}:h={resize_short}"
            ":force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={crop}:{crop}"
        )

    def _run_dewarp_worker(self, path: str, filter_chain: str, crop: int,
                           hwaccel: str):
        """Spawn the de-warp worker with a given decode backend and return
        (times, frames) arrays, or raise :class:`SamplerError` with the *real*
        reason (read from the worker's ``.err`` sidecar, or the exit signal for a
        hard crash). Runs in a spawned child with the kill-timeout."""
        import multiprocessing as mp
        import os
        import tempfile

        import numpy as np  # lazy

        fd, out_path = tempfile.mkstemp(suffix=".npz", prefix="peaks-vrdewarp-")
        os.close(fd)
        err_path = out_path + ".err"
        ctx = mp.get_context("spawn")  # fresh interpreter: no CUDA-fork hazards
        proc = ctx.Process(
            target=_sparse_dewarp_worker,
            args=(path, self.interval, crop, out_path, filter_chain, hwaccel),
        )
        try:
            proc.start()
            proc.join(self.scene_timeout if self.scene_timeout else None)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():  # pragma: no cover - stubborn C loop
                    proc.kill()
                    proc.join()
                raise SamplerError(
                    f"VR de-warp sampling exceeded {self.scene_timeout:.0f}s on "
                    f"{path} (corrupt/pathological file?) — killed"
                )
            if proc.exitcode != 0:
                reason = ""
                try:
                    if os.path.exists(err_path):
                        reason = open(err_path).read().strip()
                except OSError:
                    pass
                if not reason and proc.exitcode and proc.exitcode < 0:
                    reason = (f"decoder crashed (signal {-proc.exitcode}) — "
                              f"likely {hwaccel or 'CPU'} decode on this encode")
                detail = f": {reason}" if reason else f" (worker exit {proc.exitcode})"
                raise SamplerError(f"VR de-warp extraction failed for {path}{detail}")
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise SamplerError(f"no frames produced for {path}")
            with np.load(out_path) as data:
                return np.array(data["times"]), np.array(data["frames"])
        finally:
            for p in (out_path, err_path):
                try:
                    os.unlink(p)
                except OSError:  # pragma: no cover
                    pass

    def _iter_frames_dewarp_inproc(self, path: str, *, resize_short: int, crop: int):
        """VR de-warp at sparse cost, in ONE process: open the file once, decode
        keyframes, and apply the v360 filtergraph in-process (PyAV). Runs in a
        spawned child with a kill-timeout, like `_iter_frames_sparse`, so a
        pathological file can't wedge the run.

        Resilience ladder: try the configured decode (NVDEC when cuda/auto); if
        the worker fails or crashes, **retry once on CPU decode** — NVDEC can
        hard-crash on some 8K HEVC encodes, and CPU keyframe decode is already
        cheap, so a GPU-hostile file still embeds instead of being marked failed.
        """
        import sys

        if self.interval <= 0:
            return
        filter_chain = self._dewarp_raw_vf(resize_short, crop)

        # dedup while preserving order: configured backend, then plain CPU.
        backends: list[str] = []
        for hw in (self.hwaccel, ""):
            if hw not in backends:
                backends.append(hw)

        times = frames = None
        last_err: Exception | None = None
        for i, hw in enumerate(backends):
            try:
                times, frames = self._run_dewarp_worker(path, filter_chain, crop, hw)
                break
            except SamplerError as exc:
                last_err = exc
                is_last = i == len(backends) - 1
                if not is_last:
                    sys.stderr.write(
                        f"[peaks-vr] {hw or 'CPU'} de-warp failed for {path} "
                        f"({exc}); retrying on CPU decode\n"
                    )
        if times is None:
            raise SamplerError(str(last_err))
        for i in range(len(times)):
            yield float(times[i]), frames[i]

    def _iter_frames_dewarp_seek(self, path: str, *, resize_short: int, crop: int):
        """VR de-warp sampling with sparse cost: seek to each planned timestamp
        and decode+de-warp exactly that one frame via ffmpeg (the v360 filter is
        an ffmpeg stage, so this can't go through the PyAV sparse worker).

        Yields (timestamp, (crop, crop, 3) uint8) — the raw pipeline geometry, so
        it feeds `embedder.embed_array` like the other raw paths. Cost scales with
        the number of samples, not the file's length; `-hwaccel` (NVDEC) offloads
        the per-sample decode."""
        import numpy as np  # lazy

        duration = self.probe_duration(path)
        times = plan_timestamps(duration, self.interval)
        vf = self._dewarp_raw_vf(resize_short, crop)
        nbytes = crop * crop * 3
        # ALWAYS decode on CPU here, even if a GPU is available. This path spawns
        # one ffmpeg per sample, and a hardware decoder (NVDEC) allocates a large
        # context and initializes on EVERY process — seconds of setup to decode a
        # single 8K keyframe that the CPU handles in ~0.2s. Per-sample NVDEC is a
        # net loss, so we skip it; the embedding model still runs on the GPU.
        # (`hw` stays a variable only so the fallback branch below is a no-op.)
        hw = ""

        def _cmd(t: float, hwaccel: str) -> list[str]:
            args = ["-hwaccel", hwaccel] if hwaccel else []
            # -noaccurate_seek: take the keyframe at/before t instead of decoding
            # the whole GOP forward to the exact time — ~1 decoded frame per
            # sample (the sparse convention; snaps to keyframes within a GOP).
            return [
                self.ffmpeg, "-v", "error", *args,
                "-ss", f"{t:g}", "-noaccurate_seek", "-i", path, "-vf", vf,
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ]

        for t in times:
            try:
                out = subprocess.run(_cmd(t, hw), capture_output=True, check=True)
            except FileNotFoundError as exc:
                raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
            except subprocess.CalledProcessError as exc:
                if hw:  # hardware decode unavailable → retry on CPU, then stick
                    hw = ""
                    try:
                        out = subprocess.run(_cmd(t, hw), capture_output=True,
                                             check=True)
                    except subprocess.CalledProcessError as exc2:
                        raise SamplerError(
                            f"ffmpeg de-warp failed for {path}@{t:g}s: "
                            f"{exc2.stderr[:200].decode('utf-8', 'replace')}"
                        ) from exc2
                else:
                    raise SamplerError(
                        f"ffmpeg de-warp failed for {path}@{t:g}s: "
                        f"{exc.stderr[:200].decode('utf-8', 'replace')}"
                    ) from exc
            if len(out.stdout) < nbytes:
                # a seek past a truncated tail can yield nothing — skip it
                continue
            arr = np.frombuffer(out.stdout[:nbytes], dtype=np.uint8).reshape(
                crop, crop, 3
            )
            yield float(t), arr

    def _iter_frames_sparse(self, path: str, *, resize_short: int, crop: int):
        """Seek-based sampling, run in a CHILD PROCESS with a hard kill-timeout.

        Decoding happens in `_sparse_extract_worker`; the frames come back via
        a temp .npz. A corrupt file that wedges libav inside a C call can't be
        interrupted from Python, so the parent simply kills the child after
        `scene_timeout` seconds and the scene is marked failed. Isolation also
        removes GIL contention so `workers` actually parallelize decode."""
        import multiprocessing as mp
        import os
        import tempfile

        import numpy as np  # lazy

        try:
            import av  # noqa: F401  (fail here with a clean message, not in child)
        except ImportError as exc:  # pragma: no cover - guarded in CLI too
            raise SamplerError(
                "sparse mode needs the 'av' package (pip install av)"
            ) from exc

        if self.interval <= 0:
            return

        fd, out_path = tempfile.mkstemp(suffix=".npz", prefix="peaks-sparse-")
        os.close(fd)
        ctx = mp.get_context("spawn")  # fresh interpreter: no CUDA-fork hazards
        proc = ctx.Process(
            target=_sparse_extract_worker,
            args=(path, self.interval, resize_short, crop, out_path),
        )
        try:
            proc.start()
            proc.join(self.scene_timeout if self.scene_timeout else None)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():  # pragma: no cover - stubborn C loop
                    proc.kill()
                    proc.join()
                raise SamplerError(
                    f"scene sampling exceeded {self.scene_timeout:.0f}s on "
                    f"{path} (corrupt/pathological file?) — killed"
                )
            if proc.exitcode != 0:
                raise SamplerError(
                    f"sparse extraction failed for {path} "
                    f"(worker exit {proc.exitcode})"
                )
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise SamplerError(f"no frames produced for {path}")
            with np.load(out_path) as data:
                times = data["times"]
                frames = data["frames"]
            for i in range(len(times)):
                yield float(times[i]), frames[i]
        finally:
            try:
                os.unlink(out_path)
            except OSError:  # pragma: no cover
                pass

    def _iter_frames_raw_interval(self, path: str, *, resize_short: int, crop: int):
        """Fast path: yield (timestamp, HxWx3 uint8 numpy frame) with frames
        already at the model's input geometry.

        ffmpeg decodes (optionally NVDEC), resizes and crops, and writes raw
        RGB24 to a pipe. Framing is trivial — every frame is exactly
        crop*crop*3 bytes — so the Python side is a memcpy, not a JPEG decode.
        A reader thread keeps decode running ahead of the consumer.
        """
        import numpy as np  # lazy: keeps module import light

        if self.interval <= 0:
            return
        frame_bytes = crop * crop * 3
        cmd = [
            self.ffmpeg,
            "-v", "error",
            *self._input_args(),
            "-i", path,
            "-vf", self._raw_vf(resize_short, crop),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc

        frames: Queue = Queue(maxsize=self.queue_frames)
        stderr_buf = bytearray()

        def _read_frames():
            try:
                for chunk in iter_fixed_chunks(proc.stdout, frame_bytes):
                    frames.put(chunk)
            finally:
                frames.put(None)

        def _drain_stderr():
            for line in proc.stderr:
                stderr_buf.extend(line)

        threading.Thread(target=_read_frames, daemon=True).start()
        threading.Thread(target=_drain_stderr, daemon=True).start()

        # inactivity timeout: a healthy decode (even of a long file) emits
        # frames steadily, so a gap this long means ffmpeg has wedged on a
        # pathological file — kill it rather than hang the run. Not a
        # total-scene cap, so legit long files decode fine.
        stall = self.scene_timeout or None
        i = 0
        try:
            while True:
                try:
                    chunk = frames.get(timeout=stall)
                except Empty:
                    raise SamplerError(
                        f"no frame for {stall:.0f}s on {path} "
                        "(stalled/pathological decode?) — killed"
                    )
                if chunk is None:
                    break
                arr = np.frombuffer(chunk, dtype=np.uint8).reshape(crop, crop, 3)
                yield round(i * self.interval, 3), arr
                i += 1
            rc = proc.wait()
            if rc != 0:
                raise SamplerError(
                    f"ffmpeg failed for {path}: "
                    f"{stderr_buf.decode(errors='replace')[:300]}"
                )
            if i == 0:
                raise SamplerError(f"no frames decoded for {path}")
        finally:
            if proc.poll() is None:  # consumer bailed early: stop decoding
                proc.kill()
                proc.wait()

    def _iter_frames_interval(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Stream frames over a pipe: no temp files, and ffmpeg decodes ahead
        (up to `queue_frames`) while the consumer embeds. Frame i is from
        source time ~i*interval — timestamps come from the frame index."""
        from PIL import Image as PILImage  # lazy

        if self.interval <= 0:
            return
        cmd = [
            self.ffmpeg,
            "-v", "error",
            *self._input_args(),
            "-i", path,
            "-vf", self._vf(),
            "-q:v", "3",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc

        frames: Queue = Queue(maxsize=self.queue_frames)
        stderr_buf = bytearray()

        def _read_frames():
            try:
                for jpg in iter_jpegs(proc.stdout):
                    frames.put(jpg)
            finally:
                frames.put(None)  # sentinel: stream finished (or died)

        def _drain_stderr():
            for line in proc.stderr:
                stderr_buf.extend(line)

        threading.Thread(target=_read_frames, daemon=True).start()
        threading.Thread(target=_drain_stderr, daemon=True).start()

        stall = self.scene_timeout or None  # inactivity cap (see raw path)
        i = 0
        try:
            while True:
                try:
                    jpg = frames.get(timeout=stall)
                except Empty:
                    raise SamplerError(
                        f"no frame for {stall:.0f}s on {path} "
                        "(stalled/pathological decode?) — killed"
                    )
                if jpg is None:
                    break
                with PILImage.open(BytesIO(jpg)) as im:
                    yield round(i * self.interval, 3), im.convert("RGB")
                i += 1
            rc = proc.wait()
            if rc != 0:
                raise SamplerError(
                    f"ffmpeg failed for {path}: "
                    f"{stderr_buf.decode(errors='replace')[:300]}"
                )
            if i == 0:
                raise SamplerError(f"no frames decoded for {path}")
        finally:
            if proc.poll() is None:  # consumer bailed early: stop decoding
                proc.kill()
                proc.wait()

    def _iter_frames_keyframes(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Keyframe-only pass: `-skip_frame nokey` decodes ~1-5% of frames.
        Exact timestamps are parsed from showinfo output (which logs at info
        level, hence `-v info` + post-run parse rather than streaming).

        Note: `-skip_frame nokey` is a *software* decoder feature; NVDEC/cuvid
        ignores it and would decode every frame, defeating the whole point. So
        this path deliberately does NOT use hwaccel — sparse I-frame decode on
        the CPU is already cheap (it's skipping ~95% of frames), and the GPU is
        still used for the embedding itself."""
        from PIL import Image as PILImage  # lazy

        tmpdir = Path(tempfile.mkdtemp(prefix="peaks-frames-"))
        try:
            cmd = [
                self.ffmpeg,
                "-v", "info",  # showinfo logs at info level
                "-nostats",
                "-skip_frame", "nokey",  # CPU decoder skips non-key frames
                "-i", path,
                "-vf", self._vf(),
                "-fps_mode", "passthrough",
                "-q:v", "3",
                str(tmpdir / "f-%06d.jpg"),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as exc:
                raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
            except subprocess.CalledProcessError as exc:
                raise SamplerError(
                    f"ffmpeg failed for {path}: {exc.stderr[-300:]}"
                ) from exc

            times = parse_showinfo_times(proc.stderr)
            files = sorted(tmpdir.glob("f-*.jpg"))
            if abs(len(times) - len(files)) > 2:
                raise SamplerError(
                    f"keyframe pts/frame mismatch for {path}: "
                    f"{len(times)} pts vs {len(files)} frames"
                )
            for ts, fp in zip(times, files):
                with PILImage.open(fp) as im:
                    yield ts, im.convert("RGB")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
