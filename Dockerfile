# peaks-vr — container image for Unraid / Docker (mirrors the 2D peaks image).
#
# The default process is the real-time flagging web UI (port 8760) plus the
# HereSphere "timestamp server" listener (port 23573) that the headset pushes
# playback into. embed / recommend / dj run from the container console.
#
# CUDA torch wheels + a static NVDEC-enabled ffmpeg are baked in so `embed`
# (the GPU-hungry de-warp + embedding pass) works with `--runtime=nvidia`; the
# flag UI + timestamp intake run CPU-only, so read/flagging works without a GPU.
# torch is the CUDA 12.8 build (cu128): kernels for Blackwell (RTX 50-series,
# sm_120) through Ampere/Ada, and it still runs on CPU when no GPU is present.
# Model weights (DINOv2/CLIP) download on first use into /config so they persist.

FROM python:3.11-slim

# static ffmpeg/ffprobe with full nvidia hwaccel (nvdec, cuvid) for the v360
# de-warp + high-res decode; the stock Debian build's nvidia support is not
# guaranteed. The NVDEC driver libs are injected at runtime by the nvidia
# container runtime (NVIDIA_DRIVER_CAPABILITIES must include "video").
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget xz-utils ca-certificates \
    && wget -qO /tmp/ffmpeg.tar.xz \
        https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && mkdir -p /tmp/ff && tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ff --strip-components=1 \
    && cp /tmp/ff/bin/ffmpeg /tmp/ff/bin/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ff /tmp/ffmpeg.tar.xz \
    && apt-get purge -y wget xz-utils && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# heavy ML deps pinned to CUDA 12.8 wheels (Blackwell → Ampere), CPU-safe too
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /opt/peaks-vr
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml,web]"

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/run-flag.sh /opt/peaks-vr/run-flag.sh
RUN chmod +x /entrypoint.sh /opt/peaks-vr/run-flag.sh

# persist model downloads + all working data under the /config volume
ENV TORCH_HOME=/config/torch \
    HF_HOME=/config/hf

WORKDIR /config
# 8760 = flagging web UI; 23573 = HereSphere timestamp-server intake
EXPOSE 8760 23573
ENTRYPOINT ["/entrypoint.sh"]
# default: the flagging UI, sourcing playback from HereSphere's timestamp server
# (the headset connects IN to 23573). run-flag.sh builds the invocation from
# PEAKS_VR_* env vars (see the unraid template / docs/DEPLOY.md) so ports,
# profile, and remote-vs-listen are set without editing the command. Other
# commands run via the container console, e.g.
#   docker exec -it peaks-vr peaks-vr --model dino embed /data/clip.mp4 --vr
CMD ["/opt/peaks-vr/run-flag.sh"]
