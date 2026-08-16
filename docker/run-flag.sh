#!/bin/sh
# Launch the peaks-vr control panel + flagging UI, building the invocation from
# PEAKS_VR_* env vars so the unraid template / compose can set knobs without
# overriding the command.
#
#   PEAKS_VR_WEB_PORT   UI port (default 8801)
#   PEAKS_VR_TS_PORT    HereSphere timestamp-server intake port (default 23573)
#   PEAKS_VR_MEDIA      library dir (default /data) — enables the Embed tab
#   PEAKS_VR_MODEL      embedder for the Embed tab: dino (default) | clip | fake
#   PEAKS_VR_PROFILE    taste profile the ❤️ marks belong to (default apex)
#   PEAKS_VR_BYTEORDER  length-prefix endianness: big (default) | little
#   PEAKS_VR_REMOTE_HOST if set, DIAL the DeoVR remote at this IP instead of
#                        listening for the timestamp server (read+control)
#   PEAKS_VR_REMOTE_PORT DeoVR remote port (default 23554)
set -e

WEB_PORT="${PEAKS_VR_WEB_PORT:-8801}"
TS_PORT="${PEAKS_VR_TS_PORT:-23573}"
MEDIA="${PEAKS_VR_MEDIA:-/data}"
MODEL="${PEAKS_VR_MODEL:-dino}"
PROFILE="${PEAKS_VR_PROFILE:-apex}"
BYTEORDER="${PEAKS_VR_BYTEORDER:-big}"

set -- flag --web-host 0.0.0.0 --web-port "$WEB_PORT" \
       --profile "$PROFILE" --byteorder "$BYTEORDER" \
       --labels /config/labels.json --media "$MEDIA"

if [ -n "$PEAKS_VR_REMOTE_HOST" ]; then
    set -- "$@" --host "$PEAKS_VR_REMOTE_HOST" \
           --port "${PEAKS_VR_REMOTE_PORT:-23554}"
else
    set -- "$@" --listen --ts-port "$TS_PORT"
fi

# --model is a global option (before the subcommand); --cache defaults under /config
echo "starting: peaks-vr --model $MODEL --cache /config/cache $*"
exec peaks-vr --model "$MODEL" --cache /config/cache "$@"
