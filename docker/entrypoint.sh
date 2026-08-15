#!/bin/sh
# Seed /config on first run, refresh the bundled config defaults (without
# clobbering the user's settings), then hand off to the command.
set -e

mkdir -p /config/cache /config/models /config/references /config/torch /config/hf
cd /config

exec "$@"
