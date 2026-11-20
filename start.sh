#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec "${PAINTLAYER_PYTHON:-python3}" -m paintlayer
