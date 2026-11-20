#!/bin/bash
# Starts d1 and opens the URL it binds. Ctrl-C stops it.
cd "$(dirname "$0")" || exit 1
command -v uv >/dev/null || { echo "uv is missing: https://docs.astral.sh/uv/"; exit 1; }
exec uv run -q decide.py --open
