#!/bin/bash
# Starts d1 (decide.py) and opens the page as soon as the model is ready. Ctrl-C stops it.
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv is missing: https://docs.astral.sh/uv/"; exit 1; }
PORT=${DECIDE_PORT:-8049}
export DECIDE_PORT=$PORT

uv run -q decide.py &
D1=$!
trap 'kill $D1 2>/dev/null' EXIT INT TERM

until curl -s -o /dev/null "http://127.0.0.1:$PORT/status"; do  # the first run downloads 2.2 GB
  kill -0 $D1 2>/dev/null || { echo "decide.py did not start"; exit 1; }
  sleep 1
done
open "http://127.0.0.1:$PORT"
wait $D1
