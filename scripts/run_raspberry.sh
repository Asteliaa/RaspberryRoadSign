#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE="${1:-0}"
OUTPUT="${2:-output/raspberry_output.mp4}"
MODEL="${3:-models/deploy/best.pt}"
DEVICE="${4:-cpu}"

python scripts/main.py \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --device "$DEVICE"
