#!/usr/bin/env bash
set -euo pipefail

# Raspberry deploy helper (run on target Raspberry Pi)
# Usage:
#   ./scripts/deploy.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_raspberry.txt

echo "Deploy complete."
echo "Run: ./scripts/run_raspberry.sh 0 output/raspberry_output.mp4 models/deploy/best.pt cpu"
