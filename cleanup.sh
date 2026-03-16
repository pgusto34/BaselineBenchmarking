#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Removing checkpoints and tensorboard_logs..."
rm -rf "$SCRIPT_DIR/checkpoints" "$SCRIPT_DIR/tensorboard_logs"
echo "Done."
