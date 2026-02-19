#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${OPENCLAW_LOG_DIR:-/tmp/openclaw}"
LOG_FILE="${OPENCLAW_LOG_FILE:-$LOG_DIR/maduro_bot_gateway.log}"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

echo "Tailing: $LOG_FILE"
tail -n 20 -f "$LOG_FILE"
