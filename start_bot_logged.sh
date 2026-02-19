#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OPENCLAW_LOG_DIR:-/tmp/openclaw}"
LOG_FILE="${OPENCLAW_LOG_FILE:-$LOG_DIR/maduro_bot_gateway.log}"
PID_FILE="${OPENCLAW_PID_FILE:-$LOG_DIR/maduro_bot_gateway.pid}"
TTS_BASE_URL="${OPENAI_TTS_BASE_URL:-http://127.0.0.1:8090}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Bot already running (pid=$OLD_PID). Log: $LOG_FILE"
    exit 0
  fi
fi

if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.nvm/nvm.sh"
  nvm use 22 >/dev/null
fi

nohup env OPENAI_TTS_BASE_URL="$TTS_BASE_URL" \
  node "$ROOT_DIR/dist/entry.js" gateway >"$LOG_FILE" 2>&1 < /dev/null &

NEW_PID="$!"
echo "$NEW_PID" >"$PID_FILE"
echo "Started bot (pid=$NEW_PID)"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
