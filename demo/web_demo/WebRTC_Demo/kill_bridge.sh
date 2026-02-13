#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${BRIDGE_PORT:-8090}"
PID_FILE="$SCRIPT_DIR/.pids/bridge_service.pid"
LABEL_FILE="$SCRIPT_DIR/.pids/bridge_service.label"
LAUNCH_LABEL="${BRIDGE_LAUNCH_LABEL:-minicpm.bridge_service}"

killed_any=0

kill_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  echo "[bridge:kill] stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      killed_any=1
      return
    fi
    sleep 0.2
  done

  echo "[bridge:kill] force killing pid=$pid"
  kill -9 "$pid" 2>/dev/null || true
  killed_any=1
}

if [[ -f "$PID_FILE" ]]; then
  pid_from_file="$(cat "$PID_FILE" 2>/dev/null || true)"
  kill_pid "$pid_from_file"
  rm -f "$PID_FILE"
fi

if [[ -f "$LABEL_FILE" ]]; then
  label_from_file="$(cat "$LABEL_FILE" 2>/dev/null || true)"
  if [[ -n "$label_from_file" ]] && command -v launchctl >/dev/null 2>&1; then
    if launchctl list | awk '{print $3}' | grep -Fxq "$label_from_file"; then
      echo "[bridge:kill] removing launchctl label=$label_from_file"
      launchctl remove "$label_from_file" >/dev/null 2>&1 || true
      killed_any=1
    fi
  fi
  rm -f "$LABEL_FILE"
fi

if command -v launchctl >/dev/null 2>&1; then
  if launchctl list | awk '{print $3}' | grep -Fxq "$LAUNCH_LABEL"; then
    echo "[bridge:kill] removing launchctl label=$LAUNCH_LABEL"
    launchctl remove "$LAUNCH_LABEL" >/dev/null 2>&1 || true
    killed_any=1
  fi
fi

while IFS= read -r pid; do
  kill_pid "$pid"
done < <(pgrep -f "$SCRIPT_DIR/bridge_service.py" 2>/dev/null || true)

while IFS= read -r pid; do
  kill_pid "$pid"
done < <(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)

if [[ "$killed_any" -eq 1 ]]; then
  echo "[bridge:kill] bridge stopped"
else
  echo "[bridge:kill] no running bridge found"
fi
