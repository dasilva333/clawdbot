#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${BRIDGE_PORT:-8090}"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/.logs"
PID_FILE="$PID_DIR/bridge_service.pid"
LABEL_FILE="$PID_DIR/bridge_service.label"
LOG_FILE="$LOG_DIR/bridge_service.log"
LAUNCH_LABEL="${BRIDGE_LAUNCH_LABEL:-minicpm.bridge_service}"
START_MODE="${BRIDGE_START_MODE:-auto}"

mkdir -p "$PID_DIR" "$LOG_DIR"

if [[ -d "$SCRIPT_DIR/venv" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[bridge:start] python3 not found"
  exit 1
fi

if [[ "$START_MODE" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]] && command -v launchctl >/dev/null 2>&1; then
    START_MODE="launchctl"
  else
    START_MODE="nohup"
  fi
fi

if [[ "$START_MODE" == "launchctl" ]]; then
  if launchctl list | awk '{print $3}' | grep -Fxq "$LAUNCH_LABEL"; then
    echo "[bridge:start] already running (label=$LAUNCH_LABEL)"
    exit 0
  fi
else
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[bridge:start] already running (pid=$old_pid)"
      exit 0
    fi
    rm -f "$PID_FILE"
  fi
fi

port_pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$port_pids" ]]; then
  echo "[bridge:start] port $PORT already in use by pid(s):"
  echo "$port_pids"
  echo "[bridge:start] run ./kill_bridge.sh first or set BRIDGE_PORT"
  exit 1
fi

if [[ "$START_MODE" == "launchctl" ]]; then
  launchctl remove "$LAUNCH_LABEL" >/dev/null 2>&1 || true
  launchctl submit -l "$LAUNCH_LABEL" -- /bin/bash -lc "exec \"$PYTHON_BIN\" \"$SCRIPT_DIR/bridge_service.py\" >> \"$LOG_FILE\" 2>&1"
  echo "$LAUNCH_LABEL" > "$LABEL_FILE"
  rm -f "$PID_FILE"
else
  rm -f "$LABEL_FILE"
  nohup "$PYTHON_BIN" "$SCRIPT_DIR/bridge_service.py" > "$LOG_FILE" 2>&1 &
  bridge_pid=$!
  echo "$bridge_pid" > "$PID_FILE"
fi

sleep 1
if [[ "$START_MODE" == "launchctl" ]]; then
  if ! launchctl list | awk '{print $3}' | grep -Fxq "$LAUNCH_LABEL"; then
    echo "[bridge:start] failed to start launchctl job '$LAUNCH_LABEL'. Recent logs:"
    tail -n 80 "$LOG_FILE" || true
    rm -f "$LABEL_FILE"
    exit 1
  fi
else
  if ! kill -0 "$bridge_pid" 2>/dev/null; then
    echo "[bridge:start] failed to start. Recent logs:"
    tail -n 80 "$LOG_FILE" || true
    rm -f "$PID_FILE"
    exit 1
  fi
fi

if [[ "$START_MODE" == "launchctl" ]]; then
  echo "[bridge:start] started label=$LAUNCH_LABEL mode=launchctl log=$LOG_FILE"
else
  echo "[bridge:start] started pid=$bridge_pid mode=nohup log=$LOG_FILE"
fi
if command -v curl >/dev/null 2>&1; then
  health="$(curl -sS -m 3 "http://127.0.0.1:${PORT}/health" || true)"
  if [[ -n "$health" ]]; then
    echo "[bridge:start] health: $health"
  fi
fi
