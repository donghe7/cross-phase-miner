#!/usr/bin/env bash
# Server + robot fleet + console; preserve learned data unless --fresh is explicit.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

PORT=8000
INTERSECTIONS=1200
ZONE=48
HUBS=8
ROBOTS=20
SPEED=5
DB="${DATABASE_URL:-${CP_DB:-server_state.sqlite3}}"
FRESH=()
POSTGRES=false
MQTT=false
EXPLICIT_DB=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)          PORT="$2"; shift 2 ;;
    --intersections) INTERSECTIONS="$2"; shift 2 ;;
    --zone)          ZONE="$2"; shift 2 ;;
    --hubs)          HUBS="$2"; shift 2 ;;
    --robots)        ROBOTS="$2"; shift 2 ;;
    --speed)         SPEED="$2"; shift 2 ;;
    --db)            DB="$2"; EXPLICIT_DB=true; shift 2 ;;
    --postgres)      POSTGRES=true; shift ;;
    --mqtt)          MQTT=true; shift ;;
    --fresh)         FRESH=(--fresh); shift ;;
    --resume)        FRESH=(); shift ;;
    -h|--help)
      cat <<'HELP'
Usage: ./run.sh [options]
  --postgres         Start local PostgreSQL Docker service and use it
  --mqtt             Start local MQTT broker; publish telemetry and subscribe to clock
  --db PATH_OR_URL   Use a SQLite file or PostgreSQL URL (or set DATABASE_URL)
  --fresh            Explicitly clear CrossPhase tables before starting
  --resume           Preserve learned data (the default)
  --port N           Web console port (default 8000)
  --intersections N  Managed crossings (default 1200)
  --zone N           Delivery-zone crossings (default 48)
  --hubs N           Shared hub crossings (default 8)
  --robots N         Robot count (default 20)
  --speed N          Initial speed: 1, 2, 5, 10, 20 (default 5)
Ctrl-C stops the server and fleet. PostgreSQL and MQTT containers keep running.
HELP
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$SPEED" in
  1|2|5|10|20) ;;
  *) echo 'Speed must be one of 1, 2, 5, 10, 20.' >&2; exit 2 ;;
esac
if $POSTGRES && $EXPLICIT_DB; then
  echo 'Use either --postgres or --db, not both.' >&2
  exit 2
fi
python3 - <<'CHECK'
import sys
try:
    import fastapi, uvicorn, numpy, websockets
except ImportError as error:
    sys.exit(f"missing dependency: {error.name}\n"
             "install with: pip install -e '.[server]'")
CHECK
# Refuse an occupied port before starting/resetting any application database.
python3 - "$PORT" <<'CHECK'
import socket
import sys
try:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', int(sys.argv[1])))
except (OSError, ValueError) as error:
    sys.exit(f"Cannot start console on port {sys.argv[1]}: {error}")
CHECK
if $POSTGRES; then
  python3 -c 'import psycopg' || { echo 'Install: pip install -e '.[server]'' >&2; exit 1; }
  ./server/postgres.sh up
  DB="$(python3 -m server.postgres_env url --file "${CP_POSTGRES_ENV:-$PWD/server/.env.postgres}")"
fi
if $MQTT; then
  python3 -c 'import paho.mqtt.client' || { echo 'Install: pip install -e '.[server]'' >&2; exit 1; }
  docker compose -f server/compose.mqtt.yaml up -d --wait --wait-timeout 60
  export CP_MQTT=1
fi
# Pass credentials via the environment, not process arguments or console output.
export CP_DB="$DB"
unset DATABASE_URL
export CP_ROBOTS="$ROBOTS"

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${FLEET_PID:-}" ]] && kill "$FLEET_PID" 2>/dev/null || true
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 -m server.app --port "$PORT" --intersections "$INTERSECTIONS" \
  --zone "$ZONE" --hubs "$HUBS" --speed "$SPEED" "${FRESH[@]}" &
SERVER_PID=$!
READY=false
for _ in $(seq 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo 'Server failed to start; fleet was not started.' >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/clock" >/dev/null 2>&1; then
    READY=true
    break
  fi
  sleep 0.5
done
if ! $READY; then
  echo 'Server readiness timed out; fleet was not started.' >&2
  exit 1
fi
python3 -m server.fleet --server "http://127.0.0.1:${PORT}" --robots "$ROBOTS" &
FLEET_PID=$!
if [[ "$DB" == postgres://* || "$DB" == postgresql://* ]]; then
  DB_LABEL='PostgreSQL (persistent; credentials hidden)'
else
  DB_LABEL="SQLite: $DB"
fi
cat <<INFO

  console      http://127.0.0.1:${PORT}
  crossings    ${INTERSECTIONS} managed, ${ZONE} in the delivery zone
  fleet        ${ROBOTS} robots, clock x${SPEED}
  database     ${DB_LABEL}

  Learned data is retained by default. Use --fresh only for an explicit reset.
  Ctrl-C stops the server and fleet; PostgreSQL remains available.

INFO
wait -n "$SERVER_PID" "$FLEET_PID"
