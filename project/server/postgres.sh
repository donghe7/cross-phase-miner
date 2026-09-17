#!/usr/bin/env bash
# Local PostgreSQL lifecycle. No command here deletes the data volume.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PG_ENV="${CP_POSTGRES_ENV:-$PWD/server/.env.postgres}"
COMMAND="${1:-up}"
case "$COMMAND" in
  up|down|status|psql|backup) ;;
  *) echo 'Usage: ./server/postgres.sh {up|down|status|psql|backup}' >&2; exit 2 ;;
esac
python3 -m server.postgres_env init --file "$PG_ENV"
COMPOSE=(docker compose --env-file "$PG_ENV" -f server/compose.postgres.yaml)
case "$COMMAND" in
  up) "${COMPOSE[@]}" up -d --wait --wait-timeout 90 postgres ;;
  down) "${COMPOSE[@]}" down ;;
  status) "${COMPOSE[@]}" ps ;;
  psql) "${COMPOSE[@]}" exec postgres sh -c 'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' ;;
  backup)
    umask 077
    mkdir -p server/backups
    BACKUP_FILE="server/backups/crossphase-$(date +%Y%m%d-%H%M%S)-$$.dump"
    if "${COMPOSE[@]}" exec -T postgres sh -c 'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' >"$BACKUP_FILE"; then
      echo "Backup saved: $BACKUP_FILE"
    else
      echo "Backup failed; incomplete file: $BACKUP_FILE" >&2
      exit 1
    fi
    ;;
esac
