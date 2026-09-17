#!/usr/bin/env bash
# Offline experiments only. The online console starts with server/run_demo.sh.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$PWD"
PYTHON="${PYTHON:-python3}"
VERSION="${1:-v1}"
if [[ $# -gt 0 ]]; then shift; fi
case "$VERSION" in
  v1|v2) ;;
  *) echo 'Usage: ./run.sh {v1|v2} [test] [simulation options]' >&2; exit 2 ;;
esac

if [[ "$VERSION" == v2 ]]; then
  # Historical learning rules run in isolation from the online algorithm.
  LEGACY_WORK="$(mktemp -d)"
  trap 'rm -rf -- "$LEGACY_WORK"' EXIT
  tar -xzf experiments/archive/legacy-20260722.tar.gz -C "$LEGACY_WORK" \
    --wildcards 'crossphase_miner_2/crossphase_miner/*' 'crossphase_miner_2/main.py'
  cd "$LEGACY_WORK/crossphase_miner_2"
fi
export PYTHONPATH="$PWD"

if [[ "${1:-}" == test ]]; then
  if [[ "$VERSION" == v1 ]]; then
    "$PYTHON" -m unittest tests.test_core -v
  else
    "$PYTHON" main.py
  fi
  exit
fi

RUN_DIR="$PROJECT_ROOT/experiments/runs/$VERSION/$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p "$RUN_DIR/simulation" "$RUN_DIR/evaluation"
EXPERIMENT_SEED="${SEED:-2024}"
echo "Offline $VERSION experiment: $RUN_DIR (seed=$EXPERIMENT_SEED)"
"$PYTHON" -m crossphase_miner.cli.simulate --output-dir "$RUN_DIR/simulation" \
  --seed "$EXPERIMENT_SEED" "$@" 2>&1 | tee "$RUN_DIR/simulation.log"
"$PYTHON" -m crossphase_miner.cli.evaluate --data-dir "$RUN_DIR/simulation" \
  --output-dir "$RUN_DIR/evaluation" 2>&1 | tee "$RUN_DIR/evaluation/evaluation.log"
echo "Results: $RUN_DIR"
