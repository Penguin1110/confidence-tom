#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher for Colab:
# 1) starts git autopush loop (background)
# 2) runs full small-only matrix queue in foreground by default
#
# Usage:
#   bash tools/start_colab_full_matrix.sh
#   AUTO_BRANCH=colab-runs MODELS=qwen35_9b,gemma3_27b bash tools/start_colab_full_matrix.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs results

AUTO_BRANCH="${AUTO_BRANCH:-}"
AUTO_INTERVAL_SEC="${AUTO_INTERVAL_SEC:-300}"
AUTO_MESSAGE_PREFIX="${AUTO_MESSAGE_PREFIX:-chore(colab): autosave}"
AUTO_PATHS="${AUTO_PATHS:-results logs}"
BACKUP_INTERVAL_SEC="${BACKUP_INTERVAL_SEC:-300}"
BACKUP_DIR="${BACKUP_DIR:-/content/drive/MyDrive/confidence-tom-backups}"
BACKUP_GCS_URI="${BACKUP_GCS_URI:-}"

MODELS="${MODELS:-all}"
BENCHMARKS="${BENCHMARKS:-olympiadbench,livebench_reasoning}"
OLYMPIAD_LIMIT="${OLYMPIAD_LIMIT:-0}"
LIVEBENCH_LIMIT="${LIVEBENCH_LIMIT:-0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-colab_l4_full}"
SMALL_BACKEND="${SMALL_BACKEND:-ollama}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-42}"
FULL_TRACE_SEC="${FULL_TRACE_SEC:-1200}"
SMALL_WORKER_SEC="${SMALL_WORKER_SEC:-900}"
TASK_SEC="${TASK_SEC:-7200}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-1}"
EXTRACTOR_ENABLED="${EXTRACTOR_ENABLED:-1}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-openai/gpt-5.4}"
EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND:-openrouter}"

if [[ -n "$AUTO_BRANCH" ]]; then
  git checkout "$AUTO_BRANCH"
fi

AUTOPUSH_LOG="logs/colab_autopush.log"
MATRIX_LOG="logs/colab_l4_full_matrix.log"
RUN_FOREGROUND="${RUN_FOREGROUND:-1}"
BACKUP_LOG="logs/colab_backup_loop.log"

nohup uv run python tools/colab_autopush.py \
  --repo . \
  --paths $AUTO_PATHS \
  --interval-sec "$AUTO_INTERVAL_SEC" \
  --message-prefix "$AUTO_MESSAGE_PREFIX" \
  > "$AUTOPUSH_LOG" 2>&1 &
AUTOPUSH_PID=$!

# Optional Google Drive backup loop (works when Drive is mounted in Colab)
BACKUP_PID=""
if [[ -n "$BACKUP_GCS_URI" ]]; then
  nohup bash -lc "
    set -euo pipefail
    cd '$ROOT_DIR'
    while true; do
      ts=\$(date '+%Y%m%d_%H%M%S')
      gsutil -m rsync -r results '${BACKUP_GCS_URI%/}/latest/results' || true
      gsutil -m rsync -r logs '${BACKUP_GCS_URI%/}/latest/logs' || true
      tar -czf /tmp/snapshot_\$ts.tar.gz results logs || true
      gsutil cp /tmp/snapshot_\$ts.tar.gz '${BACKUP_GCS_URI%/}/snapshots/' || true
      rm -f /tmp/snapshot_\$ts.tar.gz || true
      sleep '$BACKUP_INTERVAL_SEC'
    done
  " > "$BACKUP_LOG" 2>&1 &
  BACKUP_PID=$!
elif [[ -d "/content/drive/MyDrive" ]]; then
  mkdir -p "$BACKUP_DIR"
  nohup bash -lc "
    set -euo pipefail
    cd '$ROOT_DIR'
    while true; do
      ts=\$(date '+%Y%m%d_%H%M%S')
      mkdir -p '$BACKUP_DIR'/latest
      rsync -a --delete results/ '$BACKUP_DIR'/latest/results/
      rsync -a --delete logs/ '$BACKUP_DIR'/latest/logs/
      tar -czf '$BACKUP_DIR'/snapshot_\$ts.tar.gz results logs || true
      ls -1t '$BACKUP_DIR'/snapshot_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm -f
      sleep '$BACKUP_INTERVAL_SEC'
    done
  " > "$BACKUP_LOG" 2>&1 &
  BACKUP_PID=$!
fi

MATRIX_CMD=(
  uv run python experiments/run_prefix_small_only_full_matrix.py
  --models "$MODELS"
  --benchmarks "$BENCHMARKS"
  --olympiad-limit "$OLYMPIAD_LIMIT"
  --livebench-limit "$LIVEBENCH_LIMIT"
  --output-prefix "$OUTPUT_PREFIX"
  --small-backend "$SMALL_BACKEND"
  --enable-thinking "$ENABLE_THINKING"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --seed "$SEED"
  --full-trace-sec "$FULL_TRACE_SEC"
  --small-worker-sec "$SMALL_WORKER_SEC"
  --task-sec "$TASK_SEC"
  --task-concurrency "$TASK_CONCURRENCY"
)

if [[ "$EXTRACTOR_ENABLED" == "1" ]]; then
  MATRIX_CMD+=(
    --extractor-enabled
    --extractor-model "$EXTRACTOR_MODEL"
    --extractor-backend "$EXTRACTOR_BACKEND"
  )
fi

echo "STARTED"
echo "AUTOPUSH_PID=$AUTOPUSH_PID LOG=$AUTOPUSH_LOG"
if [[ -n "$BACKUP_PID" ]]; then
  if [[ -n "$BACKUP_GCS_URI" ]]; then
    echo "BACKUP_PID=$BACKUP_PID LOG=$BACKUP_LOG GCS=$BACKUP_GCS_URI"
  else
    echo "BACKUP_PID=$BACKUP_PID LOG=$BACKUP_LOG DIR=$BACKUP_DIR"
  fi
else
  echo "BACKUP=disabled (set BACKUP_GCS_URI=gs://bucket/path or mount Google Drive)"
fi
echo "MATRIX_LOG=$MATRIX_LOG"
echo "tail -n 120 $AUTOPUSH_LOG"
echo "tail -n 120 $BACKUP_LOG"
echo "tail -n 120 $MATRIX_LOG"

if [[ "$RUN_FOREGROUND" == "1" ]]; then
  echo "Running matrix in foreground (prevents Colab idle timeout while cell is active)..."
  "${MATRIX_CMD[@]}" 2>&1 | tee -a "$MATRIX_LOG"
else
  nohup "${MATRIX_CMD[@]}" > "$MATRIX_LOG" 2>&1 &
  MATRIX_PID=$!
  echo "MATRIX_PID=$MATRIX_PID"
fi
