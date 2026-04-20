#!/usr/bin/env bash
set -euo pipefail

# Launch the 4B Ollama matrix on the remote desktop machine.
#
# Required env vars:
#   REMOTE_SSH_HOST
#   REMOTE_SSH_USER
#   REMOTE_SSH_PASSWORD
#
# Optional env vars:
#   REMOTE_PROJECT_DIR   default: ~/confidence-tom
#   REMOTE_TIMEOUT       default: 7200
#   JOB_NAME             default: remote_4b_matrix
#   MODELS               default: qwen35_4b,qwen35_4b_fast,gemma3_4b
#   BENCHMARKS           default: olympiadbench,livebench_reasoning
#   OUTPUT_PREFIX        default: remote4b
#   OLLAMA_BASE_URL      default: http://127.0.0.1:11434/v1
#   EXTRACTOR_ENABLED    default: 1
#   DELETE_AFTER_RUN     default: 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load local env config if present so SSH and Ollama settings come from .env.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-7200}"
JOB_NAME="${JOB_NAME:-remote_4b_matrix}"
MODELS="${MODELS:-qwen35_4b,qwen35_4b_fast,gemma3_4b}"
BENCHMARKS="${BENCHMARKS:-olympiadbench,livebench_reasoning}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-remote4b}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/confidence-tom}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
EXTRACTOR_ENABLED="${EXTRACTOR_ENABLED:-1}"
DELETE_AFTER_RUN="${DELETE_AFTER_RUN:-0}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-openai/gpt-5.4}"
EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND:-openrouter}"

REMOTE_CMD=(
  bash
  -lc
  "cd ${REMOTE_PROJECT_DIR} && \
   export OLLAMA_BASE_URL=${OLLAMA_BASE_URL} && \
   export OLLAMA_API_KEY=ollama && \
   uv run python experiments/run_prefix_small_only_full_matrix.py \
     --models ${MODELS} \
     --benchmarks ${BENCHMARKS} \
     --olympiad-limit 0 \
     --livebench-limit 0 \
     --output-prefix ${OUTPUT_PREFIX} \
     --small-backend ollama \
     --enable-thinking false \
     --task-concurrency 1 \
     --retry-attempts 3 \
     --retry-backoff-sec 2.0 \
     --full-trace-sec 1200 \
     --small-worker-sec 900 \
     --task-sec 7200 \
     --pull-before-run \
"
)

if [[ "$EXTRACTOR_ENABLED" == "1" ]]; then
  REMOTE_CMD[-1]+=" --extractor-enabled --extractor-model ${EXTRACTOR_MODEL} --extractor-backend ${EXTRACTOR_BACKEND}"
fi

if [[ "$DELETE_AFTER_RUN" == "1" ]]; then
  REMOTE_CMD[-1]+=" --delete-after-run"
fi

exec uv run python experiments/run_remote_pipeline.py \
  --timeout "$REMOTE_TIMEOUT" \
  run-bg \
  --job-name "$JOB_NAME" \
  -- "${REMOTE_CMD[@]}"
