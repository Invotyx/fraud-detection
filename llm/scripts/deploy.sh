#!/usr/bin/env bash
# =============================================================================
# Fraud Detection LLM — Full Deployment Pipeline
# Runs all phases in sequence from environment setup to production hardening.
#
# Usage
#   chmod +x llm/scripts/deploy.sh
#   bash llm/scripts/deploy.sh
#
# Override any variable via environment or by editing the section below.
# You can also copy llm/.env.example → llm/.env and this script will pick it up.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load llm/.env if present
ENV_FILE="${PROJECT_ROOT}/llm/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -o allexport; source "${ENV_FILE}"; set +o allexport
fi

# ---------------------------------------------------------------------------
# Configurable variables — override via env or llm/.env
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-${HOME}/venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/llm/data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_ROOT}/checkpoints}"
STATE_DIR="${STATE_DIR:-${PROJECT_ROOT}/.deploy_state}"
BASE_MODEL_ID="${BASE_MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET_COUNT="${DATASET_COUNT:-600}"
LLM_SERVER_PORT="${LLM_SERVER_PORT:-8001}"
SERVER_URL="${LLM_SERVER_URL:-http://localhost:${LLM_SERVER_PORT}}"
SERVER_STARTUP_WAIT_SECONDS="${SERVER_STARTUP_WAIT_SECONDS:-300}"
PROMPT_TEST_STRICT="${PROMPT_TEST_STRICT:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Colours for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

step() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

done_step() {
    echo -e "${GREEN}[DONE] $1${NC}"
}

warn_step() {
    echo -e "${CYAN}[WARN] $1${NC}"
}

# Poll server health endpoint until ready or timeout
wait_for_server() {
    local url="$1"
    local timeout="${2:-300}"
    local interval=5
    local elapsed=0
    echo "Waiting for server at ${url}/health (timeout ${timeout}s)..."
    while [[ ${elapsed} -lt ${timeout} ]]; do
        if curl -sf "${url}/health" >/dev/null 2>&1; then
            echo "Server is ready (${elapsed}s elapsed)"
            return 0
        fi
        sleep ${interval}
        elapsed=$((elapsed + interval))
    done
    echo "ERROR: Server did not become ready within ${timeout}s"
    return 1
}

cd "${PROJECT_ROOT}"
mkdir -p "${STATE_DIR}"

# =============================================================================
# Phase 0 — Environment Setup
# =============================================================================
step "Phase 0: Environment Setup"
if [[ -f "${STATE_DIR}/phase0.done" ]]; then
    warn_step "Phase 0 already completed — skipping"
else
    bash llm/scripts/setup.sh
    touch "${STATE_DIR}/phase0.done"
    done_step "setup.sh"
fi

# Always activate venv (even when Phase 0 is skipped)
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# =============================================================================
# Phase 1 — Dataset Preparation
# =============================================================================
step "Phase 1: Dataset Preparation"
if [[ -f "${STATE_DIR}/phase1.done" ]]; then
    warn_step "Phase 1 already completed — skipping"
else
    python llm/scripts/prepare_data.py \
        --output-dir "${DATA_DIR}" \
        --count "${DATASET_COUNT}"
    touch "${STATE_DIR}/phase1.done"
    done_step "prepare_data.py"
fi

# =============================================================================
# Phase 2 — System Prompt Adversarial Testing (pre-training baseline)
# =============================================================================
step "Phase 2: System Prompt Adversarial Testing (pre-training baseline)"
if [[ -f "${STATE_DIR}/phase2.done" ]]; then
    warn_step "Phase 2 already completed — skipping"
else
    if python llm/scripts/test_system_prompt.py \
        --local \
        --model "${BASE_MODEL_ID}" \
        --verbose; then
        touch "${STATE_DIR}/phase2.done"
        done_step "test_system_prompt.py"
    else
        if [[ "${PROMPT_TEST_STRICT}" == "1" ]]; then
            echo "System prompt baseline failed and PROMPT_TEST_STRICT=1; stopping deployment."
            exit 1
        fi
        touch "${STATE_DIR}/phase2.done"
        warn_step "test_system_prompt.py failed baseline threshold; continuing (advisory)"
    fi
fi

# =============================================================================
# Phase 3 — Fine-Tuning Run 1: Resilient Behavior
# =============================================================================
step "Phase 3: Fine-Tuning Run 1 — Resilient Behavior"
if [[ -f "${STATE_DIR}/phase3.done" ]]; then
    warn_step "Phase 3 already completed — skipping"
else
    # torchrun launches one process per GPU; FSDP shards the fp16 model across all 4 T4s.
    # Effective batch = per_device_batch(1) × num_gpus(4) × grad_accum(4) = 16 (same as QLoRA baseline)
    torchrun --nproc_per_node=4 \
        llm/scripts/train_run1.py --config llm/configs/run1_config.yaml
    touch "${STATE_DIR}/phase3.done"
    done_step "train_run1.py"
fi

# =============================================================================
# Phase 4 — Fine-Tuning Run 2: Fraud Detection Accuracy
# =============================================================================
step "Phase 4: Fine-Tuning Run 2 — Fraud Detection Accuracy"
if [[ -f "${STATE_DIR}/phase4.done" ]]; then
    warn_step "Phase 4 already completed — skipping"
else
    torchrun --nproc_per_node=4 \
        llm/scripts/train_run2.py \
        --config llm/configs/run2_config.yaml \
        --from-checkpoint "${CHECKPOINTS_DIR}/run1/final"
    touch "${STATE_DIR}/phase4.done"
    done_step "train_run2.py"
fi

# =============================================================================
# Phase 5 — Targeted Fix (weak parameter remediation)
# =============================================================================
step "Phase 5: Targeted Fix — Weak Parameter Remediation"
if [[ -f "${STATE_DIR}/phase5.done" ]]; then
    warn_step "Phase 5 already completed — skipping"
else
    torchrun --nproc_per_node=4 \
        llm/scripts/targeted_fix.py \
        --eval-results "${CHECKPOINTS_DIR}/run2/eval_results.json" \
        --checkpoint "${CHECKPOINTS_DIR}/run2/final" \
        --output-dir "${CHECKPOINTS_DIR}/run3"
    touch "${STATE_DIR}/phase5.done"
    done_step "targeted_fix.py"
fi

# =============================================================================
# Phase 6 — Merge LoRA and Serve
# =============================================================================
step "Phase 6: Merge LoRA Adapter and Start Inference Server"

# Kill any stale vLLM process so it releases GPU memory before we start fresh.
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
    warn_step "Killing existing vLLM process to free GPU memory..."
    pkill -TERM -f "vllm.entrypoints.openai.api_server" || true
    # Wait up to 15s for the process to exit and CUDA context to be released
    for _i in $(seq 1 15); do
        pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || break
        sleep 1
    done
    pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3  # allow CUDA driver to reclaim memory
fi

if [[ -f "${STATE_DIR}/phase6.done" ]]; then
    warn_step "Phase 6 merge already completed — starting server against existing merged dir"
    python llm/scripts/merge_and_serve.py \
        --checkpoint "${CHECKPOINTS_DIR}/run2/final" \
        --merged-dir "${CHECKPOINTS_DIR}/final_merged" \
        --port 8001 \
        --skip-merge &
else
    python llm/scripts/merge_and_serve.py \
        --checkpoint "${CHECKPOINTS_DIR}/run2/final" \
        --merged-dir "${CHECKPOINTS_DIR}/final_merged" \
        --port 8001 &
    touch "${STATE_DIR}/phase6.done"
fi

echo "Waiting for vLLM server to become ready (timeout ${SERVER_STARTUP_WAIT_SECONDS}s)..."
wait_for_server "${SERVER_URL}" "${SERVER_STARTUP_WAIT_SECONDS}"
done_step "merge_and_serve.py (server running in background)"

# =============================================================================
# Phase 7a — Evaluation on held-out test set
# =============================================================================
step "Phase 7a: Evaluation — Held-Out Test Set"
if [[ -f "${STATE_DIR}/phase7a.done" ]]; then
    warn_step "Phase 7a already completed — skipping"
else
    python llm/scripts/eval.py \
        --server-url "${SERVER_URL}" \
        --test-data "${DATA_DIR}/test.jsonl" \
        --timeout 30 \
        --workers 4 \
        --output "${CHECKPOINTS_DIR}/run2/eval_results.json"
    touch "${STATE_DIR}/phase7a.done"
    done_step "eval.py"
fi

# =============================================================================
# Phase 7b — Red-Team Adversarial Probing
# =============================================================================
step "Phase 7b: Red-Team Adversarial Probing"
if [[ -f "${STATE_DIR}/phase7b.done" ]]; then
    warn_step "Phase 7b already completed — skipping"
else
    python llm/scripts/red_team.py \
        --server-url "${SERVER_URL}" \
        --output "${CHECKPOINTS_DIR}/red_team_report.json"
    touch "${STATE_DIR}/phase7b.done"
    done_step "red_team.py"
fi

# =============================================================================
# Phase 8 — Harden (checksums, systemd, CloudWatch)
# =============================================================================
step "Phase 8: Harden — Lock Model, Checksums, systemd, CloudWatch"
if [[ -f "${STATE_DIR}/phase8.done" ]]; then
    warn_step "Phase 8 already completed — skipping"
else
    bash llm/scripts/harden.sh \
        --merged-dir "${CHECKPOINTS_DIR}/final_merged" \
        --port "${LLM_SERVER_PORT}"
    touch "${STATE_DIR}/phase8.done"
    done_step "harden.sh"
fi

# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} All phases completed successfully.${NC}"
echo -e "${GREEN} Inference server : ${SERVER_URL}${NC}"
echo -e "${GREEN} Eval results     : ${CHECKPOINTS_DIR}/run2/eval_results.json${NC}"
echo -e "${GREEN} Red-team report  : ${CHECKPOINTS_DIR}/red_team_report.json${NC}"
echo -e "${GREEN}============================================================${NC}"
