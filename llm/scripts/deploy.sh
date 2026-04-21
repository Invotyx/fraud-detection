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
BASE_MODEL_ID="${BASE_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
DATASET_COUNT="${DATASET_COUNT:-600}"
LLM_SERVER_PORT="${LLM_SERVER_PORT:-8001}"
SERVER_URL="${LLM_SERVER_URL:-http://localhost:${LLM_SERVER_PORT}}"
SERVER_STARTUP_WAIT_SECONDS="${SERVER_STARTUP_WAIT_SECONDS:-60}"

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

cd "${PROJECT_ROOT}"

# =============================================================================
# Phase 0 — Environment Setup
# =============================================================================
step "Phase 0: Environment Setup"
bash llm/scripts/setup.sh
done_step "setup.sh"

# Activate virtual environment for all subsequent Python steps
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# =============================================================================
# Phase 1 — Dataset Preparation
# =============================================================================
step "Phase 1: Dataset Preparation"
python llm/scripts/prepare_data.py \
    --output-dir "${DATA_DIR}" \
    --count "${DATASET_COUNT}"
done_step "prepare_data.py"

# =============================================================================
# Phase 2 — System Prompt Adversarial Testing (pre-training baseline)
# =============================================================================
step "Phase 2: System Prompt Adversarial Testing (pre-training baseline)"
python llm/scripts/test_system_prompt.py \
    --local \
    --model "${BASE_MODEL_ID}"
done_step "test_system_prompt.py"

# =============================================================================
# Phase 3 — Fine-Tuning Run 1: Resilient Behavior
# =============================================================================
step "Phase 3: Fine-Tuning Run 1 — Resilient Behavior"
python llm/scripts/train_run1.py \
    --config llm/configs/run1_config.yaml
done_step "train_run1.py"

# =============================================================================
# Phase 4 — Fine-Tuning Run 2: Fraud Detection Accuracy
# =============================================================================
step "Phase 4: Fine-Tuning Run 2 — Fraud Detection Accuracy"
python llm/scripts/train_run2.py \
    --config llm/configs/run2_config.yaml \
    --from-checkpoint "${CHECKPOINTS_DIR}/run1/final"
done_step "train_run2.py"

# =============================================================================
# Phase 5 — Targeted Fix (weak parameter remediation)
# =============================================================================
step "Phase 5: Targeted Fix — Weak Parameter Remediation"
python llm/scripts/targeted_fix.py \
    --eval-results "${CHECKPOINTS_DIR}/run2/eval_results.json" \
    --checkpoint "${CHECKPOINTS_DIR}/run2/final" \
    --output-dir "${CHECKPOINTS_DIR}/run3"
done_step "targeted_fix.py"

# =============================================================================
# Phase 6 — Merge LoRA and Serve
# =============================================================================
step "Phase 6: Merge LoRA Adapter and Start Inference Server"
python llm/scripts/merge_and_serve.py \
    --checkpoint "${CHECKPOINTS_DIR}/run2/final" \
    --merged-dir "${CHECKPOINTS_DIR}/final_merged" \
    --port 8001 &

echo "Waiting ${SERVER_STARTUP_WAIT_SECONDS}s for the server to become ready..."
sleep "${SERVER_STARTUP_WAIT_SECONDS}"
done_step "merge_and_serve.py (server running in background)"

# =============================================================================
# Phase 7a — Evaluation on held-out test set
# =============================================================================
step "Phase 7a: Evaluation — Held-Out Test Set"
python llm/scripts/eval.py \
    --server-url "${SERVER_URL}" \
    --test-data "${DATA_DIR}/test.jsonl" \
    --output "${CHECKPOINTS_DIR}/run2/eval_results.json"
done_step "eval.py"

# =============================================================================
# Phase 7b — Red-Team Adversarial Probing
# =============================================================================
step "Phase 7b: Red-Team Adversarial Probing"
python llm/scripts/red_team.py \
    --server-url "${SERVER_URL}" \
    --output "${CHECKPOINTS_DIR}/red_team_report.json"
done_step "red_team.py"

# =============================================================================
# Phase 8 — Harden (checksums, systemd, CloudWatch)
# =============================================================================
step "Phase 8: Harden — Lock Model, Checksums, systemd, CloudWatch"
bash llm/scripts/harden.sh \
    --merged-dir "${CHECKPOINTS_DIR}/final_merged" \
    --port "${LLM_SERVER_PORT}"
done_step "harden.sh"

# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} All phases completed successfully.${NC}"
echo -e "${GREEN} Inference server : ${SERVER_URL}${NC}"
echo -e "${GREEN} Eval results     : ${CHECKPOINTS_DIR}/run2/eval_results.json${NC}"
echo -e "${GREEN} Red-team report  : ${CHECKPOINTS_DIR}/red_team_report.json${NC}"
echo -e "${GREEN}============================================================${NC}"
