#!/usr/bin/env bash
# =============================================================================
# LLM Phase 0 — Environment Setup for AWS g4dn.xlarge (NVIDIA T4 16 GB)
# =============================================================================
# Run as: bash llm/scripts/setup.sh
# Tested on: Ubuntu 22.04 LTS (Deep Learning AMI or vanilla)
# =============================================================================
set -euo pipefail

PYTHON_VERSION="3.11"
VENV_DIR="${HOME}/venv"
CUDA_VERSION="12.1"
# Detect Ubuntu release so the correct CUDA repo URL is used (22.04 or 24.04)
UBUNTU_RELEASE=$(lsb_release -rs | tr -d '.')   # e.g. 2204 or 2404
CUDA_TOOLKIT_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_RELEASE}/x86_64"

echo "========================================="
echo " Fraud Detection LLM — Environment Setup"
echo "========================================="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/8] Installing system dependencies..."
sudo apt-get update -qq
# software-properties-common provides add-apt-repository, needed for deadsnakes
sudo apt-get install -y --no-install-recommends software-properties-common

# Python 3.11 is not in Ubuntu 22.04 default repos — add the deadsnakes PPA
if ! dpkg -l python${PYTHON_VERSION} &>/dev/null; then
    echo "  Adding deadsnakes PPA for Python ${PYTHON_VERSION}..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
fi

sudo apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    libssl-dev libffi-dev \
    screen tmux htop \
    jq

# ---------------------------------------------------------------------------
# 2. CUDA toolkit (skip if already installed)
# ---------------------------------------------------------------------------
if ! command -v nvcc &>/dev/null; then
    echo "[2/8] Installing CUDA ${CUDA_VERSION} toolkit..."
    wget -q ${CUDA_TOOLKIT_URL}/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    # Install only the packages needed for PyTorch + bitsandbytes training.
    # Avoid the cuda-toolkit-12-1 meta-package: it pulls in nsight-systems which
    # depends on libtinfo5, a library removed in Ubuntu 24.04.
    sudo apt-get install -y --no-install-recommends \
        cuda-nvcc-12-1 \
        cuda-cudart-12-1 \
        cuda-cudart-dev-12-1 \
        libcublas-12-1 \
        libcublas-dev-12-1
    rm cuda-keyring_1.1-1_all.deb
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    # shellcheck disable=SC1090
    source ~/.bashrc
fi

# Install the NVIDIA display driver separately — required for nvidia-smi and
# bitsandbytes CUDA kernels. Driver 535 supports CUDA 12.x and works on both
# Ubuntu 22.04 and 24.04. Skip if already present.
if ! command -v nvidia-smi &>/dev/null; then
    echo "[2b/8] Installing NVIDIA driver 535..."
    sudo apt-get install -y --no-install-recommends \
        nvidia-driver-535 \
        nvidia-utils-535
    echo ""
    echo "================================================================"
    echo " REBOOT REQUIRED: NVIDIA driver was just installed."
    echo " Run: sudo reboot"
    echo " Then re-run: bash llm/scripts/setup.sh"
    echo " Steps 1-2 will be skipped automatically on the next run."
    echo "================================================================"
    exit 0
else
    echo "[2b/8] NVIDIA driver already present — skipping."
fi

# ---------------------------------------------------------------------------
# 3. Verify GPU
# ---------------------------------------------------------------------------
echo "[3/8] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "WARNING: nvidia-smi failed. Ensure you are on a GPU instance."
}

# ---------------------------------------------------------------------------
# 4. Python virtual environment
# ---------------------------------------------------------------------------
echo "[4/8] Creating Python virtual environment at ${VENV_DIR}..."
python${PYTHON_VERSION} -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ---------------------------------------------------------------------------
# 5. PyTorch with CUDA 12.1
# ---------------------------------------------------------------------------
echo "[5/8] Installing PyTorch with CUDA 12.1..."
pip install -q \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python - <<'PYEOF'
import torch
print(f"  PyTorch version : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM            : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
else:
    print("  WARNING: CUDA not available — training will be on CPU (very slow)")
PYEOF

# ---------------------------------------------------------------------------
# 6. LLM training packages
# ---------------------------------------------------------------------------
echo "[6/8] Installing LLM training packages..."
pip install -q \
    transformers==4.41.2 \
    datasets==2.19.1 \
    peft==0.11.1 \
    bitsandbytes==0.43.1 \
    accelerate==0.30.0 \
    trl==0.8.6 \
    sentencepiece \
    einops \
    scipy \
    evaluate \
    scikit-learn \
    rouge-score \
    bert-score \
    nltk

# ---------------------------------------------------------------------------
# 7. Experiment tracking
# ---------------------------------------------------------------------------
echo "[7/8] Installing experiment tracking (W&B + MLflow)..."
pip install -q wandb mlflow

# ---------------------------------------------------------------------------
# 8. Verify model can load in 4-bit
# ---------------------------------------------------------------------------
echo "[8/8] Verifying 4-bit quantized model loads without OOM..."
python - <<'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
print(f"  Loading {MODEL_ID} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  Model loaded successfully. VRAM used: {alloc:.2f} GB")
    else:
        print("  Model loaded on CPU (CUDA not available)")
    del model
except Exception as exc:
    print(f"  WARN: {MODEL_ID} failed to load: {exc}")
    print("  Falling back to microsoft/Phi-3-mini-4k-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  Fallback loaded. VRAM used: {alloc:.2f} GB")
    del model

print("  Environment setup COMPLETE.")
PYEOF

echo ""
echo "Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
echo "Next step: run python llm/scripts/prepare_data.py"
