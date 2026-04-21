#!/usr/bin/env bash
# LLM Phase 8 — Harden Script
# Locks the final model version, writes checksums, configures systemd,
# and sets up CloudWatch metrics.
#
# Usage
#   chmod +x llm/scripts/harden.sh
#   ./llm/scripts/harden.sh [--merged-dir <path>] [--port <port>]

set -euo pipefail

MERGED_DIR="${MERGED_DIR:-checkpoints/final_merged}"
PORT="${PORT:-8001}"
SERVICE_NAME="fraud-llm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLM_ROOT="${PROJECT_ROOT}/llm"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --merged-dir) MERGED_DIR="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Harden: Fraud Detection LLM ==="
echo "  Merged dir : ${MERGED_DIR}"
echo "  Port       : ${PORT}"

# ---------------------------------------------------------------------------
# Step 1: Compute and record model SHA-256
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Computing model weight checksums..."

CHECKSUM_FILE="${MERGED_DIR}/model_checksums.sha256"
> "${CHECKSUM_FILE}"

find "${MERGED_DIR}" -name "*.safetensors" -o -name "*.bin" | sort | while read -r f; do
  sha256sum "${f}" >> "${CHECKSUM_FILE}"
done

MODEL_HASH=$(sha256sum "${CHECKSUM_FILE}" | awk '{print $1}')
echo "  Combined SHA-256: ${MODEL_HASH}"
echo "${MODEL_HASH}" > "${MERGED_DIR}/model_hash.txt"
echo "  Checksums written → ${CHECKSUM_FILE}"

# ---------------------------------------------------------------------------
# Step 2: Write merge_info.json if not present
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Recording merge metadata..."

MERGE_INFO="${MERGED_DIR}/merge_info.json"
if [[ ! -f "${MERGE_INFO}" ]]; then
  cat > "${MERGE_INFO}" <<JSON
{
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "checkpoint": "checkpoints/run2/final",
  "sha256": "${MODEL_HASH}",
  "hardened_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
else
  # Update hardened_at timestamp
  python3 -c "
import json, sys, datetime
with open('${MERGE_INFO}') as f: d=json.load(f)
d['sha256']='${MODEL_HASH}'
d['hardened_at']=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
with open('${MERGE_INFO}','w') as f: json.dump(d,f,indent=2)
"
fi
echo "  merge_info.json → ${MERGE_INFO}"

# ---------------------------------------------------------------------------
# Step 3: Write systemd unit file
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Writing systemd service unit..."

UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
UNIT_CONTENT="[Unit]
Description=Fraud Detection LLM Server (vLLM / HF fallback)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${HOME}/venv/bin/python ${LLM_ROOT}/scripts/merge_and_serve.py \
    --merged-dir ${MERGED_DIR} \
    --port ${PORT} \
    --skip-merge
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=${PROJECT_ROOT}
# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
"

echo "  Unit file content preview:"
echo "${UNIT_CONTENT}" | head -20

if command -v systemctl &>/dev/null && [[ $EUID -eq 0 ]]; then
  echo "${UNIT_CONTENT}" > "${UNIT_PATH}"
  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}"
  echo "  systemd unit installed and enabled → ${UNIT_PATH}"
else
  # Write locally for reference (non-root or non-Linux)
  LOCAL_UNIT="${LLM_ROOT}/configs/${SERVICE_NAME}.service"
  echo "${UNIT_CONTENT}" > "${LOCAL_UNIT}"
  echo "  (non-root) Unit file saved locally → ${LOCAL_UNIT}"
  echo "  To install: sudo cp ${LOCAL_UNIT} /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable ${SERVICE_NAME}"
fi

# ---------------------------------------------------------------------------
# Step 4: CloudWatch metrics setup (AWS-specific)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Setting up CloudWatch metrics..."

CW_CONFIG_DIR="${HOME}/.aws"
mkdir -p "${CW_CONFIG_DIR}"

CW_CONFIG="${LLM_ROOT}/configs/cloudwatch_agent.json"
cat > "${CW_CONFIG}" <<'JSON'
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "ubuntu"
  },
  "metrics": {
    "append_dimensions": {
      "InstanceId": "${aws:InstanceId}",
      "InstanceType": "${aws:InstanceType}",
      "Service": "fraud-llm"
    },
    "metrics_collected": {
      "cpu": {
        "measurement": ["cpu_usage_idle", "cpu_usage_user", "cpu_usage_system"],
        "metrics_collection_interval": 30,
        "totalcpu": true
      },
      "mem": {
        "measurement": ["mem_used_percent"],
        "metrics_collection_interval": 30
      },
      "gpu_nvidia": {
        "measurement": [
          "utilization_gpu",
          "utilization_memory",
          "temperature_gpu",
          "memory_used"
        ],
        "metrics_collection_interval": 30
      },
      "statsd": {
        "service_address": ":8125",
        "metrics_collection_interval": 10,
        "metrics_aggregation_interval": 60
      }
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/journal/fraud-llm",
            "log_group_name": "/fraud-detection/llm",
            "log_stream_name": "{instance_id}",
            "timestamp_format": "%Y-%m-%dT%H:%M:%SZ"
          }
        ]
      }
    }
  }
}
JSON

echo "  CloudWatch agent config → ${CW_CONFIG}"

if command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
  amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c "file:${CW_CONFIG}"
  echo "  CloudWatch agent configured and started."
else
  echo "  amazon-cloudwatch-agent not found; config saved for manual deployment."
  echo "  Install: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/install-CloudWatch-Agent-on-EC2-Instance.html"
fi

# ---------------------------------------------------------------------------
# Step 5: Write hardening summary
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Writing hardening summary..."

SUMMARY="${MERGED_DIR}/hardening_summary.json"
python3 -c "
import json, datetime, os
summary = {
    'model_sha256': '${MODEL_HASH}',
    'service_name': '${SERVICE_NAME}',
    'port': ${PORT},
    'hardened_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'cloudwatch_config': '${CW_CONFIG}',
    'checksums_file': '${CHECKSUM_FILE}',
}
with open('${SUMMARY}', 'w') as f:
    json.dump(summary, f, indent=2)
print('  Hardening summary → ${SUMMARY}')
"

echo ""
echo "=== Hardening complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify checksums: sha256sum -c ${CHECKSUM_FILE}"
echo "  2. Start service: sudo systemctl start ${SERVICE_NAME}"
echo "  3. Check status:  sudo systemctl status ${SERVICE_NAME}"
echo "  4. Run eval:      python llm/scripts/eval.py --server-url http://localhost:${PORT}"
echo "  5. Run red-team:  python llm/scripts/red_team.py --server-url http://localhost:${PORT}"
