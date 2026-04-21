#!/usr/bin/env bash
# =============================================================================
# Fraud Detection — Integration Environment Setup
# =============================================================================
# Installs Docker + Docker Compose (if missing), then brings the full stack up.
# Run as: bash integration/setup.sh
# Tested on: Ubuntu 22.04 / 24.04 LTS
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Resolve the integration/ directory regardless of where the script is called
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " Fraud Detection — Integration Stack Setup"
echo "============================================"
echo "  Working directory: $SCRIPT_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Docker Engine
# ---------------------------------------------------------------------------
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
    info "Docker already installed: $DOCKER_VER — skipping."
else
    info "[1/5] Installing Docker Engine..."

    # Detect distro
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        DISTRO=$ID
    else
        error "Cannot detect Linux distribution. Install Docker manually: https://docs.docker.com/engine/install/"
        exit 1
    fi

    case "$DISTRO" in
        ubuntu|debian)
            sudo apt-get update -qq
            sudo apt-get install -y --no-install-recommends \
                ca-certificates curl gnupg lsb-release

            # Add Docker's official GPG key
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/${DISTRO}/gpg \
                | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg

            # Add Docker apt repo
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/${DISTRO} $(lsb_release -cs) stable" \
              | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

            sudo apt-get update -qq
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
            ;;
        amzn)
            # Amazon Linux 2 / 2023
            sudo yum update -y -q
            sudo yum install -y docker
            sudo systemctl start docker
            ;;
        *)
            error "Unsupported distro: $DISTRO. Install Docker manually: https://docs.docker.com/engine/install/"
            exit 1
            ;;
    esac

    # Add current user to the docker group so sudo is not needed later
    sudo usermod -aG docker "$USER" || true
    info "Docker installed. NOTE: log out and back in (or run 'newgrp docker') for group change to take effect."
fi

# Make sure the Docker daemon is running
if ! sudo systemctl is-active --quiet docker 2>/dev/null; then
    info "Starting Docker daemon..."
    sudo systemctl enable --now docker
fi

# ---------------------------------------------------------------------------
# 2. Docker Compose v2 plugin (already bundled with Docker Desktop on macOS/Windows)
# ---------------------------------------------------------------------------
if docker compose version &>/dev/null 2>&1; then
    COMPOSE_VER=$(docker compose version --short 2>/dev/null || docker compose version | awk '{print $NF}')
    info "Docker Compose plugin already present: $COMPOSE_VER — skipping."
elif command -v docker-compose &>/dev/null; then
    warn "Only the legacy docker-compose v1 CLI was found. Attempting to install the v2 plugin..."
    COMPOSE_VER_TAG=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    COMPOSE_ARCH=$(uname -m)
    sudo curl -fsSL \
        "https://github.com/docker/compose/releases/download/${COMPOSE_VER_TAG}/docker-compose-linux-${COMPOSE_ARCH}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    info "Docker Compose v2 plugin installed: $COMPOSE_VER_TAG"
else
    info "[2/5] Installing Docker Compose v2 plugin..."
    COMPOSE_VER_TAG=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    COMPOSE_ARCH=$(uname -m)
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -fsSL \
        "https://github.com/docker/compose/releases/download/${COMPOSE_VER_TAG}/docker-compose-linux-${COMPOSE_ARCH}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    info "Docker Compose v2 plugin installed: $COMPOSE_VER_TAG"
fi

# ---------------------------------------------------------------------------
# 3. .env file
# ---------------------------------------------------------------------------
info "[3/5] Checking .env..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn ".env created from .env.example — review and edit before production use."
    else
        # Write a minimal dev .env so the stack can start out of the box
        cat > .env <<'ENVEOF'
# ---- Auto-generated dev defaults — DO NOT commit to source control ----
API_PORT=8000
POSTGRES_PORT=5432
REDIS_PORT=6379
POSTGRES_USER=fraud
POSTGRES_PASSWORD=fraud
POSTGRES_DB=fraud_detection
DATABASE_URL=postgresql+asyncpg://fraud:fraud@postgres:5432/fraud_detection
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
APP_ENV=development
JWT_SECRET_KEY=dev-jwt-secret-change-me
API_KEYS=dev-key-change-me
RATE_LIMIT_PER_MINUTE=200
HEALTH_CHECK_INTERVAL=5s
HEALTH_CHECK_RETRIES=5
ENVEOF
        warn ".env.example not found — a minimal dev .env was generated."
    fi
else
    info ".env already exists — keeping existing values."
fi

# ---------------------------------------------------------------------------
# 4. Build images
# ---------------------------------------------------------------------------
info "[4/5] Building Docker images..."
docker compose build --pull

# ---------------------------------------------------------------------------
# 5. Bring the stack up
# ---------------------------------------------------------------------------
info "[5/5] Starting the stack (detached)..."
docker compose up -d

echo ""
info "Stack is up. Services:"
docker compose ps
echo ""
info "API:       http://localhost:${API_PORT:-8000}"
info "Docs:      http://localhost:${API_PORT:-8000}/docs"
info "Test UI:   open integration/dev_test.html in your browser"
info "Logs:      docker compose logs -f api"
info "Teardown:  docker compose down"
