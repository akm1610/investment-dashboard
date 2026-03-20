#!/usr/bin/env bash
# =============================================================================
# start.sh – Investment Dashboard setup & launch script
#
# Usage:
#   chmod +x start.sh
#   ./start.sh              # installs deps and starts both services
#   ./start.sh --no-install # skip dependency installation (faster restart)
# =============================================================================

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DEPS=true

for arg in "$@"; do
  [[ "$arg" == "--no-install" ]] && INSTALL_DEPS=false
done

echo -e "\n${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Investment Dashboard – Deployment Script ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}\n"

# ── 1. Kill existing processes on ports 8000-9000 ───────────────────────────
info "Stopping any processes using ports 8000–9000…"
_kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    warn "  Stopping process(es) on port $port: $pids"
    kill "$pids" 2>/dev/null || true
    sleep 1
    # Only SIGKILL if still running
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      kill -9 "$pids" 2>/dev/null || true
    fi
  fi
}

for port in $(seq 8000 9000); do
  _kill_port "$port"
done

# Also kill any running React dev server on port 3000
_kill_port 3000

success "Port cleanup complete."

# ── 2. Prerequisites check ──────────────────────────────────────────────────
info "Checking prerequisites…"
command -v python3 &>/dev/null  || error "python3 is required but not found."
command -v pip3   &>/dev/null  || command -v pip &>/dev/null || error "pip is required but not found."
command -v node   &>/dev/null  || error "node is required (https://nodejs.org)."
command -v npm    &>/dev/null  || error "npm is required."
success "All prerequisites found."

# ── 3. Install Python dependencies ──────────────────────────────────────────
if [[ "$INSTALL_DEPS" == true ]]; then
  info "Installing Python dependencies…"
  cd "$SCRIPT_DIR"
  pip3 install -r requirements.txt --quiet || pip install -r requirements.txt --quiet
  success "Python dependencies installed."

  # ── 4. Install React dashboard dependencies ─────────────────────────────
  info "Installing React dashboard dependencies…"
  cd "$SCRIPT_DIR/dashboard"
  npm install --silent
  cd "$SCRIPT_DIR"
  success "React dependencies installed."
else
  info "Skipping dependency installation (--no-install flag set)."
fi

# ── 5. Start Flask API on port 9000 ─────────────────────────────────────────
info "Starting Flask API on port 9000…"
cd "$SCRIPT_DIR"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

API_PORT=9000 nohup python3 flask_api.py \
  > "${LOG_DIR}/flask_api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "${LOG_DIR}/flask_api.pid"
success "Flask API started (PID $API_PID). Logs: ${LOG_DIR}/flask_api.log"

# Wait for Flask to be ready (up to 15 s)
info "Waiting for Flask API to become ready…"
for i in $(seq 1 15); do
  if curl -sf http://localhost:9000/health &>/dev/null; then
    success "Flask API is ready at http://localhost:9000"
    break
  fi
  sleep 1
  if [[ $i -eq 15 ]]; then
    warn "Flask API did not respond in 15 s. Check ${LOG_DIR}/flask_api.log"
  fi
done

# ── 6. Start React dashboard on port 3000 ───────────────────────────────────
info "Starting React dashboard on port 3000…"
cd "$SCRIPT_DIR/dashboard"

nohup npm run dev \
  > "${LOG_DIR}/react_dashboard.log" 2>&1 &
REACT_PID=$!
echo "$REACT_PID" > "${LOG_DIR}/react_dashboard.pid"
cd "$SCRIPT_DIR"
success "React dashboard started (PID $REACT_PID). Logs: ${LOG_DIR}/react_dashboard.log"

# Wait for React dev server (up to 20 s)
info "Waiting for React dashboard to become ready…"
for i in $(seq 1 20); do
  if curl -sf http://localhost:3000 &>/dev/null; then
    success "React dashboard is ready at http://localhost:3000"
    break
  fi
  sleep 1
  if [[ $i -eq 20 ]]; then
    warn "React dashboard did not respond in 20 s. Check ${LOG_DIR}/react_dashboard.log"
  fi
done

# ── 7. Summary ───────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║             Services Running                  ║${RESET}"
echo -e "${BOLD}╠═══════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║${RESET}  Flask API  →  ${GREEN}http://localhost:9000${RESET}          ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Dashboard  →  ${GREEN}http://localhost:3000${RESET}          ${BOLD}║${RESET}"
echo -e "${BOLD}╠═══════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Verification commands:                       ║${RESET}"
echo -e "${BOLD}║${RESET}  curl http://localhost:9000/health            ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  curl http://localhost:9000/predict/AAPL      ${BOLD}║${RESET}"
echo -e "${BOLD}╠═══════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  To stop all services:                        ║${RESET}"
echo -e "${BOLD}║${RESET}  ./stop.sh                                    ${BOLD}║${RESET}"
echo -e "${BOLD}╚═══════════════════════════════════════════════╝${RESET}"
echo ""
