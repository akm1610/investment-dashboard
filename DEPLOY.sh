#!/usr/bin/env bash
# =============================================================================
# DEPLOY.sh – Investment Dashboard one-command deployment script
#
# Usage:
#   chmod +x DEPLOY.sh
#   ./DEPLOY.sh              # full install + start both services
#   ./DEPLOY.sh --no-install # skip dependency installation (faster restart)
#   ./DEPLOY.sh --check-only # only verify ports are clear, then exit
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
CHECK_ONLY=false

for arg in "$@"; do
  [[ "$arg" == "--no-install" ]] && INSTALL_DEPS=false
  [[ "$arg" == "--check-only" ]] && CHECK_ONLY=true
done

# ── Port configuration ───────────────────────────────────────────────────────
API_PORT=9000
DASHBOARD_PORT=3000
LEGACY_API_PORT=8000      # old default port – cleared for safety
API_READY_TIMEOUT=20      # seconds to wait for Flask API
DASHBOARD_READY_TIMEOUT=25 # seconds to wait for React dashboard

echo -e "\n${BOLD}╔═══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║    Investment Dashboard – Production Deployment    ║${RESET}"
echo -e "${BOLD}╠═══════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Flask API   →  port ${API_PORT}                        ║${RESET}"
echo -e "${BOLD}║  React UI    →  port ${DASHBOARD_PORT}                        ║${RESET}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════╝${RESET}\n"

# ── 1. Check / clear ports ───────────────────────────────────────────────────
info "Checking ports ${API_PORT} and ${DASHBOARD_PORT}…"

_check_port_clear() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo -e "  ${YELLOW}Port $port in use (PID $pids)${RESET}"
    return 1
  fi
  return 0
}

_kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    warn "  Stopping process(es) on port $port: $pids"
    kill "$pids" 2>/dev/null || true
    sleep 1
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      kill -9 "$pids" 2>/dev/null || true
    fi
  fi
}

# Report current port status
API_CLEAR=true
DASH_CLEAR=true
_check_port_clear "$API_PORT"       || API_CLEAR=false
_check_port_clear "$DASHBOARD_PORT" || DASH_CLEAR=false

if [[ "$API_CLEAR" == true && "$DASH_CLEAR" == true ]]; then
  success "Ports ${API_PORT} and ${DASHBOARD_PORT} are clear."
else
  if [[ "$CHECK_ONLY" == true ]]; then
    error "One or more required ports are already in use. Run './stop.sh' or './DEPLOY.sh' (without --check-only) to clear them automatically."
  fi
  info "Clearing occupied ports…"
  _kill_port "$API_PORT"
  _kill_port "$DASHBOARD_PORT"
  # Also clear the legacy port used by older versions of the app
  _kill_port "$LEGACY_API_PORT"
  success "Port cleanup complete."
fi

[[ "$CHECK_ONLY" == true ]] && { success "Port check complete – nothing to start."; exit 0; }

# ── 2. Prerequisites ─────────────────────────────────────────────────────────
info "Checking prerequisites…"
command -v python3 &>/dev/null || error "python3 is required but not found."
command -v pip3    &>/dev/null || command -v pip &>/dev/null || error "pip is required but not found."
command -v node    &>/dev/null || error "node is required (https://nodejs.org)."
command -v npm     &>/dev/null || error "npm is required."
success "All prerequisites found."

# ── 3. Install dependencies ──────────────────────────────────────────────────
if [[ "$INSTALL_DEPS" == true ]]; then
  info "Installing Python dependencies…"
  cd "$SCRIPT_DIR"
  pip3 install -r requirements.txt --quiet || pip install -r requirements.txt --quiet
  success "Python dependencies installed."

  info "Installing React dashboard dependencies…"
  cd "$SCRIPT_DIR/dashboard"
  npm install --silent
  cd "$SCRIPT_DIR"
  success "React dependencies installed."
else
  info "Skipping dependency installation (--no-install flag)."
fi

# ── 4. Run test suite ────────────────────────────────────────────────────────
info "Running test suite to verify system integrity…"
cd "$SCRIPT_DIR"
if python3 -m pytest tests/ -q --tb=short 2>&1; then
  success "All tests passed."
else
  error "Test suite failed – deployment aborted. Fix failing tests before deploying."
fi

# ── 5. Start Flask API on port 9000 ─────────────────────────────────────────
info "Starting Flask API on port ${API_PORT}…"
cd "$SCRIPT_DIR"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

API_PORT=${API_PORT} nohup python3 flask_api.py \
  > "${LOG_DIR}/flask_api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "${LOG_DIR}/flask_api.pid"
success "Flask API started (PID ${API_PID}). Logs: ${LOG_DIR}/flask_api.log"

# Wait for Flask to be ready (up to 20 s)
info "Waiting for Flask API to become ready…"
for i in $(seq 1 "$API_READY_TIMEOUT"); do
  if curl -sf "http://localhost:${API_PORT}/health" &>/dev/null; then
    success "Flask API is ready at http://localhost:${API_PORT}"
    break
  fi
  sleep 1
  if [[ $i -eq "$API_READY_TIMEOUT" ]]; then
    warn "Flask API did not respond in ${API_READY_TIMEOUT} s. Check ${LOG_DIR}/flask_api.log"
  fi
done

# ── 6. Start React dashboard on port 3000 ───────────────────────────────────
info "Starting React dashboard on port ${DASHBOARD_PORT}…"
cd "$SCRIPT_DIR/dashboard"

nohup npm run dev \
  > "${LOG_DIR}/react_dashboard.log" 2>&1 &
REACT_PID=$!
echo "$REACT_PID" > "${LOG_DIR}/react_dashboard.pid"
cd "$SCRIPT_DIR"
success "React dashboard started (PID ${REACT_PID}). Logs: ${LOG_DIR}/react_dashboard.log"

# Wait for React dev server (up to 25 s)
info "Waiting for React dashboard to become ready…"
for i in $(seq 1 "$DASHBOARD_READY_TIMEOUT"); do
  if curl -sf "http://localhost:${DASHBOARD_PORT}" &>/dev/null; then
    success "React dashboard is ready at http://localhost:${DASHBOARD_PORT}"
    break
  fi
  sleep 1
  if [[ $i -eq "$DASHBOARD_READY_TIMEOUT" ]]; then
    warn "React dashboard did not respond in ${DASHBOARD_READY_TIMEOUT} s. Check ${LOG_DIR}/react_dashboard.log"
  fi
done

# ── 7. Summary ───────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║                  Services Running                   ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║${RESET}  Flask API   →  ${GREEN}http://localhost:${API_PORT}${RESET}           ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  Dashboard   →  ${GREEN}http://localhost:${DASHBOARD_PORT}${RESET}           ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Quick verification:                                 ║${RESET}"
echo -e "${BOLD}║${RESET}  curl http://localhost:${API_PORT}/health               ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  curl http://localhost:${API_PORT}/predict/AAPL         ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Test suite:                                         ║${RESET}"
echo -e "${BOLD}║${RESET}  python3 -m pytest tests/ -v                         ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  To stop all services:  ./stop.sh                    ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
