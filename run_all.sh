#!/usr/bin/env bash
# =============================================================================
# run_all.sh – Investment Dashboard unified launcher
#
# Starts the Flask prediction API and the Streamlit dashboard from a single
# terminal.  Handles virtual-environment setup, port conflict detection, log
# collection, graceful shutdown on Ctrl+C, and prints a live status summary.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh                # full setup + start all services
#   ./run_all.sh --no-install   # skip pip install (faster restart)
#   ./run_all.sh --streamlit-only  # only start the Streamlit dashboard
#   ./run_all.sh --api-only     # only start the Flask API
#   ./run_all.sh stop           # stop all managed services and exit
#   ./run_all.sh restart        # stop + start all services
#   ./run_all.sh status         # check which services are running
#   ./run_all.sh logs           # tail live logs from all services
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}   $*"; }
success() { echo -e "${GREEN}[OK]${RESET}     $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}   $*"; }
error()   { echo -e "${RED}[ERROR]${RESET}  $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}==> $*${RESET}"; }

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${LOG_DIR}"

# ── Port configuration ─────────────────────────────────────────────────────
API_PORT="${API_PORT:-9000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

# ── Timeouts (seconds) ─────────────────────────────────────────────────────
API_READY_TIMEOUT=20
STREAMLIT_READY_TIMEOUT=25

# ── Feature flags (can be overridden by command-line flags) ────────────────
INSTALL_DEPS=true
START_API=true
START_STREAMLIT=true
FORCE_KILL_PORTS=false  # set to true via --force to skip interactive port prompts

# ── Parse subcommands and flags ────────────────────────────────────────────
SUBCOMMAND=""
for arg in "$@"; do
  case "$arg" in
    stop|restart|status|logs) SUBCOMMAND="$arg" ;;
    --no-install)     INSTALL_DEPS=false ;;
    --streamlit-only) START_API=false ;;
    --api-only)       START_STREAMLIT=false ;;
    --force)          FORCE_KILL_PORTS=true ;;
  esac
done

# ── Utility: check if a port is in use ────────────────────────────────────
_port_pids() {
  lsof -ti tcp:"$1" 2>/dev/null || true
}

_port_in_use() {
  [[ -n "$(_port_pids "$1")" ]]
}

# ── Utility: gracefully (then forcibly) free a port ───────────────────────
_kill_port() {
  local port=$1
  local pids
  pids=$(_port_pids "$port")
  if [[ -n "$pids" ]]; then
    warn "  Port $port occupied by PID(s) $pids – stopping…"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids=$(_port_pids "$port")
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

# ── Utility: stop a tracked service ───────────────────────────────────────
_stop_service() {
  local name=$1
  local pid_file="${PID_DIR}/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      success "  Stopped $name (PID $pid)"
    else
      warn "  $name (PID $pid) was not running."
    fi
    rm -f "$pid_file"
  else
    warn "  No PID file found for $name."
  fi
}

# ── Subcommand: stop ──────────────────────────────────────────────────────
_do_stop() {
  echo -e "\n${BOLD}Stopping Investment Dashboard services…${RESET}\n"
  _stop_service flask_api
  _stop_service streamlit
  echo ""
  success "All managed services stopped."
}

# ── Subcommand: status ────────────────────────────────────────────────────
_do_status() {
  echo ""
  echo -e "${BOLD}╔═══════════════════════════════════════════════════╗${RESET}"
  echo -e "${BOLD}║          Investment Dashboard – Service Status    ║${RESET}"
  echo -e "${BOLD}╠═══════════════════════════════════════════════════╣${RESET}"

  _svc_status() {
    local label=$1 port=$2 name=$3
    local pid_file="${PID_DIR}/${name}.pid"
    local status url
    url="http://localhost:${port}"

    if _port_in_use "$port"; then
      if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        status="${GREEN}● RUNNING${RESET} (PID $pid)"
      else
        status="${GREEN}● RUNNING${RESET} (external)"
      fi
    else
      status="${RED}○ STOPPED${RESET}"
    fi
    # shellcheck disable=SC2059
    printf "  %-12s →  ${url}  [%b]\n" "$label" "$status"
  }

  echo -e "${BOLD}║${RESET}"
  _svc_status "Flask API"  "$API_PORT"       "flask_api"
  _svc_status "Streamlit"  "$STREAMLIT_PORT" "streamlit"
  echo -e "${BOLD}║${RESET}"
  echo -e "${BOLD}╚═══════════════════════════════════════════════════╝${RESET}"
  echo ""
}

# ── Subcommand: logs ──────────────────────────────────────────────────────
_do_logs() {
  local files=()
  [[ -f "${LOG_DIR}/flask_api.log" ]]  && files+=("${LOG_DIR}/flask_api.log")
  [[ -f "${LOG_DIR}/streamlit.log" ]]  && files+=("${LOG_DIR}/streamlit.log")
  if [[ ${#files[@]} -eq 0 ]]; then
    warn "No log files found in ${LOG_DIR}/"
    exit 0
  fi
  info "Tailing logs (Ctrl+C to exit):"
  tail -f "${files[@]}"
}

# ── Handle subcommands ────────────────────────────────────────────────────
case "$SUBCOMMAND" in
  stop)
    _do_stop
    exit 0
    ;;
  restart)
    _do_stop
    echo ""
    info "Restarting services…"
    # Re-exec without the 'restart' argument so we go through normal startup
    args=()
    for arg in "$@"; do
      [[ "$arg" != "restart" ]] && args+=("$arg")
    done
    exec "$0" "${args[@]}" --no-install
    ;;
  status)
    _do_status
    exit 0
    ;;
  logs)
    _do_logs
    exit 0
    ;;
esac

# =============================================================================
# NORMAL START PATH
# =============================================================================

echo -e "\n${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       Investment Dashboard – Unified Launcher        ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Flask API   →  port ${API_PORT}                          ║${RESET}"
echo -e "${BOLD}║  Streamlit   →  port ${STREAMLIT_PORT}                         ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}\n"

mkdir -p "$LOG_DIR"

# ── Trap Ctrl+C for graceful shutdown ─────────────────────────────────────
PIDS_TO_KILL=()

_shutdown() {
  echo ""
  warn "Caught signal – shutting down all services…"
  for pid in "${PIDS_TO_KILL[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  # Clean up PID files
  rm -f "${PID_DIR}/flask_api.pid" "${PID_DIR}/streamlit.pid"
  echo ""
  success "All services stopped. Goodbye!"
  exit 0
}

trap '_shutdown' INT TERM

# ── 1. Prerequisites ──────────────────────────────────────────────────────
header "Checking prerequisites"
command -v python3 &>/dev/null || error "python3 is required but not found."
PYTHON="${PYTHON:-python3}"
success "python3 found: $(python3 --version)"

# ── 2. Virtual environment ────────────────────────────────────────────────
header "Virtual environment"
VENV_DIR="${SCRIPT_DIR}/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at ${VENV_DIR}…"
  python3 -m venv "$VENV_DIR"
  success "Virtual environment created."
else
  success "Virtual environment already exists at ${VENV_DIR}."
fi

# Activate the venv for all subsequent commands
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
PYTHON="python"   # now refers to the venv python

# ── 3. Install Python dependencies ───────────────────────────────────────
if [[ "$INSTALL_DEPS" == true ]]; then
  header "Installing Python dependencies"
  pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
  success "Python dependencies installed."
else
  info "Skipping dependency installation (--no-install flag)."
fi

# ── 4. Port management ────────────────────────────────────────────────────
header "Port management"

_ensure_port_free() {
  local port=$1 label=$2
  if _port_in_use "$port"; then
    local pids
    pids=$(_port_pids "$port")
    warn "  ${label} port ${port} is occupied by PID(s): ${pids}"
    if [[ "$FORCE_KILL_PORTS" == true ]]; then
      info "  --force flag set: killing process(es) on port ${port}…"
      _kill_port "$port"
      success "  Cleared port $port."
    else
      read -r -p "  Kill the process(es) and continue? [y/N] " confirm
      if [[ "$confirm" =~ ^[Yy]$ ]]; then
        _kill_port "$port"
        success "  Cleared port $port."
      else
        error "Port $port still in use. Cannot start ${label}. Free it manually or use --force to kill automatically."
      fi
    fi
  else
    success "  Port ${port} is free (${label})."
  fi
}

[[ "$START_API" == true ]]       && _ensure_port_free "$API_PORT"       "Flask API"
[[ "$START_STREAMLIT" == true ]] && _ensure_port_free "$STREAMLIT_PORT" "Streamlit"

# ── 5. Start Flask API ────────────────────────────────────────────────────
if [[ "$START_API" == true ]]; then
  header "Starting Flask API"
  cd "$SCRIPT_DIR"
  API_PORT=$API_PORT nohup "$PYTHON" flask_api.py \
    > "${LOG_DIR}/flask_api.log" 2>&1 &
  API_PID=$!
  echo "$API_PID" > "${PID_DIR}/flask_api.pid"
  PIDS_TO_KILL+=("$API_PID")
  success "Flask API started (PID ${API_PID}). Logs: ${LOG_DIR}/flask_api.log"

  # Wait for Flask to be ready
  info "Waiting for Flask API to become ready…"
  FLASK_READY=false
  for i in $(seq 1 "$API_READY_TIMEOUT"); do
    if curl -sf "http://localhost:${API_PORT}/health" &>/dev/null; then
      success "Flask API is ready at http://localhost:${API_PORT}"
      FLASK_READY=true
      break
    fi
    sleep 1
    if [[ $i -eq "$API_READY_TIMEOUT" ]]; then
      warn "Flask API did not respond in ${API_READY_TIMEOUT}s. Check ${LOG_DIR}/flask_api.log"
    fi
  done
fi

# ── 6. Start Streamlit dashboard ──────────────────────────────────────────
if [[ "$START_STREAMLIT" == true ]]; then
  header "Starting Streamlit dashboard"
  cd "$SCRIPT_DIR"

  # Determine the Streamlit entry point (prefer src/app.py; fall back to app.py)
  if [[ -f "${SCRIPT_DIR}/src/app.py" ]]; then
    STREAMLIT_APP="src/app.py"
  elif [[ -f "${SCRIPT_DIR}/app.py" ]]; then
    STREAMLIT_APP="app.py"
  else
    error "Could not find a Streamlit app (src/app.py or app.py). Aborting."
  fi

  STREAMLIT_SERVER_PORT=$STREAMLIT_PORT \
  nohup "$PYTHON" -m streamlit run "$STREAMLIT_APP" \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    --browser.serverAddress localhost \
    > "${LOG_DIR}/streamlit.log" 2>&1 &
  STREAMLIT_PID=$!
  echo "$STREAMLIT_PID" > "${PID_DIR}/streamlit.pid"
  PIDS_TO_KILL+=("$STREAMLIT_PID")
  success "Streamlit started (PID ${STREAMLIT_PID}). Logs: ${LOG_DIR}/streamlit.log"

  # Wait for Streamlit to be ready
  info "Waiting for Streamlit dashboard to become ready…"
  STREAMLIT_READY=false
  for i in $(seq 1 "$STREAMLIT_READY_TIMEOUT"); do
    if curl -sf "http://localhost:${STREAMLIT_PORT}" &>/dev/null; then
      success "Streamlit is ready at http://localhost:${STREAMLIT_PORT}"
      STREAMLIT_READY=true
      break
    fi
    sleep 1
    if [[ $i -eq "$STREAMLIT_READY_TIMEOUT" ]]; then
      warn "Streamlit did not respond in ${STREAMLIT_READY_TIMEOUT}s. Check ${LOG_DIR}/streamlit.log"
    fi
  done

  # Auto-open browser (best-effort, non-fatal)
  _open_url() {
    local url=$1
    if command -v open &>/dev/null; then      # macOS
      open "$url" &>/dev/null &
    elif command -v xdg-open &>/dev/null; then # Linux/freedesktop
      xdg-open "$url" &>/dev/null &
    fi
  }
  [[ "${STREAMLIT_READY:-false}" == true ]] && _open_url "http://localhost:${STREAMLIT_PORT}"
fi

# ── 7. Status summary ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║               Services Now Running                  ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"

if [[ "$START_API" == true ]]; then
  echo -e "${BOLD}║${RESET}  Flask API    →  ${GREEN}http://localhost:${API_PORT}${RESET}            ${BOLD}║${RESET}"
  echo -e "${BOLD}║${RESET}    Health      →  ${GREEN}http://localhost:${API_PORT}/health${RESET}     ${BOLD}║${RESET}"
  echo -e "${BOLD}║${RESET}    Predict     →  ${GREEN}http://localhost:${API_PORT}/predict/AAPL${RESET}${BOLD}║${RESET}"
fi

if [[ "$START_STREAMLIT" == true ]]; then
  echo -e "${BOLD}║${RESET}  Streamlit UI  →  ${GREEN}http://localhost:${STREAMLIT_PORT}${RESET}           ${BOLD}║${RESET}"
fi

echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Logs:                                               ║${RESET}"
[[ "$START_API" == true ]]       && echo -e "${BOLD}║${RESET}    ${LOG_DIR}/flask_api.log          ${BOLD}║${RESET}"
[[ "$START_STREAMLIT" == true ]] && echo -e "${BOLD}║${RESET}    ${LOG_DIR}/streamlit.log          ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Commands:                                           ║${RESET}"
echo -e "${BOLD}║${RESET}    ./run_all.sh status   – check service status      ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}    ./run_all.sh logs     – tail all logs              ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}    ./run_all.sh stop     – stop all services          ${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}    ./run_all.sh restart  – restart all services       ${BOLD}║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║  Press Ctrl+C to stop all services                   ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── 8. Keep script alive and wait for child processes ─────────────────────
# Spin until all tracked children exit or the user presses Ctrl+C.
info "Services are running. Press Ctrl+C to stop everything."
while true; do
  ALL_DONE=true
  for pid in "${PIDS_TO_KILL[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      ALL_DONE=false
      break
    fi
  done
  [[ "$ALL_DONE" == true ]] && break
  sleep 5
done

# If we exit naturally (all children died), clean up
warn "All services have stopped."
rm -f "${PID_DIR}/flask_api.pid" "${PID_DIR}/streamlit.pid"
