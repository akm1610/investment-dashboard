#!/usr/bin/env bash
# stop.sh – Stop all Investment Dashboard services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
info()    { echo -e "\033[0;36m[INFO]\033[0m  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }

echo -e "\nStopping Investment Dashboard services…\n"

for service in flask_api react_dashboard; do
  PID_FILE="${LOG_DIR}/${service}.pid"
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID" && success "Stopped $service (PID $PID)"
    else
      warn "$service (PID $PID) was already stopped."
    fi
    rm -f "$PID_FILE"
  else
    warn "No PID file for $service."
  fi
done

success "All services stopped."
