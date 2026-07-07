#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/scripts/voila_common.sh"

NOTEBOOK="00_configure_workshop.ipynb"
PID_FILE="$SCRIPT_DIR/.voila_configure_server.pid"
echo "$$" > "$PID_FILE"
cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

trust_notebook "$NOTEBOOK"
exec voila "$NOTEBOOK" --strip_sources=True
