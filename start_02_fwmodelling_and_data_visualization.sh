#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/scripts/voila_common.sh"

NOTEBOOK="02_fwmodelling_and_data_visualization.ipynb"
PID_FILE="$SCRIPT_DIR/.voila_fwmodelling_server.pid"
echo "$$" > "$PID_FILE"
cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

trust_notebook "$NOTEBOOK"
exec voila "$NOTEBOOK" --strip_sources=True
