# Shared Voila settings for workshop start_*.sh launchers.
# Source this file after SCRIPT_DIR is set and cd to repo root.

export JUPYTER_CONFIG_DIR="${SCRIPT_DIR}/jupyter_config"

trust_notebook() {
  if command -v jupyter >/dev/null 2>&1; then
    jupyter trust "$1" >/dev/null 2>&1 || true
  fi
}
