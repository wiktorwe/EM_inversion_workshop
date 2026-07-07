#!/bin/sh
set -eu

# Points at the validated rockem-suite checkout - see scripts/modules/
# rockem_bridge.py for why this must NOT be the old ~/software/rockem-suite
# (stale, pre-CFL/source-scaling-fix checkout).
ROCKEM_SUITE_ROOT="${ROCKEM_SUITE_ROOT:-$HOME/software/new_rockem/rockem-suite}"

META_PATH="setup_metadata.json"
ENGINE="mpiEmmodTE2d"
CFG="mod.cfg"

if [ -f "$META_PATH" ]; then
  MODE_DATA=$(python - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path("setup_metadata.json").read_text())
engine = str(meta.get("forward_engine", "mpiEmmodTE2d"))
cfg = str(meta.get("forward_cfg", "mod.cfg"))
print(engine)
print(cfg)
PY
)
  ENGINE=$(printf "%s" "$MODE_DATA" | sed -n '1p')
  CFG=$(printf "%s" "$MODE_DATA" | sed -n '2p')
fi

mpirun "$ROCKEM_SUITE_ROOT/bin/$ENGINE" "$CFG"
