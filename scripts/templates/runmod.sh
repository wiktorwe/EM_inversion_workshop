#!/bin/sh
set -eu

META_PATH="setup_metadata.json"
ENGINE="mpiEmmodADITE2d"
CFG="mod.cfg"

if [ -f "$META_PATH" ]; then
  MODE_DATA=$(python - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path("setup_metadata.json").read_text())
engine = str(meta.get("forward_engine", "mpiEmmodADITE2d"))
cfg = str(meta.get("forward_cfg", "mod.cfg"))
print(engine)
print(cfg)
PY
)
  ENGINE=$(printf "%s" "$MODE_DATA" | sed -n '1p')
  CFG=$(printf "%s" "$MODE_DATA" | sed -n '2p')
fi

mpirun "$HOME/software/rockem-suite/bin/$ENGINE" "$CFG"
