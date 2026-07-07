"""Workshop configuration loader and validator.

Reads ``workshop_config.json`` from the repo root (if present), merges over
defaults, and sets ``ROCKEM_SUITE_ROOT`` before ``rockem_bridge`` is imported.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_rockem_root() -> Path:
    return Path.home() / "software" / "new_rockem" / "rockem-suite"


@dataclass
class WorkshopConfig:
    rockem_suite_root: Path
    mpirun: str
    nproc_default: int
    default_segy: Path
    workspace_dir: str = "workspace"
    _root: Path = field(repr=False, default=Path())

    @property
    def workspace(self) -> Path:
        return self._root / self.workspace_dir

    @property
    def fwd_2d_dir(self) -> Path:
        return self.workspace / "2D" / "forward"

    @property
    def inv_2d_input_dir(self) -> Path:
        return self.workspace / "2D" / "inversion" / "input"

    @property
    def inv_2d_runs_dir(self) -> Path:
        return self.workspace / "2D" / "inversion"

    @property
    def results_2d_dir(self) -> Path:
        return self.workspace / "2D" / "results"

    @property
    def inv_1d_runs_dir(self) -> Path:
        return self.workspace / "1D" / "inversion"

    @property
    def results_1d_dir(self) -> Path:
        return self.workspace / "1D" / "results"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rockem_suite_root": str(self.rockem_suite_root),
            "mpirun": self.mpirun,
            "nproc_default": self.nproc_default,
            "default_segy": str(self.default_segy),
            "workspace_dir": self.workspace_dir,
        }


def _defaults(root: Path | None = None) -> WorkshopConfig:
    root = (root or Path.cwd()).resolve()
    return WorkshopConfig(
        rockem_suite_root=_default_rockem_root(),
        mpirun="mpirun",
        nproc_default=4,
        default_segy=root / "examples" / "Fault_1.sgy",
        workspace_dir="workspace",
        _root=root,
    )


def config_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()).resolve() / "workshop_config.json"


def load_config(root: Path | None = None) -> WorkshopConfig:
    """Load config and set ``ROCKEM_SUITE_ROOT`` in the environment."""
    root = (root or Path.cwd()).resolve()
    cfg = _defaults(root)
    path = config_path(root)
    if path.exists():
        data = json.loads(path.read_text())
        if "rockem_suite_root" in data:
            cfg.rockem_suite_root = Path(data["rockem_suite_root"]).expanduser().resolve()
        if "mpirun" in data:
            cfg.mpirun = str(data["mpirun"])
        if "nproc_default" in data:
            cfg.nproc_default = int(data["nproc_default"])
        if "default_segy" in data:
            cfg.default_segy = Path(data["default_segy"]).expanduser()
            if not cfg.default_segy.is_absolute():
                cfg.default_segy = (root / cfg.default_segy).resolve()
        if "workspace_dir" in data:
            cfg.workspace_dir = str(data["workspace_dir"])
    os.environ["ROCKEM_SUITE_ROOT"] = str(cfg.rockem_suite_root)
    return cfg


def save_config(cfg: WorkshopConfig, root: Path | None = None) -> Path:
    root = (root or Path.cwd()).resolve()
    path = config_path(root)
    path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    return path


def validate_config(cfg: WorkshopConfig) -> list[tuple[str, bool, str]]:
    """Return (check_name, passed, detail) tuples."""
    checks: list[tuple[str, bool, str]] = []
    root = cfg.rockem_suite_root
    python_dir = root / "python"
    checks.append((
        "rockem-suite python/",
        python_dir.is_dir(),
        str(python_dir),
    ))
    for binary in ("mpiEmmodTE2d", "mpiEminvTE2d"):
        p = root / "bin" / binary
        checks.append((f"binary {binary}", p.exists(), str(p)))
    mpirun = shutil.which(cfg.mpirun)
    checks.append((
        f"MPI launcher ({cfg.mpirun})",
        mpirun is not None,
        mpirun or "not found on PATH",
    ))
    if mpirun:
        try:
            proc = subprocess.run(
                [cfg.mpirun, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = proc.returncode == 0
            detail = (proc.stdout or proc.stderr or "").strip().splitlines()[0] if ok else proc.stderr.strip()
        except Exception as exc:
            ok = False
            detail = str(exc)
        checks.append(("mpirun --version", ok, detail))
    segy_ok = cfg.default_segy.exists()
    checks.append((
        "default SEG-Y",
        segy_ok,
        str(cfg.default_segy),
    ))
    greens = (
        root
        / "doc"
        / "examples"
        / "validate_layered_1d_model"
        / "shared"
        / "greens_layered_2d.py"
    )
    checks.append((
        "analytic greens solver",
        greens.is_file(),
        str(greens),
    ))
    return checks
