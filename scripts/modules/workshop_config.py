"""Workshop configuration loader and validator.

Reads ``workshop_config.json`` from the repo root (if present), merges over
defaults, and sets ``ROCKEM_SUITE_ROOT`` before ``rockem_bridge`` is imported.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CPU_FORWARD_TE2D = "mpiEmmodTE2d"
GPU_FORWARD_TE2D = "mpiEmmodTE2dGpu"
CPU_INVERSION_TE2D = "mpiEminvTE2d"
GPU_INVERSION_TE2D = "mpiEminvTE2dGpu"


def _default_rockem_root() -> Path:
    return Path.home() / "software" / "new_rockem" / "rockem-suite"


@dataclass
class WorkshopConfig:
    rockem_suite_root: Path
    mpirun: str
    nproc_default: int
    default_segy: Path
    workspace_dir: str = "workspace"
    use_gpu_forward_2d: bool = False
    use_gpu_inversion_2d: bool = False
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

    def forward_engine_te2d(self) -> str:
        return GPU_FORWARD_TE2D if self.use_gpu_forward_2d else CPU_FORWARD_TE2D

    def inversion_engine_te2d(self) -> str:
        return GPU_INVERSION_TE2D if self.use_gpu_inversion_2d else CPU_INVERSION_TE2D

    def binary_path(self, name: str) -> Path:
        return self.rockem_suite_root / "bin" / name

    def to_dict(self) -> dict[str, Any]:
        return {
            "rockem_suite_root": str(self.rockem_suite_root),
            "mpirun": self.mpirun,
            "nproc_default": self.nproc_default,
            "default_segy": str(self.default_segy),
            "workspace_dir": self.workspace_dir,
            "use_gpu_forward_2d": bool(self.use_gpu_forward_2d),
            "use_gpu_inversion_2d": bool(self.use_gpu_inversion_2d),
        }


def _defaults(root: Path | None = None) -> WorkshopConfig:
    root = (root or Path.cwd()).resolve()
    return WorkshopConfig(
        rockem_suite_root=_default_rockem_root(),
        mpirun="mpirun",
        nproc_default=4,
        default_segy=root / "examples" / "Fault_1.sgy",
        workspace_dir="workspace",
        use_gpu_forward_2d=False,
        use_gpu_inversion_2d=False,
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
        if "use_gpu_forward_2d" in data:
            cfg.use_gpu_forward_2d = bool(data["use_gpu_forward_2d"])
        if "use_gpu_inversion_2d" in data:
            cfg.use_gpu_inversion_2d = bool(data["use_gpu_inversion_2d"])
    os.environ["ROCKEM_SUITE_ROOT"] = str(cfg.rockem_suite_root)
    return cfg


def save_config(cfg: WorkshopConfig, root: Path | None = None) -> Path:
    root = (root or Path.cwd()).resolve()
    path = config_path(root)
    path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    return path


def _nvidia_smi_devices() -> tuple[bool, str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, "nvidia-smi not found on PATH"
    try:
        proc = subprocess.run(
            [exe, "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, detail or f"nvidia-smi exited {proc.returncode}"
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return False, "nvidia-smi returned no GPU devices"
    return True, "; ".join(lines)


def _binary_check(cfg: WorkshopConfig, name: str) -> tuple[bool, str]:
    path = cfg.binary_path(name)
    if not path.exists():
        return False, f"missing: {path}"
    if not os.access(path, os.X_OK):
        return False, f"not executable: {path}"
    return True, str(path)


def patch_runinv_template(template_text: str, cfg: WorkshopConfig, nproc: int) -> str:
    """Inject MPI launcher, process count, and selected inversion binary."""
    content = template_text.replace("mpirun ", f"{cfg.mpirun} -np {nproc} ", 1)
    inv_bin = cfg.inversion_engine_te2d()
    content = re.sub(r"mpiEminvTE2d(Gpu)?", inv_bin, content)
    return content


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

    cpu_fwd_ok, cpu_fwd_detail = _binary_check(cfg, CPU_FORWARD_TE2D)
    checks.append((f"CPU forward binary ({CPU_FORWARD_TE2D})", cpu_fwd_ok, cpu_fwd_detail))

    cpu_inv_ok, cpu_inv_detail = _binary_check(cfg, CPU_INVERSION_TE2D)
    checks.append((f"CPU inversion binary ({CPU_INVERSION_TE2D})", cpu_inv_ok, cpu_inv_detail))

    gpu_fwd_ok, gpu_fwd_detail = _binary_check(cfg, GPU_FORWARD_TE2D)
    gpu_inv_ok, gpu_inv_detail = _binary_check(cfg, GPU_INVERSION_TE2D)
    gpu_smi_ok, gpu_smi_detail = _nvidia_smi_devices()

    checks.append((
        "GPU forward binary (optional)",
        gpu_fwd_ok,
        gpu_fwd_detail if gpu_fwd_ok else f"{gpu_fwd_detail} — build with: make -f Makefile.gpu bin/{GPU_FORWARD_TE2D}",
    ))
    checks.append((
        "GPU inversion binary (optional)",
        gpu_inv_ok,
        gpu_inv_detail if gpu_inv_ok else f"{gpu_inv_detail} — build with: make -f Makefile.gpu bin/{GPU_INVERSION_TE2D}",
    ))
    checks.append((
        "NVIDIA driver / nvidia-smi",
        gpu_smi_ok,
        gpu_smi_detail if gpu_smi_ok else gpu_smi_detail,
    ))

    if cfg.use_gpu_forward_2d:
        checks.append((
            f"GPU forward enabled → {GPU_FORWARD_TE2D}",
            gpu_fwd_ok and gpu_smi_ok,
            "ready" if (gpu_fwd_ok and gpu_smi_ok) else "enable GPU forward only when binary and GPU are available",
        ))
    if cfg.use_gpu_inversion_2d:
        checks.append((
            f"GPU inversion enabled → {GPU_INVERSION_TE2D}",
            gpu_inv_ok and gpu_smi_ok,
            "ready" if (gpu_inv_ok and gpu_smi_ok) else "enable GPU inversion only when binary and GPU are available",
        ))

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
