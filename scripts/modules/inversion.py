"""Inversion setup helpers for staging inputs and local runs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from third_party.rockseis.io.rsfile import rsfile


CFG_LINE_RE = re.compile(r'^(\s*([A-Za-z0-9_]+)\s*=\s*")([^"]*)(";\s*(?:#.*)?)$')


def read_cfg_values(cfg_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in Path(cfg_path).read_text().splitlines():
        match = CFG_LINE_RE.match(line)
        if match:
            values[match.group(2)] = match.group(3)
    return values


def update_cfg_values(cfg_path: Path, updates: Dict[str, str]) -> Dict[str, str]:
    cfg_path = Path(cfg_path)
    lines = cfg_path.read_text().splitlines()
    seen = set()
    out: List[str] = []
    for line in lines:
        match = CFG_LINE_RE.match(line)
        if not match:
            out.append(line)
            continue
        key = match.group(2)
        if key in updates:
            out.append(f'{match.group(1)}{updates[key]}{match.group(4)}')
            seen.add(key)
        else:
            out.append(line)
    missing = sorted(set(updates.keys()) - seen)
    if missing:
        raise KeyError(f"Keys not found in {cfg_path}: {missing}")
    cfg_path.write_text("\n".join(out) + "\n")
    return read_cfg_values(cfg_path)


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    _ensure_parent(dst)
    shutil.copyfile(src, dst)
    return dst


def prepare_data_from_fdmodel(fdmodel_dir: Path, output_dir: Path) -> Dict[str, Path]:
    fdmodel_dir = Path(fdmodel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "Wavelet": fdmodel_dir / "wav2d.rss",
        "Recordfile_HX": fdmodel_dir / "Data" / "Hxshot.rss",
        "Recordfile_HZ": fdmodel_dir / "Data" / "Hzshot.rss",
    }
    targets = {
        "Wavelet": output_dir / "wav2d.rss",
        "Recordfile_HX": output_dir / "Hx_data.rss",
        "Recordfile_HZ": output_dir / "Hz_data.rss",
    }
    for key, src in sources.items():
        copy_file(src, targets[key])
    return targets


def copy_permittivity_model(fdmodel_dir: Path, output_dir: Path) -> Path:
    fdmodel_dir = Path(fdmodel_dir)
    output_dir = Path(output_dir)
    target = output_dir / "ep.rss"
    return copy_file(fdmodel_dir / "ep.rss", target)


def create_initial_sg0_model(
    fdmodel_dir: Path,
    output_dir: Path,
    mode: str = "uniform_resistivity",
    uniform_conductivity: Optional[float] = None,
    uniform_resistivity: Optional[float] = None,
    min_conductivity: float = 1e-8,
) -> Path:
    fdmodel_dir = Path(fdmodel_dir)
    output_dir = Path(output_dir)
    target = output_dir / "sg0.rss"
    ref_sg = fdmodel_dir / "sg.rss"
    if not ref_sg.exists():
        raise FileNotFoundError(f"Missing reference conductivity model: {ref_sg}")

    ref = rsfile()
    ref.read(str(ref_sg))
    arr = np.asarray(ref.data, dtype=np.float64)

    if mode == "uniform_conductivity":
        if uniform_conductivity is None:
            raise ValueError("uniform_conductivity must be provided.")
        sigma = float(uniform_conductivity)
    elif mode == "uniform_resistivity":
        if uniform_resistivity is None:
            raise ValueError("uniform_resistivity must be provided.")
        rho = float(uniform_resistivity)
        if rho <= 0.0:
            raise ValueError("uniform_resistivity must be > 0.")
        sigma = 1.0 / rho
    else:
        raise ValueError(
            f"Unsupported initial model mode: {mode}. "
            "Allowed modes: uniform_conductivity, uniform_resistivity."
        )

    sigma = max(float(min_conductivity), sigma)
    ref.data = np.asfortranarray(np.full(arr.shape, sigma, dtype=arr.dtype))
    _ensure_parent(target)
    ref.write(str(target))
    return target


def create_weight_file_from_hx(hx_record_path: Path, weight_path: Path) -> Path:
    hx_record_path = Path(hx_record_path)
    weight_path = Path(weight_path)
    if not hx_record_path.exists():
        raise FileNotFoundError(f"Missing HX record file: {hx_record_path}")
    f = rsfile()
    f.read(str(hx_record_path))
    data = np.asarray(f.data, dtype=np.float64)
    nt, ntr = data.shape
    weight = np.tile(np.hanning(nt)[:, np.newaxis], (1, ntr))
    f.data = np.asfortranarray(weight.astype(data.dtype, copy=False))
    _ensure_parent(weight_path)
    f.write(str(weight_path))
    return weight_path


def write_inv_cfg(
    template_cfg: Path,
    output_cfg: Path,
    max_iterations: int,
    apertx: float,
    dtx: float,
    dtz: float,
    input_file_values: Dict[str, str],
) -> Dict[str, str]:
    template_cfg = Path(template_cfg)
    output_cfg = Path(output_cfg)
    copy_file(template_cfg, output_cfg)
    updates = {
        "max_iterations": str(int(max_iterations)),
        "apertx": f"{float(apertx):.6f}",
        "dtx": f"{float(dtx):.6f}",
        "dtz": f"{float(dtz):.6f}",
    }
    updates.update(input_file_values)
    return update_cfg_values(output_cfg, updates)


def prepare_inversion_inputs(
    fdmodel_dir: Path,
    template_cfg: Path,
    output_dir: Path,
    max_iterations: int,
    apertx: float,
    dtx: float,
    dtz: float,
    initial_model_mode: str = "uniform_resistivity",
    uniform_conductivity: Optional[float] = None,
    uniform_resistivity: Optional[float] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = prepare_data_from_fdmodel(fdmodel_dir=fdmodel_dir, output_dir=output_dir)
    ep_path = copy_permittivity_model(fdmodel_dir=fdmodel_dir, output_dir=output_dir)
    if initial_model_mode not in {"uniform_conductivity", "uniform_resistivity"}:
        raise ValueError(
            "Initial model must be uniform. "
            "Use 'uniform_conductivity' or 'uniform_resistivity'."
        )

    sg0_path = create_initial_sg0_model(
        fdmodel_dir=fdmodel_dir,
        output_dir=output_dir,
        mode=initial_model_mode,
        uniform_conductivity=uniform_conductivity,
        uniform_resistivity=uniform_resistivity,
    )
    weight_path = create_weight_file_from_hx(
        hx_record_path=data_paths["Recordfile_HX"],
        weight_path=output_dir / "weight.rss",
    )
    cfg_path = output_dir / "inv.cfg"
    write_inv_cfg(
        template_cfg=template_cfg,
        output_cfg=cfg_path,
        max_iterations=max_iterations,
        apertx=apertx,
        dtx=dtx,
        dtz=dtz,
        input_file_values={
            "Sg": "sg0.rss",
            "Ep": "ep.rss",
            "Wavelet": "wav2d.rss",
            "Recordfile_HX": "Hx_data.rss",
            "Recordfile_HZ": "Hz_data.rss",
            "Dataweightfile": "weight.rss",
        },
    )
    return {
        "inv_cfg": cfg_path,
        "sg0": sg0_path,
        "ep": ep_path,
        "wavelet": data_paths["Wavelet"],
        "hx": data_paths["Recordfile_HX"],
        "hz": data_paths["Recordfile_HZ"],
        "weight": weight_path,
    }


def stage_run_directory(
    input_dir: Path,
    run_dir: Path,
    clean: bool = True,
    include_patterns: Optional[Iterable[str]] = None,
) -> List[Path]:
    input_dir = Path(input_dir)
    run_dir = Path(run_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    staged: List[Path] = []
    patterns = list(include_patterns) if include_patterns else ["*"]
    picked = set()
    for pattern in patterns:
        for path in input_dir.glob(pattern):
            if path.is_file():
                picked.add(path)
    for src in sorted(picked):
        dst = run_dir / src.name
        copy_file(src, dst)
        staged.append(dst)
    return staged


__all__ = [
    "create_initial_sg0_model",
    "create_weight_file_from_hx",
    "prepare_data_from_fdmodel",
    "prepare_inversion_inputs",
    "read_cfg_values",
    "stage_run_directory",
    "update_cfg_values",
    "write_inv_cfg",
]
