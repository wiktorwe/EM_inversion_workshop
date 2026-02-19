"""Survey helpers for workshop workflows."""

import re
from pathlib import Path

import numpy as np

from third_party.rockseis.io.rsfile import rsfile


CFG_LINE_RE = re.compile(r'^(\s*([A-Za-z0-9_]+)\s*=\s*")([^"]*)(";\s*(?:#.*)?)$')


def read_cfg_values(cfg_path):
    """Read key/value pairs from survey.cfg style files."""
    values = {}
    for line in Path(cfg_path).read_text().splitlines():
        match = CFG_LINE_RE.match(line)
        if match:
            values[match.group(2)] = match.group(3)
    return values


def update_cfg_values(cfg_path, updates):
    """Update selected keys in a config file while preserving comments."""
    cfg_path = Path(cfg_path)
    lines = cfg_path.read_text().splitlines()
    seen = set()
    new_lines = []
    for line in lines:
        match = CFG_LINE_RE.match(line)
        if not match:
            new_lines.append(line)
            continue
        key = match.group(2)
        if key in updates:
            new_value = str(updates[key])
            new_lines.append(f'{match.group(1)}{new_value}{match.group(4)}')
            seen.add(key)
        else:
            new_lines.append(line)
    missing = sorted(set(updates.keys()) - seen)
    if missing:
        raise KeyError(f"Keys not found in {cfg_path}: {missing}")
    cfg_path.write_text("\n".join(new_lines) + "\n")


def _as_float(values, key):
    return float(values[key])


def _as_int(values, key):
    return int(float(values[key]))


def _as_bool(values, key):
    text = str(values[key]).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {key}: {values[key]}")


def _compute_trace_geometry(cfg_values):
    dim = _as_int(cfg_values, "dim")
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dim={dim}. Expected 2 or 3.")
    obc = _as_bool(cfg_values, "OBC")

    nsx = _as_int(cfg_values, "nsx")
    nsy = _as_int(cfg_values, "nsy")
    ngx = _as_int(cfg_values, "ngx")
    ngy = _as_int(cfg_values, "ngy")

    sx0 = _as_float(cfg_values, "sx0")
    sy0 = _as_float(cfg_values, "sy0")
    sz0 = _as_float(cfg_values, "sz0")
    dsx = _as_float(cfg_values, "dsx")
    dsy = _as_float(cfg_values, "dsy")
    ds1x = _as_float(cfg_values, "ds1x")
    ds1y = _as_float(cfg_values, "ds1y")
    ds2x = _as_float(cfg_values, "ds2x")
    ds2y = _as_float(cfg_values, "ds2y")

    gx0 = _as_float(cfg_values, "gx0")
    gy0 = _as_float(cfg_values, "gy0")
    gz0 = _as_float(cfg_values, "gz0")
    dgx = _as_float(cfg_values, "dgx")
    dgy = _as_float(cfg_values, "dgy")

    src_x, src_y, src_z = [], [], []
    rec_x, rec_y, rec_z = [], [], []

    fldr = 0
    for isy in range(nsy):
        for isx in range(nsx):
            fldr += 1
            dx, dy = (ds1x, ds1y) if (fldr % 2 == 0) else (ds2x, ds2y)
            sx = sx0 + dx + isx * dsx
            sy = sy0 + dy + isy * dsy
            for igy in range(ngy):
                gy = (sy - dy + gy0 + igy * dgy) if not obc else (gy0 + igy * dgy)
                for igx in range(ngx):
                    gx = (sx - dx + gx0 + igx * dgx) if not obc else (gx0 + igx * dgx)
                    src_x.append(sx)
                    src_y.append(sy)
                    src_z.append(sz0)
                    rec_x.append(gx)
                    rec_y.append(gy)
                    rec_z.append(gz0)

    return {
        "dim": dim,
        "obc": obc,
        "num_sources": nsx * nsy,
        "num_receivers_per_source": ngx * ngy,
        "num_receivers_total": len(rec_x),
        "source_x": np.asarray(src_x, dtype=np.float32),
        "source_y": np.asarray(src_y, dtype=np.float32),
        "source_z": np.asarray(src_z, dtype=np.float32),
        "receiver_x": np.asarray(rec_x, dtype=np.float32),
        "receiver_y": np.asarray(rec_y, dtype=np.float32),
        "receiver_z": np.asarray(rec_z, dtype=np.float32),
    }


def _write_survey_rss(output_path, traces):
    ntr = int(traces["num_receivers_total"])
    data = np.zeros((1, ntr), dtype=np.float32, order="F")
    datatype = 3 if traces["dim"] == 3 else 2
    out = rsfile(data=data, datatype=datatype)
    out.Ndims = 2
    out.geomN[:] = 0
    out.geomD[:] = 0.0
    out.geomO[:] = 0.0
    out.geomN[0] = np.uint64(1)
    out.geomN[1] = np.uint64(ntr)
    out.geomD[0] = 1.0
    out.geomD[1] = 1.0
    out.geomO[0] = 0.0
    out.geomO[1] = 0.0
    out.srcX = traces["source_x"]
    out.srcZ = traces["source_z"]
    out.GroupX = traces["receiver_x"]
    out.GroupZ = traces["receiver_z"]
    if datatype == 3:
        out.srcY = traces["source_y"]
        out.GroupY = traces["receiver_y"]
    out.write(str(output_path))


def generate_survey_rss(survey_dir, cfg_filename="survey.cfg", output_filename="Survey.rss"):
    """Generate Survey.rss directly in Python from survey.cfg."""
    survey_dir = Path(survey_dir)
    cfg_path = survey_dir / cfg_filename
    output_path = survey_dir / output_filename
    values = read_cfg_values(cfg_path)
    traces = _compute_trace_geometry(values)
    _write_survey_rss(output_path, traces)
    metadata = {
        "num_sources": traces["num_sources"],
        "num_receivers_per_source": traces["num_receivers_per_source"],
        "num_receivers_total": traces["num_receivers_total"],
        "survey_cfg": str(cfg_path),
        "survey_rss": str(output_path),
        "dim": traces["dim"],
        "obc": traces["obc"],
        "source_x": traces["source_x"],
        "source_z": traces["source_z"],
        "receiver_x": traces["receiver_x"],
        "receiver_z": traces["receiver_z"],
    }
    return metadata


__all__ = ["read_cfg_values", "update_cfg_values", "generate_survey_rss"]
