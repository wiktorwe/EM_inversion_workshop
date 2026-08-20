"""Read 2D conductivity RSS models with the workshop's axis convention.

Matches the ``_read_rss_model`` helpers in notebooks 04 and 06: data is stored
as ``(nx, nz)``, returned as a ``(nz, nx)`` grid with 1-D ``x`` and ``z`` axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from third_party.rockseis.io.rsfile import rsfile

MIN_SIGMA = 1e-12


def read_rss_model(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, z, grid)`` for a 2D RSS earth model.

    ``grid`` has shape ``(nz, nx)`` and is conductivity in S/m for ``sg*.rss``.
    """
    path = Path(path)
    f = rsfile()
    f.read(str(path))
    data = np.squeeze(np.asarray(f.data, dtype=float))
    if data.ndim != 2:
        raise ValueError(f"Expected 2D model for {path}, got shape {data.shape}")
    nx, nz = int(data.shape[0]), int(data.shape[1])
    grid = np.asarray(data.T, dtype=float)
    dx = float(f.geomD[0]) if f.geomD[0] else 1.0
    ox = float(f.geomO[0])
    iz = 2 if (len(f.geomN) > 2 and int(f.geomN[2]) > 0) else 1
    dz = float(f.geomD[iz]) if f.geomD[iz] else 1.0
    oz = float(f.geomO[iz])
    x = ox + dx * np.arange(nx)
    z = oz + dz * np.arange(nz)
    return x, z, grid


def conductivity_to_resistivity(grid, min_sigma: float = MIN_SIGMA) -> np.ndarray:
    """Convert a conductivity grid (S/m) to resistivity (Ohm-m)."""
    sigma = np.clip(np.asarray(grid, dtype=float), min_sigma, 1e12)
    return 1.0 / sigma


def resistivity_from_sg_rss(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an ``sg*.rss`` conductivity model and return ``(x, z, rho)``."""
    x, z, grid = read_rss_model(path)
    return x, z, conductivity_to_resistivity(grid)


__all__ = [
    "MIN_SIGMA",
    "conductivity_to_resistivity",
    "read_rss_model",
    "resistivity_from_sg_rss",
]
