"""Bridge to the validated `rockem-suite` checkout.

THE CHECKOUT THIS WORKSHOP NEEDS is the validated one at
`~/software/new_rockem/rockem-suite` - explicit TE2D passes the layered-model
Green's-function check there, ADI TE2D does not (see that repo's
`doc/examples/validate_layered_1d_model/README.md`). `~/software/rockem-suite`
is a different, unfixed checkout, and pointing at it is not a version
inconvenience: its `rockem.greens.greens_layered_2d` is what gives the *exact*
2D line-source answer directly, and without it the only 1D reference available
is `empymod`'s 3D point dipole reconciled through a stack of ad-hoc correction
constants.

This module is the single place that locates that checkout, puts its
`python/` package and the validation examples' `shared/` directory on
`sys.path`, and re-exports what the rest of the workshop needs. Import
`scripts.modules.rockem_bridge` before anything that needs `rockem.*` or
the analytic line-source solvers.

The Green's solvers are package code (`rockem.greens`), imported as such.
`doc/examples/validate_layered_1d_model/shared/` stays on `sys.path` anyway,
because `phasor.py` (used by `fd_visualization`) lives only there and has no
package home.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_root() -> Path:
    return Path.home() / "software" / "new_rockem" / "rockem-suite"


ROCKEM_SUITE_ROOT = Path(os.environ.get("ROCKEM_SUITE_ROOT", str(_default_root()))).expanduser().resolve()

_PYTHON_DIR = ROCKEM_SUITE_ROOT / "python"
# Only `phasor.py` is still sourced from here - the Green's solvers moved
# into the `rockem.greens` package (see module docstring).
_SHARED_DIR = ROCKEM_SUITE_ROOT / "doc" / "examples" / "validate_layered_1d_model" / "shared"
_BIN_DIR = ROCKEM_SUITE_ROOT / "bin"

if not _PYTHON_DIR.exists():
    raise RuntimeError(
        f"ROCKEM_SUITE_ROOT={ROCKEM_SUITE_ROOT} has no python/ directory - set the "
        "ROCKEM_SUITE_ROOT environment variable to a valid rockem-suite checkout "
        "(the validated one, not a stale copy)."
    )

for _p in (str(_PYTHON_DIR), str(_SHARED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rockem.config as config  # noqa: E402
import rockem.model as model  # noqa: E402
import rockem.run as run  # noqa: E402
import rockem.survey as survey  # noqa: E402
import rockem.utils as utils  # noqa: E402
import rockem.wavelet as wavelet  # noqa: E402

try:
    from rockem.greens import (  # noqa: E402
        GreensSolverError,
        line_source_fields_layered,
        magnetic_line_source_fields_layered,
        magnetic_z_line_source_fields_layered,
        project_tilted_h,
        tilted_magnetic_line_source_fields_layered,
    )
except ImportError as exc:
    raise RuntimeError(
        f"ROCKEM_SUITE_ROOT={ROCKEM_SUITE_ROOT} does not provide the Hx-type magnetic "
        "line-source solver (rockem.greens.magnetic_line_source_fields_layered) - "
        "this checkout either predates the promotion of the Green's solvers into "
        "the rockem package, or is missing scipy (rockem.greens.greens_2d needs "
        "scipy.special.hankel2 at import time). Point ROCKEM_SUITE_ROOT at an "
        "up-to-date checkout and `pip install scipy`."
    ) from exc

from phasor import steady_state_phasor  # noqa: E402


def binary_path(name: str) -> Path:
    """Absolute path to a rockem-suite binary (e.g. 'mpiEmmodTE2d')."""
    p = _BIN_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - build it in {ROCKEM_SUITE_ROOT} (make mpi) first.")
    return p


__all__ = [
    "ROCKEM_SUITE_ROOT",
    "config",
    "model",
    "run",
    "survey",
    "utils",
    "wavelet",
    "GreensSolverError",
    "line_source_fields_layered",
    "magnetic_line_source_fields_layered",
    "magnetic_z_line_source_fields_layered",
    "project_tilted_h",
    "tilted_magnetic_line_source_fields_layered",
    "steady_state_phasor",
    "binary_path",
]
