#!/usr/bin/env python3
"""Rebuild `workspace/2D/forward` as a full acquisition matrix, from scratch.

Step 01's headless path (`headless.build_forward_matrix`) with the workshop's
stored production survey: 30 transmitters at z = 6050 m along a horizontal well,
two colinear receivers at -13.1 / -25.3 m, `examples/Fault_1.sgy`, FD order 6.
Three tones x two sources = six datasets.

Why this exists: `sg.rss`/`ep.rss` are now resampled with `method="nearest"`
instead of `"linear"`, so every model on disk built before that change has
one-cell interface RAMPS where the new ones have steps. That is a change to the
forward model, so no dataset or calibration built before it is comparable with
one built after - the whole workspace has to be rebuilt, not topped up.

The existing forward directory is MOVED aside (never deleted), so the old
linear-resampled workspace stays available for comparison.

    python scripts/rebuild_matrix_workspace.py --dry-run
    python scripts/rebuild_matrix_workspace.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modules.workshop_config import load_config  # noqa: E402

# 2/4/6 kHz is the band the skill's per-dataset table is written against
# (eps_r_used 1198.3 / 599.2 / 399.4, dx 1.40 / 0.95 / 0.80 m), so a workspace
# built here is directly comparable with the numbers recorded there. 1 kHz is
# dropped: it is the longest record in the band and buys the least resolution.
FREQS_HZ = (2000.0, 4000.0, 6000.0)
SOURCES = ("HX", "HZ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freqs", default=",".join(f"{f:g}" for f in FREQS_HZ),
                    help="Comma-separated tones in Hz.")
    ap.add_argument("--sources", default=",".join(SOURCES),
                    help="Comma-separated source fields (HX/HZ).")
    ap.add_argument("--keep-existing", action="store_true",
                    help="Write into the existing forward dir instead of moving it aside.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from scripts.modules.headless import SetupParams, build_forward_matrix, fd_design_for

    cfg = load_config(ROOT)
    fwd = Path(cfg.fwd_2d_dir)
    freqs = tuple(float(v) for v in args.freqs.split(",") if v.strip())
    sources = tuple(s.strip().upper() for s in args.sources.split(",") if s.strip())

    p = SetupParams(flist_hz=freqs)
    print(f"forward root: {fwd}")
    print(f"{len(freqs)} tone(s) x {len(sources)} source(s) = "
          f"{len(freqs) * len(sources)} dataset(s)\n")
    for f in freqs:
        d, order, lpml = fd_design_for(replace(p, flist_hz=(f,), f_min_hz=f, f_max_hz=f),
                                       max_depth_offset_m=0.0)
        print(f"  {f:6.0f} Hz  dx={d['dx_m']:.3f} m  dt={d['dt_s']:.4e} s  "
              f"eps_r={d['eps_r_used']:.1f}  apertx={d['apertx_m']:.1f} m  "
              f"order={order} lpml={lpml}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if fwd.exists() and not args.keep_existing:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        aside = fwd.parent / f"{fwd.name}_pre_nearest_{stamp}"
        print(f"\nmoving the existing forward dir aside -> {aside}")
        shutil.move(str(fwd), str(aside))

    t0 = time.time()
    manifest = build_forward_matrix(fwd, p, source_fields=sources, verbose=True)
    print(f"\nwrote {manifest['n_datasets']} dataset(s) in {time.time() - t0:.1f} s")
    print("next: python scripts/run_matrix.py --method lateral_average_true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
