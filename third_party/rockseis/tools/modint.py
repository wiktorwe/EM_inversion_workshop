import argparse
import os
import sys
import tempfile

import numpy as np

from ..io.rsfile import MAXDIMS
from ..io.rsfile import rsfile
from .interp_kernels import apply_triangle_antialias
from .interp_kernels import calculate_nbox
from .interp_kernels import normalize_method
from .interp_kernels import resample_nd_regular


DATA2D = 2
DATA3D = 3


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _read_input(path: str) -> rsfile:
    tmp_path = None
    try:
        read_path = path
        if path == "-":
            payload = sys.stdin.buffer.read()
            if not payload:
                raise ValueError("No RSS bytes available on stdin.")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            read_path = tmp_path
        f = rsfile()
        f.read(read_path)
        return f
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _write_output(path: str, out_file: rsfile) -> None:
    tmp_path = None
    try:
        if path == "-":
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            out_file.write(tmp_path)
            with open(tmp_path, "rb") as handle:
                sys.stdout.buffer.write(handle.read())
        else:
            out_file.write(path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _extract_geometry(infile: rsfile):
    n_in = np.asarray(infile.geomN, dtype=np.int64).copy()
    d_in = np.asarray(infile.geomD, dtype=np.float64).copy()
    o_in = np.asarray(infile.geomO, dtype=np.float64).copy()
    d_in[d_in == 0.0] = 1.0
    return n_in, d_in, o_in


def _compute_output_geometry(args, n_in, d_in, o_in):
    d_out = d_in.copy()
    o_min = o_in.copy()
    o_max = o_in + np.maximum(n_in - 1, 0) * d_in

    for i in range(MAXDIMS):
        d_value = getattr(args, f"d{i + 1}f")
        o_value = getattr(args, f"o{i + 1}f")
        m_value = getattr(args, f"max{i + 1}")
        if d_value is not None:
            d_out[i] = float(d_value)
        if o_value is not None:
            o_min[i] = float(o_value)
        if m_value is not None:
            o_max[i] = float(m_value)
        if n_in[i] > 0 and d_out[i] == 0.0:
            raise ValueError("Output sampling interval cannot be zero.")

    n_out = np.zeros(MAXDIMS, dtype=np.int64)
    for i in range(MAXDIMS):
        if n_in[i] > 0:
            count = int(np.floor((o_max[i] - o_min[i]) / d_out[i])) + 1
            n_out[i] = max(1, count)

    return d_out, o_min, o_max, n_out


def _prepare_antialias(args, n_in, d_in, d_out):
    use_antialias = bool(args.antialias)
    nbox_vals = np.zeros(MAXDIMS, dtype=np.int64)
    if use_antialias:
        for i in range(MAXDIMS):
            if n_in[i] > 1 and d_out[i] > d_in[i]:
                nbox_vals[i] = calculate_nbox(d_in[i], d_out[i])
    return use_antialias, nbox_vals


def _process_seismic(infile, method, d_in, d_out, o_in, o_min, n_out, use_antialias, nbox_vals):
    raw = np.asarray(infile.data, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    else:
        raw = raw.reshape((raw.shape[0], -1), order="F")

    n1_out = int(n_out[0])
    ntr = int(raw.shape[1])
    work = raw

    if use_antialias and nbox_vals[0] > 0 and d_out[0] > d_in[0]:
        work = apply_triangle_antialias(work, (int(nbox_vals[0]), 0))

    out_data = resample_nd_regular(
        work,
        in_origin=(o_in[0], 0.0),
        in_sampling=(d_in[0], 1.0),
        out_origin=(o_min[0], 0.0),
        out_sampling=(d_out[0], 1.0),
        out_shape=(n1_out, ntr),
        method=method,
    )

    dtype = np.float32 if int(infile.data_format) == 4 else np.float64
    out_file = rsfile(data=np.asarray(out_data, dtype=dtype, order="F"), datatype=int(infile.type))
    out_file.data_format = int(infile.data_format)
    out_file.header_format = int(infile.header_format)
    out_file.type = int(infile.type)
    out_file.Nheader = int(infile.Nheader)

    out_file.geomN[:] = 0
    out_file.geomD[:] = 0.0
    out_file.geomO[:] = 0.0
    out_file.geomN[0] = np.uint64(n1_out)
    out_file.geomN[1] = np.uint64(ntr)
    out_file.geomD[0] = float(d_out[0])
    out_file.geomD[1] = float(d_in[1] if d_in[1] != 0.0 else 1.0)
    out_file.geomO[0] = float(o_min[0])
    out_file.geomO[1] = float(o_in[1])
    out_file.Ndims = 2

    if infile.Nheader > 0:
        out_file.srcX = np.asarray(infile.srcX[:ntr], dtype=np.float32)
        out_file.srcZ = np.asarray(infile.srcZ[:ntr], dtype=np.float32)
        out_file.GroupX = np.asarray(infile.GroupX[:ntr], dtype=np.float32)
        out_file.GroupZ = np.asarray(infile.GroupZ[:ntr], dtype=np.float32)
        if infile.Nheader == 6:
            out_file.srcY = np.asarray(infile.srcY[:ntr], dtype=np.float32)
            out_file.GroupY = np.asarray(infile.GroupY[:ntr], dtype=np.float32)

    return out_file


def _process_regular_generic(infile, method, n_in, d_in, d_out, o_in, o_min, n_out, use_antialias, nbox_vals):
    data = np.asarray(infile.data, dtype=np.float64, order="F")
    active_idx = [i for i in range(MAXDIMS) if n_in[i] > 0]
    if not active_idx:
        raise ValueError("Input has no active dimensions.")
    if len(active_idx) > 3:
        raise ValueError(f"Unsupported dimensionality for REGULAR/GENERIC data: {len(active_idx)}")

    if data.ndim != len(active_idx):
        data = data.reshape(tuple(int(n_in[i]) for i in active_idx), order="F")

    out_shape = tuple(int(n_out[i]) for i in active_idx)
    in_origin = tuple(float(o_in[i]) for i in active_idx)
    out_origin = tuple(float(o_min[i]) for i in active_idx)
    in_sampling = tuple(float(d_in[i]) for i in active_idx)
    out_sampling = tuple(float(d_out[i]) for i in active_idx)
    nbox_active = tuple(int(nbox_vals[i]) if use_antialias else 0 for i in active_idx)

    work = data
    if use_antialias and any(v > 0 for v in nbox_active):
        work = apply_triangle_antialias(work, nbox_active)

    out_data = resample_nd_regular(
        work,
        in_origin=in_origin,
        in_sampling=in_sampling,
        out_origin=out_origin,
        out_sampling=out_sampling,
        out_shape=out_shape,
        method=method,
    )

    dtype = np.float32 if int(infile.data_format) == 4 else np.float64
    out_file = rsfile(data=np.asarray(out_data, dtype=dtype, order="F"))
    out_file.data_format = int(infile.data_format)
    out_file.header_format = int(infile.header_format)
    out_file.type = int(infile.type)
    out_file.Nheader = 0

    out_file.geomN[:] = 0
    out_file.geomD[:] = 0.0
    out_file.geomO[:] = 0.0
    for axis, dim in enumerate(active_idx):
        out_file.geomN[dim] = np.uint64(out_shape[axis])
        out_file.geomD[dim] = out_sampling[axis]
        out_file.geomO[dim] = out_origin[axis]
    out_file.Ndims = len(active_idx)
    return out_file


def run_modint(args) -> int:
    method = normalize_method(args.method)
    infile = _read_input(args.input)

    n_in, d_in, o_in = _extract_geometry(infile)
    d_out, o_min, _o_max, n_out = _compute_output_geometry(args, n_in, d_in, o_in)
    use_antialias, nbox_vals = _prepare_antialias(args, n_in, d_in, d_out)

    if args.verbose:
        print(f"method={method} antialias={use_antialias}", file=sys.stderr)
        print(f"input_type={int(infile.type)} data_format={int(infile.data_format)}", file=sys.stderr)

    if int(infile.type) in {DATA2D, DATA3D}:
        out_file = _process_seismic(infile, method, d_in, d_out, o_in, o_min, n_out, use_antialias, nbox_vals)
    else:
        out_file = _process_regular_generic(
            infile, method, n_in, d_in, d_out, o_in, o_min, n_out, use_antialias, nbox_vals
        )

    _write_output(args.output, out_file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpolate regularly sampled Rockseis models/data.")
    parser.add_argument("input", nargs="?", default="-", help="Input RSS file path, or '-' for stdin.")
    parser.add_argument("output", nargs="?", default="-", help="Output RSS file path, or '-' for stdout.")
    parser.add_argument("--method", default="linear", choices=["linear", "bspline", "sinc"])
    parser.add_argument("--antialias", type=_parse_bool, default=True)
    parser.add_argument("--verbose", type=_parse_bool, default=False)

    for i in range(1, MAXDIMS + 1):
        parser.add_argument(f"--d{i}f", type=float, default=None)
        parser.add_argument(f"--o{i}f", type=float, default=None)
        parser.add_argument(f"--max{i}", type=float, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_modint(args)
    except Exception as exc:
        print(f"modint.py error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
