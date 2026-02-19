"""SEGY helpers for workshop workflows."""

import shutil
from pathlib import Path

import numpy as np
import segyio

from third_party.rockseis.io.rsfile import rsfile


def _resolve_scalco(raw_scalco):
    if raw_scalco < 0:
        return 1.0 / np.abs(raw_scalco)
    if raw_scalco > 0:
        return float(raw_scalco)
    return 1.0


def read_resistivity_from_segy(file_path):
    """Read a SEGY file and return resistivity data with sampling metadata."""
    with segyio.open(file_path, mode="r", strict=False, ignore_geometry=True) as segyfile:
        if segyfile.tracecount < 2:
            raise ValueError("Cannot compute horizontal spacing from a single trace.")

        oz = float(segyfile.samples[0])
        dz = float(segyfile.samples[1] - segyfile.samples[0])
        scalco = _resolve_scalco(segyfile.header[0][segyio.TraceField.SourceGroupScalar])

        x0 = segyfile.header[0][segyio.TraceField.SourceX]
        x1 = segyfile.header[1][segyio.TraceField.SourceX]
        dx = abs(x1 - x0) * scalco

        x = segyfile.attributes(segyio.TraceField.SourceX)[:] * scalco
        z = segyfile.samples
        ox = float(x[0])
        resistivity = segyfile.trace.raw[:].T

    return {
        "resistivity": resistivity,
        "oz": oz,
        "dz": dz,
        "dx": float(dx),
        "ox": ox,
        "x": x,
        "z": z,
    }


def read_segy_template_metadata(file_path):
    """Read SEG-Y metadata used as export template geometry."""
    with segyio.open(file_path, mode="r", strict=False, ignore_geometry=True) as segyfile:
        if segyfile.tracecount < 2:
            raise ValueError("Cannot compute horizontal spacing from a single trace.")
        if len(segyfile.samples) < 2:
            raise ValueError("Cannot compute vertical spacing from fewer than two samples.")

        raw_scalco = segyfile.header[0][segyio.TraceField.SourceGroupScalar]
        scalco = _resolve_scalco(raw_scalco)

        x = np.asarray(segyfile.attributes(segyio.TraceField.SourceX)[:], dtype=np.float64) * scalco
        z = np.asarray(segyfile.samples, dtype=np.float64)

        x0 = float(x[0])
        x1 = float(x[1])
        z0 = float(z[0])
        z1 = float(z[1])

    dx = float(abs(x1 - x0))
    dz = float(z1 - z0)
    return {
        "nx": int(x.size),
        "nz": int(z.size),
        "ox": float(x0),
        "oz": float(z0),
        "dx": dx,
        "dz": dz,
        "x": x,
        "z": z,
        "scalco": float(scalco),
        "raw_scalco": int(raw_scalco),
    }


def _axis_sampling(axis_values, axis_name):
    values = np.asarray(axis_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{axis_name} must be a 1D array with at least two samples.")
    return float(values[1] - values[0])


def resample_resistivity_to_template_grid(
    resistivity_grid,
    x_model,
    z_model,
    template_meta,
    method="linear",
):
    """
    Resample resistivity (nz, nx) to template SEG-Y grid using workshop interpolator.
    Returns (resampled_grid, was_resampled).
    """
    rho = np.asarray(resistivity_grid, dtype=np.float64)
    if rho.ndim != 2:
        raise ValueError(f"Expected resistivity_grid with 2D shape (nz, nx), got {rho.shape}")

    x_model = np.asarray(x_model, dtype=np.float64)
    z_model = np.asarray(z_model, dtype=np.float64)
    if x_model.ndim != 1 or z_model.ndim != 1:
        raise ValueError("x_model and z_model must be 1D arrays.")

    nz_model, nx_model = int(rho.shape[0]), int(rho.shape[1])
    if nx_model != int(x_model.size) or nz_model != int(z_model.size):
        raise ValueError(
            "resistivity_grid shape does not match x_model/z_model sizes: "
            f"grid={rho.shape}, x={x_model.size}, z={z_model.size}"
        )

    dx_model = _axis_sampling(x_model, "x_model")
    dz_model = _axis_sampling(z_model, "z_model")

    nx_t = int(template_meta["nx"])
    nz_t = int(template_meta["nz"])
    ox_t = float(template_meta["ox"])
    oz_t = float(template_meta["oz"])
    dx_t = float(template_meta["dx"])
    dz_t = float(template_meta["dz"])

    same_grid = (
        nx_model == nx_t
        and nz_model == nz_t
        and np.isclose(float(x_model[0]), ox_t, rtol=0.0, atol=1e-6)
        and np.isclose(float(z_model[0]), oz_t, rtol=0.0, atol=1e-6)
        and np.isclose(dx_model, dx_t, rtol=0.0, atol=1e-6)
        and np.isclose(dz_model, dz_t, rtol=0.0, atol=1e-6)
    )
    if same_grid:
        return np.asarray(rho, dtype=np.float64), False

    try:
        from third_party.rockseis.tools.interp_kernels import resample_nd_regular
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Interpolation requires scipy via third_party.rockseis.tools.interp_kernels. "
            "Install scipy in the active environment."
        ) from exc

    out = resample_nd_regular(
        data=rho,
        in_origin=(float(z_model[0]), float(x_model[0])),
        in_sampling=(float(dz_model), float(dx_model)),
        out_origin=(oz_t, ox_t),
        out_sampling=(dz_t, dx_t),
        out_shape=(nz_t, nx_t),
        method=method,
    )
    return np.asarray(out, dtype=np.float64), True


def write_resistivity_to_segy_from_template(
    template_segy_path,
    output_segy_path,
    resistivity_grid,
    x_model,
    z_model,
    method="linear",
):
    """
    Write resistivity to SEG-Y using template geometry/header layout.
    Trace/file headers are preserved by copying the template file.
    """
    template_segy_path = Path(template_segy_path)
    output_segy_path = Path(output_segy_path)
    if not template_segy_path.exists():
        raise FileNotFoundError(f"SEG-Y template file not found: {template_segy_path}")

    template_meta = read_segy_template_metadata(template_segy_path)
    rho_out, did_resample = resample_resistivity_to_template_grid(
        resistivity_grid=resistivity_grid,
        x_model=x_model,
        z_model=z_model,
        template_meta=template_meta,
        method=method,
    )

    output_segy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_segy_path, output_segy_path)

    traces = np.asarray(rho_out.T, dtype=np.float32)
    expected = (int(template_meta["nx"]), int(template_meta["nz"]))
    if traces.shape != expected:
        raise ValueError(f"Export trace shape mismatch: got {traces.shape}, expected {expected}")

    with segyio.open(output_segy_path, mode="r+", strict=False, ignore_geometry=True) as segyfile:
        if segyfile.tracecount != expected[0]:
            raise ValueError(
                f"Output SEG-Y tracecount mismatch: got {segyfile.tracecount}, expected {expected[0]}"
            )
        if len(segyfile.samples) != expected[1]:
            raise ValueError(
                f"Output SEG-Y sample count mismatch: got {len(segyfile.samples)}, expected {expected[1]}"
            )
        segyfile.trace.raw[:] = traces

    return {
        "output_path": str(output_segy_path),
        "template_path": str(template_segy_path),
        "interpolated": bool(did_resample),
        "nx": int(template_meta["nx"]),
        "nz": int(template_meta["nz"]),
        "dx": float(template_meta["dx"]),
        "dz": float(template_meta["dz"]),
        "ox": float(template_meta["ox"]),
        "oz": float(template_meta["oz"]),
    }


def save_resistivity_npz(output_path, resistivity, oz, dz, dx, x, z):
    """Save resistivity and sampling metadata to NPZ."""
    np.savez(output_path, resistivity=resistivity, oz=oz, dz=dz, dx=dx, x=x, z=z)


def write_sg_ep_rss(resistivity, dx, dz, ox, oz, sg_path, ep_path, ep_value=7.0):
    """Write conductivity (sg) and permittivity (ep) RSS files."""
    sg = 1.0 / resistivity
    nz, nx = sg.shape
    sg = np.reshape(sg.T, [nx, 1, nz])
    ep = np.ones(sg.shape) * ep_value

    outfile = rsfile(sg)
    outfile.geomD[0] = dx
    outfile.geomD[1] = dx
    outfile.geomD[2] = dz
    outfile.geomO[0] = ox
    outfile.geomO[2] = oz
    outfile.write(sg_path)

    outfile.data = ep
    outfile.write(ep_path)


__all__ = [
    "read_resistivity_from_segy",
    "read_segy_template_metadata",
    "resample_resistivity_to_template_grid",
    "save_resistivity_npz",
    "write_resistivity_to_segy_from_template",
    "write_sg_ep_rss",
]
