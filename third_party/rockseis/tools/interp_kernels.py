import numpy as np
from scipy import ndimage

from ..utils.utils import triangle_smoothing_nd


METHOD_TO_ORDER = {
    "linear": 1,
    "bspline": 3,
    "sinc": 5,
}


def normalize_method(method: str) -> str:
    value = method.lower().strip()
    if value not in METHOD_TO_ORDER:
        raise ValueError(f"Unknown interpolation method: {method}")
    return value


def calculate_nbox(d_in: float, d_out: float, base_filter_size: int = 20) -> int:
    if d_out <= d_in:
        return 0
    ratio = d_in / d_out
    return max(1, int(base_filter_size * (1.0 - ratio)))


def apply_triangle_antialias(data: np.ndarray, nbox_by_axis) -> np.ndarray:
    out = np.asarray(data, dtype=np.float64)
    for axis, nbox in enumerate(nbox_by_axis):
        if nbox > 0 and out.shape[axis] > 1:
            out = triangle_smoothing_nd(out, nbox=int(nbox), nrep=1, axes=(axis,))
    return out


def resample_nd_regular(
    data: np.ndarray,
    in_origin,
    in_sampling,
    out_origin,
    out_sampling,
    out_shape,
    method: str,
) -> np.ndarray:
    method = normalize_method(method)
    order = METHOD_TO_ORDER[method]
    coords = []
    for axis, n_out in enumerate(out_shape):
        if data.shape[axis] <= 1:
            axis_indices = np.zeros(int(n_out), dtype=np.float64)
        else:
            axis_coords = out_origin[axis] + np.arange(int(n_out), dtype=np.float64) * out_sampling[axis]
            axis_indices = (axis_coords - in_origin[axis]) / in_sampling[axis]
            axis_indices = np.clip(axis_indices, 0.0, float(data.shape[axis] - 1))
        coords.append(axis_indices)

    mesh = np.meshgrid(*coords, indexing="ij")
    coord_stack = np.stack(mesh, axis=0)
    return ndimage.map_coordinates(
        np.asarray(data, dtype=np.float64),
        coord_stack,
        order=order,
        mode="nearest",
        prefilter=(order > 1),
    )
