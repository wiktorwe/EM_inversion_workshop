import numpy as np


def triangle_smoothing(data, nbox, nrep=1):
    n = len(data)
    smoothed = np.zeros_like(data)
    weights = np.zeros(2 * nbox + 1)
    for i in range(nbox + 1):
        weights[i] = i + 1
        weights[2 * nbox - i] = i + 1
    weights = weights / np.sum(weights)

    for _ in range(nrep):
        for i in range(n):
            sum_val = 0.0
            sum_weight = 0.0
            for j in range(-nbox, nbox + 1):
                idx = i + j
                if 0 <= idx < n:
                    sum_val += data[idx] * weights[j + nbox]
                    sum_weight += weights[j + nbox]
            smoothed[i] = sum_val / sum_weight if sum_weight > 0 else data[i]
        if nrep > 1:
            data = smoothed.copy()
    return smoothed


def triangle_smoothing_nd(data, nbox, nrep=1, axes=None):
    data = np.asarray(data)
    ndim = data.ndim
    if ndim == 1:
        return triangle_smoothing(data, nbox, nrep)

    if isinstance(nbox, int):
        nbox = (nbox,) * ndim
    if len(nbox) != ndim:
        raise ValueError(f"nbox must have length {ndim} for {ndim}D data, got {len(nbox)}")

    if axes is None:
        axes = tuple(range(ndim))
    if not all(0 <= ax < ndim for ax in axes):
        raise ValueError(f"Axes must be between 0 and {ndim - 1}")

    smoothed = data.copy()
    for _ in range(nrep):
        for axis in axes:
            smoothed = _smooth_along_axis(smoothed, nbox[axis], axis)
    return smoothed


def _smooth_along_axis(data, nbox, axis):
    weights = np.zeros(2 * nbox + 1)
    for i in range(nbox + 1):
        weights[i] = i + 1
        weights[2 * nbox - i] = i + 1
    weights = weights / np.sum(weights)

    axis_size = data.shape[axis]
    result = np.zeros_like(data)

    for idx in np.ndindex(data.shape[:axis] + data.shape[axis + 1 :]):
        slice_obj = list(idx)
        slice_obj.insert(axis, slice(None))
        slice_obj = tuple(slice_obj)
        data_slice = data[slice_obj]
        result_slice = np.zeros_like(data_slice)

        for i in range(axis_size):
            sum_val = 0.0
            sum_weight = 0.0
            for j in range(-nbox, nbox + 1):
                slice_idx = i + j
                if 0 <= slice_idx < axis_size:
                    sum_val += data_slice[slice_idx] * weights[j + nbox]
                    sum_weight += weights[j + nbox]
            result_slice[i] = sum_val / sum_weight if sum_weight > 0 else data_slice[i]

        result[slice_obj] = result_slice
    return result
