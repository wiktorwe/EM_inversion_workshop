"""Minimal transform implementations used by the inversion core."""

from __future__ import annotations

import numpy as np


class IdentityTransform:
    """Identity transform f(x)=x."""

    def trans(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def inv_trans(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(x, dtype=float))

    def update(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return self.inv_trans(self.trans(x) + np.asarray(dy, dtype=float))


class LogTransform:
    """Log transform with optional lower/upper bounds.

    This follows pyGIMLi semantics closely:
    - without upper bound: f(x) = log(x - lb)
    - with upper bound: f(x) = log(x-lb) - log(ub-x)
    """

    def __init__(self, lower_bound: float = 0.0, upper_bound: float | None = None):
        self.lower_bound = float(lower_bound)
        self.upper_bound = None if upper_bound is None else float(upper_bound)

    def _rangify(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float).copy()
        # mirror TRANSTOL from pyGIMLi's trans.h
        trans_tol = 1e-8
        lb = self.lower_bound * (1.0 + trans_tol)
        out = np.maximum(out, lb)
        if self.upper_bound is not None:
            ub = self.upper_bound * (1.0 - trans_tol)
            out = np.minimum(out, ub)
        return out

    def trans(self, x: np.ndarray) -> np.ndarray:
        x = self._rangify(x)
        if self.upper_bound is None:
            return np.log(x - self.lower_bound)
        return np.log(x - self.lower_bound) - np.log(self.upper_bound - x)

    def inv_trans(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.upper_bound is None:
            return np.exp(y) + self.lower_bound
        expy = np.exp(np.minimum(y, 50.0))
        return (expy * self.upper_bound + self.lower_bound) / (expy + 1.0)

    def deriv(self, x: np.ndarray) -> np.ndarray:
        x = self._rangify(x)
        if self.upper_bound is None:
            return 1.0 / (x - self.lower_bound)
        return 1.0 / (x - self.lower_bound) + 1.0 / (self.upper_bound - x)

    def update(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        # This is the key pyGIMLi update rule: inv(trans(x) + dm)
        return self.inv_trans(self.trans(x) + np.asarray(dy, dtype=float))

