"""Self-contained block1D inversion package for Empymod integration.

The package mirrors pyGIMLi's layered inversion workflow while keeping the
forward response and misfit logic pluggable.
"""

from .covariance import covariance_from_jacobian, uncertainty_from_result
from .interfaces import InversionConfig, InversionResult, InversionState
from .inversion import run_block1d_inversion, with_start_model
from .model import (
    create_start_model,
    pack_model,
    parameter_count,
    split_model,
    validate_positive_model,
)
from .plotting import plot_block_model, plot_model_with_errorbars

__all__ = [
    "InversionConfig",
    "InversionResult",
    "InversionState",
    "covariance_from_jacobian",
    "uncertainty_from_result",
    "run_block1d_inversion",
    "with_start_model",
    "create_start_model",
    "pack_model",
    "parameter_count",
    "split_model",
    "validate_positive_model",
    "plot_block_model",
    "plot_model_with_errorbars",
]

