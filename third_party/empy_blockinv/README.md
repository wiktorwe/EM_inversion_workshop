# `pygimli.empy_blockinv`

Self-contained 1D block inversion helpers designed to mimic pyGIMLi's
block-model workflow while allowing pluggable forward and misfit callbacks.

## Quick start

```python
import numpy as np
from pygimli.empy_blockinv import InversionConfig, run_block1d_inversion

cfg = InversionConfig(n_layers=4, lam=1000.0, lambda_factor=0.8)
result = run_block1d_inversion(
    observed=data,
    error=err,
    config=cfg,
    forward_fn=my_empymod_forward,
    misfit_fn=my_phase_residual,  # optional
)
```

## Key behaviors mirrored from pyGIMLi

- Layered parameter vector:
  `[thk_1..thk_(nlay-1), rho_1..rho_n]`
- Log-domain model updates:
  `m_new = inv(trans(m_old) + dm)`
- Finite-difference Jacobian with multiplicative perturbation (`*1.05` default)
- Marquardt-like damping schedule (`lam`, `lambda_factor`, `min_dphi_percent`)
- cType=0-like damping regularization in transformed model space
- Linearized covariance and tutorial-style uncertainty bars

## Integration notes

- Plug your Empymod forward into `forward_fn(model, context) -> predicted`.
- If you need custom phase handling (wrapping, bespoke weighting), pass
  `misfit_fn(observed, predicted, error, context) -> residual`.
- See `examples/example_empymod_integration.py` for a heavily commented
  integration template.

