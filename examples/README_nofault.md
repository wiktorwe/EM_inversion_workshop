# `Fault_1_nofault.sgy` - the laterally invariant reference

Built from `Fault_1.sgy` by replicating its column at x = 1099 m (well away from
the fault at x = 1478 m) across the whole section, so it has the identical layer
stack - 2 / 25 / 100 Ohm-m with interfaces at z = 6020 / 6070 / 6080 m - and **no
lateral structure anywhere**.

It exists so that "how far ahead can this observable see the fault?" can be
answered by a difference between two runs that share the survey, the grid, the
wavelet and the whole FD design, and differ only in the presence of the fault.
The alternative - calling the far transmitters of the fault model "background" -
is circular, because "far enough that the signal has decayed" is precisely the
quantity being measured, and it silently reports the modelling aperture when
that is what makes those transmitters quiet.

Regenerate with:

```python
import numpy as np, segyio, shutil
from scripts.modules.segy import read_resistivity_from_segy
m = read_resistivity_from_segy("examples/Fault_1.sgy")
rho, x = np.asarray(m["resistivity"], float), np.asarray(m["x"], float)
j = int(np.argmin(np.abs(x - 1100.0)))
shutil.copyfile("examples/Fault_1.sgy", "examples/Fault_1_nofault.sgy")
with segyio.open("examples/Fault_1_nofault.sgy", mode="r+", strict=False,
                 ignore_geometry=True) as f:
    col = rho[:, j].astype(np.float32)
    for i in range(f.tracecount):
        f.trace[i] = col
```

Used by [`../scripts/experiments/lookahead.py`](../scripts/experiments/lookahead.py).
