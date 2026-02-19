# Vendored Rockseis Python modules

This folder contains the Rockseis Python code required by this workshop so the
project can run without an external `rockseis-suite/python` checkout.

Included modules are intentionally scoped to what this repository uses:

- `io/rsfile.py`
- `tools/modint.py`
- `tools/interp_kernels.py`
- `utils/utils.py` (triangle smoothing utilities used by interpolation)
- `wavelet/wavelet.py`

Workshop code imports these modules through:

- `third_party.rockseis.io.rsfile`
- `third_party.rockseis.tools.modint`
- `third_party.rockseis.wavelet.wavelet`

If you later update Rockseis functionality, keep this folder in sync.
