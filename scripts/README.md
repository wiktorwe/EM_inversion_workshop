# Workshop Scripts

This folder is the module-oriented script codebase used by the GUI.

- `scripts/modules/segy.py`: SEGY read/write helpers
- `scripts/modules/source.py`: wavelet creation helper
- `scripts/modules/survey.py`: survey config and Survey.rss helpers
- `scripts/modules/fd.py`: FD design, interpolation, and mod.cfg update helpers
- `scripts/templates/survey.cfg`: survey template copied to a temp workspace
- `scripts/templates/mod.cfg`: fallback mod.cfg template

The notebook imports from `scripts.modules.*` and templates in `scripts/templates`
so logic is not tied to temporary project folders.
