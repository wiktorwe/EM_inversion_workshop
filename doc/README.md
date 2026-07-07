# Workshop documentation

## Contents

| Document | Description |
|----------|-------------|
| [`../README.md`](../README.md) | Quick start: install, configure, run notebooks, workspace layout |
| [`gui_manual.tex`](gui_manual.tex) | Full GUI manual (LaTeX source) |
| [`gui_manual.pdf`](gui_manual.pdf) | Built PDF manual |
| [`../examples/README.md`](../examples/README.md) | Example resistivity model (`Fault_1.sgy`) |
| [`../scripts/README.md`](../scripts/README.md) | Python modules and templates |

## Building the GUI manual (PDF)

From this directory (`doc/`):

```bash
pdflatex gui_manual.tex
```

Or with latexmk (run twice if needed for references):

```bash
latexmk gui_manual.tex
```

Output: `gui_manual.pdf`. Copy to the repo root if you want `gui_manual.pdf` at the top level:

```bash
cp gui_manual.pdf ../gui_manual.pdf
```

## Workshop structure (summary)

All notebook-generated files live under `workspace/` (gitignored):

```
workspace/
  2D/
    forward/              # Step 01–02: FD model, mod.cfg, shot gathers
    inversion/
      input/              # Step 03: prepared inversion inputs
      Run{N}/             # Step 03: 2D inversion runs
    results/Run{N}/       # Step 04: SEG-Y exports
  1D/
    inversion/Run{N}/     # Step 05: 1D inversion runs
    results/Run{N}/       # Step 06: SEG-Y exports
```

Machine-specific settings are stored in `workshop_config.json` (gitignored), written by **Step 00 — Configure**. Fields include the rockem-suite path, MPI launcher, default SEG-Y, workspace directory, and optional GPU flags (`use_gpu_forward_2d`, `use_gpu_inversion_2d`) for 2D TE2D forward and inversion.

To restore a pristine checkout:

```bash
./clean.sh
```
