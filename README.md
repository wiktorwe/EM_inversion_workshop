# EM Inversion Workshop

This repository contains an interactive workshop for electromagnetic (EM) forward modelling, inversion, and result analysis.

## 1) Prerequisites

Install these before starting:

- Python 3.10+
- `mpirun` (OpenMPI or compatible MPI runtime)
- RockEM suite binaries at:
  `~/software/rockem-suite/bin/mpiEmmodADITE2d`
  `~/software/rockem-suite/bin/mpiEminvADITE2d`

### Environment setup

**Option A — Conda**

```bash
conda create -n em_workshop python=3.10
conda activate em_workshop
```

**Option B — Python venv**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install --upgrade pip
```

### Install dependencies

With your environment activated, install the required Python packages:

```bash
pip install voila ipywidgets plotly numpy ipykernel matplotlib scipy segyio
```

| Package     | Purpose |
|-------------|---------|
| voila       | Serves notebooks as standalone web apps (code hidden) |
| ipywidgets  | Interactive widgets in the GUI |
| plotly      | Interactive plots |
| numpy       | Numerical arrays |
| ipykernel   | Jupyter kernel for Python (required for Voila) |
| matplotlib  | Plotting (used by workshop modules) |
| scipy       | Scientific computing (used by workshop modules) |
| segyio      | Read/write SEG-Y files (required by 01_fw_setup) |

**Conda users:** After installing, register the kernel so Voila can find it:

```bash
python -m ipykernel install --user --name=em_workshop --display-name="Python (em_workshop)"
```

## 2) Get the workshop

```bash
git clone git@github.com:wiktorwe/EM_inversion_workshop.git
cd EM_inversion_workshop
```

## 3) Run the workshop GUIs

Launch each step from the project root:

```bash
./start_01_fw_setup.sh
./start_02_visualization.sh
./start_03_inversion.sh
./start_04_results.sh
```

Each command starts a Voila app for that stage.

## 4) Recommended workflow

1. **Step 01 - FW setup**
   - Load a SEG-Y resistivity model.
   - Configure source/survey settings.
   - Click **Generate FD inputs (Finalize setup)**.
   - This step creates the FD workspace automatically.

2. **Step 02 - Visualization**
   - Run forward modelling.
   - Inspect Hx/Hz data, amplitudes, and phases.

3. **Step 03 - Inversion**
   - Generate inversion inputs.
   - Start inversion and monitor progress.

4. **Step 04 - Results**
   - Compare models and data.
   - Export outputs as needed.

## 5) Troubleshooting

- If a step reports missing setup data, run Step 01 first and finalize setup.
- If modelling cannot start, verify:
  - `mpirun` is available in your shell
  - `~/software/rockem-suite/bin/mpiEmmodADITE2d` exists
- If inversion cannot start, verify:
  - `mpirun` is available in your shell
  - `~/software/rockem-suite/bin/mpiEminvADITE2d` exists
- If a GUI does not open, ensure `voila` is installed in the active environment.
- If you see "No Jupyter kernel for language 'python' found", install `ipykernel` and register the kernel (see Install dependencies above).
