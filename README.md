# EM Inversion Workshop

This repository contains an interactive workshop for electromagnetic (EM) forward modelling, inversion, and result analysis.

## 1) Prerequisites

Install these before starting:

- Python 3.10+ (or a conda environment with Python)
- `mpirun` (OpenMPI or compatible MPI runtime)
- RockEM suite binary at:
  `~/software/rockem-suite/bin/mpiEmmodADITE2d`

Install Python packages:

```bash
python3 -m pip install voila ipywidgets plotly numpy
```

If you use conda, activate your environment first and run the same command.

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
- If a GUI does not open, ensure `voila` is installed in the active environment.
