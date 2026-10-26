# 1.3 - Batch Template Setup

## Two modes of data generation

This step supports **two** data-generation modes:

1. **HPC batch mode** (the original, documented below) - for *expensive*
   external solvers (e.g. VPSC / EVP-FFT crystal plasticity) that must be fanned
   out to one SLURM run folder per design-of-experiment sample. Driven by
   `references/setup_run_folders.sh`.
2. **Local mode** (using a local data generation model) - for *cheap* models that can be
   evaluated in-process on a laptop. Used by, e.g., the **chemical-kinetics example**
   (Cantera GRI-Mech 3.0 autoignition). No per-run folders, no SLURM, no HPC:
   the script reads `samples.csv` and writes **one** `.pickle` directly.

---

## HPC batch mode

## Purpose

`setup_run_folders.sh` automates the creation of **run folders** that will be submitted to an HPC scheduler (SLURM). Each folder corresponds to a single line (one sample) in `samples.csv`, which is the design‑of‑experiments file generated in the *12_Design_of_Experiments* stage.

The script:

1. **Reads** the `samples.csv` created in step 1.2 - Design of Experiment. The file should be copied to the template input folder.
2. **Creates** a new directory (`run_1`, `run_2`, …).
3. **Copies** only the files that are needed for a simulation (e.g., `*.slurm`, `*.in`, `*.dat`, `*.sx`) from a template folder (`template_run_folder`). These are specific to the model and its expected input format. This should be adjusted in `setup_run_folders.sh`  for each model.
4. **Replaces** placeholder strings inside the copied files with the concrete values read from the CSV line. These placeholders should be consistent with the header strings for each variable in the `references/samples.csv` file that was generated in step *12_Design_of_Experiments*.
5. (Optionally) **launches** an AutoJob helper to submit the whole batch at once (`--autojob`).

Because the script works purely with *text substitution* (`sed`), it is agnostic to the actual physics solver – any code that can be driven by a set 
of input files and a SLURM batch script will work as long as the placeholders are present. Below are instructions to help you understand and adapt the shell script for your purpose.

### Adapting `setup_run_folders.sh`

The script is set up for a three-parameter example (`TEMP`, `VMJ2`, `RHOC`). To use it with your own solver, edit these parts:

1. **CSV columns:** list your `samples.csv` columns, in order, in the `read` line.
   ```bash
   tail -n +2 "$input_file" | while IFS=, read -r TEMP VMJ2 RHOC; do
   ```

2. **Files copied:** edit the `--include` patterns to match your template file types.
   ```bash
   rsync -avz --include="*.slurm" --include="*.in" --include="*.dat" --include="*.sx" --exclude="*"$sourcedir/ $newdir/
   ```

3. **Job name:** the text after `--job-name=` must match the placeholder in your `run.slurm`.
   ```bash
   sed -i "s/#SBATCH --job-name=LApx/#SBATCH --job-name=$newdir/g" $newdir/run.slurm
   ```

4. **Placeholders:** add or remove one `sed` line per placeholder. File names are case-sensitive and must match your template files.
   ```bash
   sed -i "s/TEMP_PLACE/$TEMP/g" "$newdir/BCFile.in"
   sed -i "s/RHOC_PLACE/$RHOC/g" "$newdir/single_crystal_model.sx"
   ```

5. **Derived or fixed values:** replace the problem-specific calculations (`time_ramp`, `stress_rate_hoop`, `stress_rate_rad`) with your own, or delete them.

6. **Job submission:** the `--autojob` option calls `~/bin/autojob`. Replace this with your own submission command if needed.

---

## Expected folder layout

```
13_Batch_Template_Setup/
├─ references/
│   └─ setup_run_folders.sh   ← the script you are reading (may need minor adaptation)
└─ template_run_folder/        ← **user‑provided** folder that contains ONLY the placeholder files
    ├─ run.slurm               ← SLURM header with a placeholder job name
    ├─ samples.csv            ← generated in step 1.2 (copy here)
    ├─ BCFile.in               ← input file with placeholders:
    │   ├─ TEMP_PLACE
    │   ├─ TIME_PLACE
    │   ├─ STRESS_HOOP_PLACE
    │   └─ STRESS_RADIAL_PLACE
    └─ single_crystal_model.sx         ← single‑crystal texture file with placeholder RHOC_PLACE
```

> **Important** – The `template_run_folder` is **manually prepared by the user**. It should contain **only** the skeleton input files with the placeholder strings indicated above. No actual simulation data belongs in this folder; the script will later copy and populate it for each sample.

> The repository does **not** ship concrete input files. Users must create the placeholder files appropriate to their own simulation workflow and, if needed, edit `setup_run_folders.sh` to match any custom naming conventions.

---

## How to use the script (HPC‑focused)

1. **Prepare the template locally** – Create `template_run_folder` with the placeholder files described above.  This step is entirely manual; no code execution is required.
2. **Copy the whole `13_Batch_Template_Setup` directory** (or at least `template_run_folder` and the `references` sub‑folder) to the HPC filesystem where you have a scratch or work directory.
3. **Log in to the HPC node** and navigate to the copied directory.
4. **Run the script on the HPC** (the script itself changes to `/scratch` internally, which is typical on clusters):

```bash
cd $SCRATCH/13_Batch_Template_Setup   # example path on the HPC
bash references/setup_run_folders.sh          # normal mode – creates the run folders
bash references/setup_run_folders.sh --autojob   # also triggers the auto‑submission helper
```

> **Why run on the HPC?** The script expects to be executed on a compute node (or a login node with access to the scratch filesystem). It creates directories, copies potentially large input files, and may launch SLURM jobs; performing these steps locally would either fail (missing `/scratch`) or be inefficient.

---
 
## Local mode: (e.g., chemical-kinetics example)
The Cantera 0-D autoignition model (defined in
`../11_Generative_Model_Summary/references/src/gri_autoignition.py`) runs in
milliseconds-to-seconds per case, so the whole dataset is generated locally in
one pass. The script:

1. **Reads** the DOE `samples.csv` from step 1.2
   (`../12_Design_of_Experiment/output/samples.csv`, columns `TEMP,PRESSURE,PHI`).
2. **Integrates** one constant-pressure autoignition trajectory per row by
   calling `gri_autoignition.run_single_case(T0, P, phi)`.
3. **Assembles** all trajectories into the exact LAROMance Stage-2 layout
   (`{"meta_data": {...}, "data": {int_id: {inputs..., "U": {rates...}}}}`,
   every array of equal length) and writes it directly to
   `output/Cantera_kinetics_runs.pickle`.

### How to run (locally)

```bash
cd 1_Data_Generation/13_Batch_Template_Setup

# Quick wiring test on the first 20 samples
python generate_cantera_data.py --smoke

# Full run over the DOE csv, parallelised across 4 processes
python generate_cantera_data.py --nproc 4

# Explicit control
python generate_cantera_data.py \
    --samples ../12_Design_of_Experiment/output/samples.csv \
    --out output/Cantera_kinetics_runs.pickle \
    --limit 500 --nproc 4 --n_steps 200 --t_end 1.0
```

Requires the `cantera` package in the active environment
(`conda install -c conda-forge cantera` or `pip install cantera`).

### Resulting pickle contents

Per simulation ID:

* **Inputs** - `temperature`, `pressure`, `phi` (scalar controls, repeated per
  step), `t` (monotonic progress axis), and the feedback **state**: `T` plus the
  tracked species mole fractions (`CH4, O2, CO2, H2O, CO, OH, H, O`).
* **Outputs** (`U`) - the production **rates** `dT_dt` and `d<species>_dt`,
  which span many orders of magnitude (near-zero during induction, spiking at
  ignition), suiting the log10/symlog output scalings used downstream.

### Prepare the next stage

Copy `output/Cantera_kinetics_runs.pickle` into
`../../2_Data_Processing/references/` and run `load_and_plot_data.py`
(see `../../2_Data_Processing/CONTEXT.md`).

---