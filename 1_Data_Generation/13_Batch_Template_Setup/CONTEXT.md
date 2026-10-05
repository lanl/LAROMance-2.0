# 1.3 - Batch Template Setup

## Purpose

`setup_run_folders.sh` automates the creation of **run folders** that will be submitted to an HPC scheduler (SLURM). Each folder corresponds to a single line (one sample) in `samples.csv`, which is the design‑of‑experiments file generated in the *12_Design_of_Experiments* stage.

The script:

1. **Reads** the `samples.csv` created in step 1.2 - Design of Experiment. The file should be copied to the template input folder.
2. **Creates** a new directory (`run_1`, `run_2`, …).
3. **Copies** only the files that are needed for a simulation (`*.slurm`, `*.in`, `*.dat`, `*.sx`) from a template folder (`template_run_folder`).
4. **Replaces** placeholder strings inside the copied files with the concrete values read from the CSV line.
5. (Optionally) **launches** an AutoJob helper to submit the whole batch at once (`--autojob`).

Because the script works purely with *text substitution* (`sed`), it is agnostic to the actual physics solver – any code that can be driven by a set of input files and a SLURM batch script will work as long as the placeholders are present.

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

## `samples.csv` format

| Column (order) | Placeholder used in the script | Meaning |
|----------------|----------------------------------|---------|
| `TEMP`         | `TEMP_PLACE`                     | Temperature (°C or K) for the BC file |
| `VMJ2`         | `STRESS_HOOP_PLACE` (via `stress_rate_hoop`) | Hoop stress magnitude (used to compute a time ramp) |
| `RHOC`         | `RHOC_PLACE`                      | Density (or other material constant) for the *.sx* file |

The first line of `samples.csv` must be a header (it is ignored by the script). Example:

```csv
TEMP,VMJ2,RHOC
800,150,19.25
850,175,19.30
900,200,19.35
```

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
