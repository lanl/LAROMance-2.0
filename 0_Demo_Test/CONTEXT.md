# 0 – Demo / Test

## Purpose

This stage provides a single, self-contained script – `run_test_demo.py` – that
exercises the **entire LAROMance pipeline (stages 1 → 5) end-to-end** against the
synthetic example datasets shipped in `../examples/`. It exists so that a new
user (or a CI job) can confirm, in one command, that every stage of the workflow
is wired up correctly on their machine *before* investing time in their own data.

Unlike stages 1–5, this stage **never requires the training GUI** (PySide6). The
training step is reproduced *headlessly* by calling `utils.sm_build.sm_train`
directly with a mesh/argument bundle derived from the example training-module
config JSON (`../examples/Viscoplasicity_Creep/Example_Training_Module_Config.json`).

The script has two mutually-exclusive modes:

| Mode     | Flag      | Behaviour                                                                             |
|----------|-----------|---------------------------------------------------------------------------------------|
| **Demo** | `--demo`  | Interactive walkthrough. Prints the equivalent terminal command for each stage, runs it against the example data, narrates the WHY behind each step, summarises the artifacts produced, and pauses for `Enter` between stages. ALWAYS keeps the generated artifacts for exploration. |
| **Test** | `--test`  | Non-interactive smoke test. Runs the same sequence, asserts the expected artifacts exist, prints a concise PASS summary, cleans up, and exits `0` (all passed) or non-zero (any failure). Add `--keep` to a `--test` run for adeeper check that retains the artifacts and prints the full per-stage
`PASS`/`FAIL` table so you can inspect the files by hand.|

## Example data

The harness uses the **Viscoplasticity / Creep** example
(`../examples/Viscoplasicity_Creep/`) as its reference dataset. It runs the
**full creep dataset** through the real workflow, using the **same training
configuration** as the shipped
`examples/Viscoplasicity_Creep/Example_Training_Module_Config.json` (same mesh
element counts, input/output maps and ROI). Data augmentation (anchor points +
concatenation) is applied in stage 2 exactly as in the documented creep
workflow, so the surrogate mesh has no empty elements and the regression is
well-conditioned (R² ≈ 1.0, finite nodal values). The goal is both a *wiring*
check – that each stage runs and emits the right files – and a basic
*correctness* check that the surrogate fits the example data. All generated
artifacts are written under a temporary working directory (`./_work/`) inside
this stage folder and are cleaned up on success.

## Environment

- Python ≥ 3.10 with the core dependencies installed (`numpy`, `scipy`,
  `scikit-learn`, `matplotlib`, `pandas`, `rtree`, `tqdm`). See the root
  `README.md` → *Installation*.
- **No GUI toolkit is required.** The script sets `QT_QPA_PLATFORM=offscreen`
  and `MPLBACKEND=Agg` internally so it runs on headless servers / CI.
- Optional: if you drive this repo from **VS Code** with an AI coding assistant
  (e.g. Continue.dev or Claude Code), this stage is a convenient "does it all
  still work?" target to ask the agent to run after making changes.

## How to run
### Quick Start

```bash
# From the repository root
cd 0_Demo_Test
python run_test_demo.py --demo
```

That single command runs the entire LAROMance pipeline (stages 1 → 5) against
the shipped Viscoplasticity/Creep example, narrating each step and keeping all
generated artifacts so you can explore them afterwards.

### Call Flags for Demo and Testing

```bash
# From the repository root
cd 0_Demo_Test

# Interactive guided tour of the whole pipeline against the example data.
# Pauses between stages and ALWAYS keeps the generated artifacts.
python run_test_demo.py --demo

# Non-interactive smoke test: concise PASS summary + exit code, then cleans up.
python run_test_demo.py --test

# Deeper test: keep the artifacts and print the full per-stage PASS/FAIL table.
python run_test_demo.py --test --keep

# Show all options
python run_test_demo.py --help
```

## What each stage check does

1. **1 – Data Generation**: no high-fidelity model ships with LAROMance, so this
   stage is represented by confirming the example dataset exists and reporting
   how it would be produced (see `../1_Data_Generation/CONTEXT.md`).
2. **2 – Data Processing**: runs `2_Data_Processing/load_and_plot_data.py` on the
   full example pickle with augmentation (`--concat --anchor_points ...`) to
   produce curated train/test pickles + diagnostic plots.
3. **3 – Surrogate Modeling**:
   - *3.1 Training* – headless `sm_train` call (GUI bypassed) → `SM.pickle`.
   - *3.2 Testing* – runs `32_Surrogate_Testing/test_surrogate.py` (coupled
     oracle mode) → validation plots + `Testing_result_*.pickle`.
4. **4 – Port MOOSE**: runs `41_Export_Model/port_surrogate_to_moose.py` →
   MOOSE `.C`/`.h` source, `benchmarks.csv`, mesh visualization.
5. **5 – Clean Pipeline**: runs `5_Clean_Pipeline/clean_pipeline.sh --dry-run`
   to confirm the clean-up tooling is present and reports what *would* be
   removed (the harness never deletes stage 1–5 artifacts itself).
