# 4.2 - Benchmark Report (Deploy)

This final step closes the pipeline by (1) letting you visualise the compiled
MOOSE benchmark output and (2) assembling a single, Color-coded **benchmark
report** that summarises every stage of the workflow.

---

## Part A - Generate the Benchmark Report (automated)

### What it is

Throughout stages **2 → 4** each script already prints its stats and metrics to
the console. At those same locations the scripts now *also* append a small,
per-stage JSON file via the shared helper `utils/sm_report.py`
(`log_stage_metrics`). Each stage writes its own file into its **`output/`**
folder:

| Stage | Written by | File (in that stage's `output/`) |
|-------|------------|----------------------------------|
| 2 Data Processing | `load_and_plot_data.py` | `benchmark_2_data_processing.json` |
| 3.1 Training | `utils/sm_build.py` (`sm_train` / `mlp_train`) | `benchmark_3_1_training.json` |
| 3.2 Testing | `test_surrogate.py` + `sm_build.sm_test` | `benchmark_3_2_testing.json` |
| 4.1 Export MOOSE | `port_surrogate_to_moose.py` | `benchmark_4_1_port_moose.json` |

The script `generate_benchmark_report.py` **globs the whole repository** for
these `benchmark_*.json` files, merges them, and writes a formatted Markdown
report to `./output/benchmark_report.md`.

### What the report contains

The report is a **single vertical two-column table** (`Metric | Value`), with
bold full-width rows acting as group dividers so the whole surrogate is
overviewable at once. Colored metrics flag quality inline. The groups are:

- **DATA** — train/test simulations, data points, increments/sim (avg · max),
  input/output variables, and (if used) the points added by augmentation
  (total · average per simulation).
- **TRAINING** — R² per output (colored), condition number, and the
  interp-matrix & regression wall-times.
- **TESTING** — relative-RMSE per output as mean · median · max (mean colored),
  total testing wall-time, and wall-time **per increment** (single-evaluation cost).
- **MODEL & DEPLOYMENT** — architecture (`grid`/`mlp`/`pointnet`/`moe`) and its
  settings (grid: element numbers/type & mesh counts; mlp: epochs/hidden/params;
  moe: experts/top-k), plus the ported MOOSE `.C`/`.h`/`.pt` and verification files.

### Color coding

Key metrics are rendered as **Colored text** using inline HTML
(`<span style="color:…">`), which renders in the VS Code Markdown preview and
most Markdown viewers. The thresholds match those already used on the console:

- **R²**: <span style="color:#2ca02c">green ≥ 0.99</span> · <span style="color:#e6a700">amber ≥ 0.95</span> · <span style="color:#d62728">red &lt; 0.95</span>
- **Relative-RMSE**: <span style="color:#2ca02c">green ≤ 0.05</span> · <span style="color:#e6a700">amber ≤ 0.10</span> · <span style="color:#d62728">red &gt; 0.10</span>

The thresholds live in `utils/sm_report.py` (`R2_GOOD`, `R2_OK`,
`RELRMSE_GOOD`, `RELRMSE_OK`) — adjust them there if your tolerances differ.

### How to run

```bash
cd 4_Deploy/42_Benchmark_Report
conda activate laromance          # if applicable
python generate_benchmark_report.py
```

Useful flags:

```bash
# Point at a different repository root to scan
python generate_benchmark_report.py --repo-root /path/to/LAROMance_process

# Write the report somewhere else
python generate_benchmark_report.py --output ./output/my_report.md
```

The generator is **fail-safe**: if a stage's JSON is missing it simply notes
that section as "not found" rather than erroring, so you can produce a partial
report at any point in the workflow.

> **Note:** the per-stage JSON files live in each stage's `output/` folder, so a
> full run of `5_Clean_Pipeline/clean_pipeline.sh` removes them along with the
> other generated artefacts. Generate (or copy out) the report before cleaning.

---

## Part B - Visualise the MOOSE benchmark output (manual)

After the surrogate has been exported to a MOOSE material (step 4.1) the user
runs the verification simulations and then examines the results. The utility
functions in `utils/sm_analyze.py` can be used to visualise the benchmark
output. A typical call looks like:

```python
import os
from utils import sm_analyze as sma

output_directory = "./output"
filepath = os.path.join(output_directory, "verification_cube_out.csv")

# Produce a 3-D plot of the rate difference (example variable name)
sma.moose_verification(
    filepath=filepath,
    x='vmJ2',
    y='temperature',
    z='rhom_rate_diff'
)
```

Replace the column names (`vmJ2`, `temperature`, `rhom_rate_diff`) with the
appropriate fields from your CSV. The function will generate the requested
visualisation in the `output` folder.

---

## Congratulations!
You have completed the workflow and should now have a verified surrogate plus a
full benchmark report.

*There is no constitutive law. Bend the code.*

