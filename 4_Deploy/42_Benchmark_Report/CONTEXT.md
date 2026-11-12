# 4.2 - Benchmark Report (Deploy)

This final step closes the pipeline by (1) letting you visualise the compiled
MOOSE benchmark output and (2) assembling a single, Color-coded **benchmark
report** that summarises every stage of the workflow.

---

## Part A - Generate the Benchmark Report (automated)

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
  the points added by augmentation (total · average per simulation, if used),
  and the input/output variables.
- **TRAINING** — R² per output (colored) and condition number.
- **TESTING** — relative-RMSE per output as mean · std (mean colored), and the
  testing wall-time as total · per increment (single-evaluation cost).
  When bootstrapping is enabled (stage 3.2), each output also shows a
  **bootstrap 95 % CI** row. This interval is obtained by repeatedly
  resampling the per-increment relative-RMSE values with replacement (1000
  times) and taking the 2.5 % / 97.5 % percentiles of the resulting
  distribution. The **mean** of the bootstrap distribution is the usual point
  estimate; the **CI** gives a non-parametric measure of uncertainty around
  that estimate. A narrow CI indicates the observed error is stable across
  resamples; a wide CI signals higher variability in the test-set errors.
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
python generate_benchmark_report.py
```

Congratulations! You have completed a surrogate model build!

*There is no constitutive law. Bend the code.*
