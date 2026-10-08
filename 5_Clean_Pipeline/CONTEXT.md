# 5 – Clean Pipeline (Port MOOSE)

## Purpose

After you have completed the full surrogate‑building workflow (up to **4.2 Analyze Benchmark Test**) you may want to **archive** the entire repository – keeping the results, configuration files and documentation – and then **restart** the pipeline from scratch on new data.  The *Clean Pipeline* step removes all automatically generated artefacts (`output` and `references` folders) while **preserving**:

* `utils/` and `examples/` directories (they contain reusable scripts and example data)
* documentation files (`CONTEXT.md`, `README.md`, `.gitignore`, `.gitkeep`)
* In `4_Port_MOOSE/41_Export_Model/references` only `*.pickle` files are deleted; `*.i` files and the `base` sub‑directory are kept because they are required for later MOOSE runs.

The step is implemented as a small Bash script (`clean_pipeline.sh`).  It can be run **manually** or **automatically** as the final optional step of stage **4.2 Analyze Benchmark Test**.

## How to run

```bash
# From the repository root
cd 5_Clean_Pipeline
./clean_pipeline.sh            # performs the clean‑up
./clean_pipeline.sh --dry-run  # shows what would be removed without deleting
```

The script is safe to execute on a **copied** version of the repository (e.g., after you `cp -r LAROMance_process LAROMance_process_archive`).  The original data remain untouched because the script only operates inside the stage directories it knows about.

## Optional integration with stage 4.2

If you prefer a completely automated workflow, add the following line to the end of the Python/command sequence that finishes **4.2 Analyze Benchmark Test**:

```bash
# After user confirms the surrogate is approved
../5_Clean_Pipeline/clean_pipeline.sh
```

You can also expose a flag in the calling script, e.g. `--clean-after`, to let the user decide whether to invoke the clean‑up automatically.

---

**Note:** This step does **not** delete any source code, configuration files, or the example datasets. It merely resets the generated artefacts, allowing a fresh run of the pipeline without re‑creating the repository structure.
