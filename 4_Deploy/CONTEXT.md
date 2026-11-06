# 4 - Deploy (Export to MOOSE & Benchmark Report)

The **4_Deploy** stage bridges the surrogate model generated in the previous
steps with the MOOSE (and BISON) simulation environment and produces a final
benchmark report.  It is split into two
sub‑steps:

1. **41_Export_MOOSE** – Takes a trained surrogate (`SM.pickle`) and produces the
   C++ source files (`*.C` and `*.h`) that implement the surrogate as a MOOSE
   material.  It also creates an extrapolated mesh, a verification CSV table,
   and visualisation assets.
2. **42_Benchmark_Report** – Globs the per-stage `benchmark_*.json` artefacts
   written throughout the pipeline and assembles a single, colour-coded Markdown
   benchmark report.  It can also run the exported MOOSE material through a set of
   benchmark simulations and visualise the comparison between the high‑fidelity
   data and the surrogate predictions.

Both sub‑steps are documented in their own `CONTEXT.md` files located in the
corresponding directories.

---