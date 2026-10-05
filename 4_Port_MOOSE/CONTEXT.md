# 4 - Port Model to MOOSE (Finite‑Element Framework)

The **4_Port_MOOSE** stage bridges the surrogate model generated in the previous
steps with the MOOSE (and BISON) simulation environment.  It is split into two
sub‑steps:

1. **41_Export_Model** – Takes a trained surrogate (`SM.pickle`) and produces the
   C++ source files (`*.C` and `*.h`) that implement the surrogate as a MOOSE
   material.  It also creates an extrapolated mesh, a verification CSV table,
   and visualisation assets.
2. **42_Analyze_Benchmark_Test** – Uses the exported MOOSE material to run a set
   of benchmark simulations, reads the resulting `benchmarks.csv`, and visualises
   the comparison between the high‑fidelity data and the surrogate predictions.

Both sub‑steps are documented in their own `CONTEXT.md` files located in the
corresponding directories.

---