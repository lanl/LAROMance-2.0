#!/usr/bin/env python3
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National Nuclear
Security Administration. All rights in the program are reserved by Triad
National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others
acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare derivative works, distribute copies
to the public, perform publicly and display publicly, and to permit others
to do so.

==============================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

generate_cantera_data.py - local data generator for the chemical-kinetics
example of the LAROMance workflow.

Unlike the crystal-plasticity creep workflow (which fans each design-of-
experiment sample out to its own SLURM run folder on an HPC via
``setup_run_folders.sh``), the Cantera 0-D autoignition model is cheap enough
to run *locally*: a single reactor integration takes milliseconds to a second.

This script therefore COLLAPSES steps 1.2 -> 1.3 into one local pass:

    1. Read the design-of-experiment ``samples.csv`` produced by step 1.2
       (columns ``TEMP, PRESSURE, PHI``).
    2. For each row, call :func:`gri_autoignition.run_single_case` to integrate
       one autoignition trajectory with GRI-Mech 3.0.
    3. Assemble ALL trajectories into a single dictionary in the exact LAROMance
       Stage-2 layout and write it directly to ONE ``.pickle`` file.

No per-run folders, no SLURM, no HPC. The resulting pickle in ``output/`` is
ready to be copied into ``2_Data_Processing/references/``.

Usage
-----
    # Full run over the DOE csv (5000 samples by default)
    python generate_cantera_data.py

    # Quick wiring test on the first 20 samples
    python generate_cantera_data.py --smoke

    # Explicit control
    python generate_cantera_data.py \\
        --samples references/samples.csv \\
        --out output/Cantera_kinetics_runs.pickle \\
        --limit 500 --nproc 4 --n_steps 200
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# --- Make the step 1.1 chemistry model importable -------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_SRC = os.path.abspath(os.path.join(
    _THIS_DIR, "..", "11_Generative_Model_Summary", "references", "src"))
if _MODEL_SRC not in sys.path:
    sys.path.insert(0, _MODEL_SRC)

import gri_autoignition as model  # noqa: E402

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_SAMPLES = os.path.abspath(os.path.join(
    _THIS_DIR, "references", "samples.csv"))
DEFAULT_OUT = os.path.join(_THIS_DIR, "output", "Cantera_kinetics_runs.pickle")


def _run_one(task):
    """Worker: integrate one case. ``task`` = (idx, T0, P, phi, kwargs).

    Returns ``(idx, record)`` or ``(idx, None)`` if the integration failed, so
    a single bad sample never aborts the whole batch.
    """
    idx, T0, P, phi, kwargs = task
    try:
        rec = model.run_single_case(T0, P, phi, **kwargs)
        return idx, rec
    except Exception as exc:  # pragma: no cover - robustness guard
        sys.stderr.write(
            "  [warn] sample {} (T0={:.1f}, P={:.3e}, phi={:.3f}) failed: {}\n"
            .format(idx, T0, P, phi, exc))
        return idx, None


def generate(samples_csv, out_path, *, limit=None, nproc=1, n_steps=200,
             t_end=1.0, peak_refine=False, peak_species="CH4",
             peak_density=200):
    """Generate the kinetics dataset pickle from a DOE csv.

    Parameters
    ----------
    samples_csv : str
        Path to ``samples.csv`` with columns ``TEMP, PRESSURE, PHI``.
    out_path : str
        Output ``.pickle`` path.
    limit : int or None
        If given, only process the first ``limit`` rows (for smoke tests).
    nproc : int
        Number of worker processes (1 = serial).
    n_steps : int
        Output points on the backbone grid. With ``peak_refine`` the final
        trajectory length is larger (and may vary per case).
    t_end : float
        Integration end time [s] passed to the model.
    peak_refine : bool
        If True, densely resolve the ignition rise and drop via a two-pass
        peak-aware grid (see :func:`gri_autoignition.run_single_case`). This
        produces variable-length trajectories, which the LAROMance layout
        supports via the per-simulation ``sim_lengths`` metadata.
    peak_species : str
        Tracked species whose rate magnitude locates the ignition pulse when
        ``peak_refine`` is True. Default ``"CH4"``.
    peak_density : int
        Number of extra points inserted across the ignition window when
        ``peak_refine`` is True.
    """
    df = pd.read_csv(samples_csv)
    required = {"TEMP", "PRESSURE", "PHI"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "samples.csv is missing required columns: {}. Found: {}"
            .format(sorted(missing), list(df.columns)))
    if limit is not None:
        df = df.head(limit)

    kwargs = {"n_steps": n_steps, "t_end": t_end}
    if peak_refine:
        kwargs.update({
            "peak_refine": True,
            "peak_species": peak_species,
            "peak_density": peak_density,
        })
    tasks = [
        (i, float(r.TEMP), float(r.PRESSURE), float(r.PHI), kwargs)
        for i, r in enumerate(df.itertuples(index=False))
    ]

    print("Generating {} autoignition trajectories (GRI-Mech 3.0)".format(len(tasks)))
    print("  mechanism      : {}".format(model.MECHANISM))
    print("  tracked species: {}".format(", ".join(model.TRACKED_SPECIES)))
    print("  n_steps        : {}   t_end : {} s".format(n_steps, t_end))
    if peak_refine:
        print("  peak_refine    : ON (species={}, +{} points around pulse)"
              .format(peak_species, peak_density))
    print("  processes      : {}".format(nproc))

    t0 = time.time()
    results = {}
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(processes=nproc) as pool:
            for k, (idx, rec) in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=8)):
                if rec is not None:
                    results[idx] = rec
                if (k + 1) % 250 == 0 or (k + 1) == len(tasks):
                    print("  ... {}/{} done".format(k + 1, len(tasks)))
    else:
        for k, task in enumerate(tasks):
            idx, rec = _run_one(task)
            if rec is not None:
                results[idx] = rec
            if (k + 1) % 100 == 0 or (k + 1) == len(tasks):
                print("  ... {}/{} done".format(k + 1, len(tasks)))

    elapsed = time.time() - t0


    # --- Assemble into LAROMance Stage-2 layout ---------------------------
    # Re-key to contiguous integer IDs so downstream indexing is clean.
    data = {}
    sim_lengths = []
    for new_id, old_idx in enumerate(sorted(results.keys())):
        rec = results[old_idx]
        data[new_id] = rec
        sim_lengths.append(len(rec["t"]))
    failed = sorted(set(idx for idx, _, _, _, _ in tasks) - set(results.keys()))

    meta_data = {
        "num_increments": n_steps,
        "sim_lengths": sim_lengths,
        "job_number": list(range(len(data))),
        "empty_sims": failed,
        "mechanism": model.MECHANISM,
        "tracked_species": list(model.TRACKED_SPECIES),
        "reactor": "IdealGasConstPressureReactor (constant-pressure autoignition)",
        "doe_source": os.path.abspath(samples_csv),
    }
    DATA = {"meta_data": meta_data, "data": data}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(DATA, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Report -----------------------------------------------------------
    print("\nDone in {:.1f} s.".format(elapsed))
    print("  successful trajectories : {}".format(len(data)))
    print("  failed trajectories     : {}".format(len(failed)))
    if data:
        first = next(iter(data.values()))
        inputs = [k for k in first if k != "U"]
        print("  input variables         : {}".format(inputs))
        print("  output variables (U)     : {}".format(list(first["U"].keys())))
    print("  written to              : {}".format(os.path.abspath(out_path)))
    print("\nNext: copy this pickle into 2_Data_Processing/references/ and run")
    print("      load_and_plot_data.py (see 2_Data_Processing/CONTEXT.md).")
    return DATA


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default=DEFAULT_SAMPLES,
                   help="Path to the DOE samples.csv (default: step 1.2 output).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Output pickle path (default: output/Cantera_kinetics_runs.pickle).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N samples.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick wiring test: equivalent to --limit 20.")
    p.add_argument("--nproc", type=int, default=1,
                   help="Number of worker processes (default: 1 = serial).")
    p.add_argument("--n_steps", type=int, default=200,
                   help="Output time points per trajectory (array length).")
    p.add_argument("--t_end", type=float, default=1.0,
                   help="Integration end time [s].")
    args = p.parse_args()

    limit = 20 if args.smoke else args.limit
    out = args.out
    if args.smoke and out == DEFAULT_OUT:
        out = os.path.join(_THIS_DIR, "output", "Cantera_kinetics_runs_smoke.pickle")

    generate(args.samples, out, limit=limit, nproc=args.nproc, 
             peak_refine=True, peak_species="CH4", peak_density=200,      
             n_steps=args.n_steps, t_end=args.t_end)


if __name__ == "__main__":
    main()
