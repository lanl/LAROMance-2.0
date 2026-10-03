#!/usr/bin/env python3
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

doe.py – Design of Experiments helper

This script generates a Latin Hypercube Sampling (LHS) set based on user‑provided
parameter ranges.  It wraps the ``utils.sm_jobsubmit.generate_LHS_set_modular``
function, which performs the actual sampling and CSV export.

Usage (as described in ``CONTEXT.md``):

    python doe.py TEMP 600 1023 VMJ2 0 250 EVM 0 0.2 RHOC 1e11 1e13 RHOW 1e11 1e13 \
        --number 1000

The positional arguments are interpreted as *triples* ``<label> <min> <max>``.
If no triples are supplied, the script falls back to an interactive prompt where
the user can enter label‑min‑max groups one by one.

The generated samples are written to ``output/samples.csv`` inside this stage's
folder.

"""

import argparse
import os
import sys
import numpy as np

# Add the project root to ``sys.path`` so that ``utils`` can be imported.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.sm_jobsubmit import generate_LHS_set_modular  # noqa: E402  pylint: disable=import-error


def _parse_triples(triple_list):
    """Convert a flat list ``[label, min, max, ...]`` into a dict.

    Parameters
    ----------
    triple_list: list[str]
        A list whose length is a multiple of three.

    Returns
    -------
    dict
        Mapping ``label -> {"min": float, "max": float}``.
    """
    if len(triple_list) % 3 != 0:
        raise ValueError("Parameter arguments must be provided in label min max triples.")
    parameters = {}
    for i in range(0, len(triple_list), 3):
        label = triple_list[i]
        try:
            lo = float(triple_list[i + 1])
            hi = float(triple_list[i + 2])
        except ValueError as exc:
            raise ValueError(f"Non‑numeric bounds for parameter '{label}'.") from exc
        parameters[label] = {"min": lo, "max": hi}
    return parameters


def _prompt_parameters():
    """Interactively ask the user for label‑min‑max entries.

    Returns
    -------
    dict
        Same structure as produced by ``_parse_triples``.
    """
    print("Enter parameter triples (label min max). Press ENTER on an empty label to finish.")
    params = {}
    while True:
        label = input("  label: ").strip()
        if not label:
            break
        lo_str = input(f"  min for {label}: ").strip()
        hi_str = input(f"  max for {label}: ").strip()
        try:
            lo = float(lo_str)
            hi = float(hi_str)
        except ValueError as exc:
            print(f"  Invalid numeric value – please retry for '{label}'.")
            continue
        params[label] = {"min": lo, "max": hi}
    if not params:
        raise RuntimeError("No parameters were provided.")
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Generate LHS design of experiments samples.")
    # Positional arguments are collected as a flat list; they will be interpreted as triples.
    parser.add_argument(
        "triples",
        nargs="*",
        help="Parameter triples: <label> <min> <max> ..."
    )
    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="Number of LHS samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--log10",
        nargs='*',
        default=[],
        help="Space‑separated list of parameter labels to sample in log10 space."
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Enable a 3‑D scatter plot of the generated LHS samples."
    )
    parser.add_argument(
        "--axes",
        nargs=3,
        default=None,
        help="Three parameter labels to use for the 3‑D visualization (order matters)."
    )
    args = parser.parse_args()

    # Determine parameters either from command line or via interactive prompts.
    if args.triples:
        parameters = _parse_triples(args.triples)
    else:
        parameters = _prompt_parameters()

    # Determine which parameters, if any, should be sampled in log10 space.
    log10_params = set(args.log10)
    if not log10_params:
        # If not provided via CLI, prompt the user interactively.
        try:
            resp = input(
                "Enter space‑separated parameter labels to sample in log10 space (or press ENTER for none): "
            ).strip()
            if resp:
                log10_params = set(resp.split())
        except EOFError:
            # Non‑interactive environment – keep empty set.
            log10_params = set()

    # Prepare output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Generate the LHS set.  We suppress the visualisation for automated runs.
    # Prepare a temporary parameters dict for LHS generation. For log‑scaled
    # parameters we pass the log10 of the bounds; the transformation back to
    # linear space is applied after the CSV is created.
    from collections import OrderedDict
    sampling_params = OrderedDict()
    for name, bounds in parameters.items():
        if name in log10_params:
            # Guard against non‑positive bounds for log10.
            if bounds['min'] <= 0 or bounds['max'] <= 0:
                raise ValueError(f"Log10 sampling requires positive bounds for '{name}'.")
            sampling_params[name] = {
                "min": np.log10(bounds['min']),
                "max": np.log10(bounds['max'])
            }
        else:
            sampling_params[name] = bounds

    # If visualization axes are requested, reorder the OrderedDict so that
    # the chosen axes appear first (the LHS function will plot the first three).
    if args.visualize and args.axes:
        chosen = []
        for ax in args.axes:
            if ax not in sampling_params:
                raise ValueError(f"Axis '{ax}' not found among parameters.")
            chosen.append((ax, sampling_params[ax]))
        # Append remaining parameters preserving original order.
        remaining = [(k, v) for k, v in sampling_params.items() if k not in args.axes]
        sampling_params = OrderedDict(chosen + remaining)

    csv_path = generate_LHS_set_modular(
        parameters=sampling_params,
        num_samples=args.number,
        output_file="samples.csv",
        visualize=args.visualize,
        path_to_template=output_dir,
    )

    # Post‑process CSV to convert log10‑sampled columns back to linear scale.
    if log10_params:
        import pandas as pd
        df = pd.read_csv(csv_path)
        for name in log10_params:
            if name in df.columns:
                df[name] = 10 ** df[name]
        df.to_csv(csv_path, index=False)

    print(f"LHS samples written to {csv_path}")


if __name__ == "__main__":
    main()
