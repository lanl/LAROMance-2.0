#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Data loading script for the LAROMance process.

It demonstrates how to:
- Load a raw pickle dataset.
- (Optionally) rename input / output labels.
- (Optionally) de‑concatenate it.
- Curate / down‑sample the data.
- Store the curated dataset back to disk.
- Plot the data in scatter plots and time history traces

"""

import os, copy, pickle, argparse, sys, re
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from utils import sm_data as smd
from utils import sm_plot as smp

# Directory containing the reference pickle files (relative to this script)
data_path = os.path.join(os.path.dirname(__file__), "references")

# Keys at the top level of a simulation that are containers, not input variables
PROTECTED_KEYS = ("U", "x")


# ======================================================================
# Helper functions
# ======================================================================
def merge_datasets(datasets):
    """Merge a list of loaded pickle datasets into one DATA dict."""
    if len(datasets) == 1:
        return datasets[0]

    merged_data = {}
    merged_sim_lengths = []
    id_offset = 0
    for d in datasets:
        data_dict = d["data"]
        meta = d.get("meta_data", {})
        sim_lengths = meta.get("sim_lengths", [len(v["t"]) for v in data_dict.values()])
        for old_id, sim in data_dict.items():
            merged_data[old_id + id_offset] = sim
        merged_sim_lengths.extend(sim_lengths)
        if merged_data:
            id_offset = max(merged_data.keys()) + 1
    merged_meta = {
        "sim_lengths": merged_sim_lengths,
        "job_number": list(range(len(merged_data))),
    }
    return {"data": merged_data, "meta_data": merged_meta}


def get_variable_names(data_dict):
    """Return (input_vars, output_vars) based on the first simulation."""
    first_sim = next(iter(data_dict.values()))
    input_vars = [k for k in first_sim.keys() if k not in PROTECTED_KEYS]
    output_vars = list(first_sim.get("U", {}).keys())
    return input_vars, output_vars


def print_summary(data_dict):
    """Print the dataset summary (number of sims, input and output variables)."""
    input_vars, output_vars = get_variable_names(data_dict)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Number of simulations : {len(data_dict)}")
    print(f"Input variables       : {input_vars}")
    print(f"Output variables (U)  : {output_vars}")
    print("=" * 60 + "\n")


def parse_rename_string(rename_str):
    """
    Parse "old:new,U[old]:new" into (input_map, output_map).

    Plain names refer to inputs (top-level keys).
    Names written as U[name] refer to outputs (nested 'U' dict).
    """
    input_map, output_map = {}, {}
    for item in rename_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            print(f"  [rename] Skipping '{item}': expected format old:new")
            continue
        old, new = (p.strip() for p in item.split(":", 1))
        if not old or not new:
            print(f"  [rename] Skipping '{item}': empty name")
            continue
        match = re.fullmatch(r"U\[(.+)\]", old)
        if match:
            output_map[match.group(1)] = new
        else:
            input_map[old] = new
    return input_map, output_map


def _validate_map(mapping, existing, kind):
    """Drop invalid entries (missing, protected, colliding) and warn."""
    valid = {}
    for old, new in mapping.items():
        if old not in existing:
            print(f"  [rename] {kind} '{old}' not found. Skipping.")
        elif kind == "input" and (old in PROTECTED_KEYS or new in PROTECTED_KEYS):
            print(f"  [rename] '{old}' -> '{new}' involves a protected key {PROTECTED_KEYS}. Skipping.")
        elif new in existing and new not in mapping:
            print(f"  [rename] {kind} '{old}' -> '{new}': '{new}' already exists. Skipping.")
        else:
            valid[old] = new
    # Guard against two olds mapping to the same new name
    if len(set(valid.values())) != len(valid):
        print(f"  [rename] Duplicate target names among {kind} renames. Skipping all {kind} renames.")
        return {}
    return valid


def rename_labels(DATA, rename_str=None):
    """
    Rename input labels (top-level keys) and output labels (keys of the
    nested 'U' dict) in every simulation of DATA, in place.

    If rename_str is None the user is prompted. Press Enter to skip.
    Format: "old:new,U[old_out]:new_out"
    Returns True if any label was renamed.
    """
    data_dict = DATA["data"]
    input_vars, output_vars = get_variable_names(data_dict)

    if rename_str is None:
        print("Rename labels (optional).")
        print("  Inputs : old:new          (e.g., vmJ2:stress)")
        print("  Outputs: U[old]:new       (e.g., U[drhoc_dt]:rhoc)")
        print(f"  Current inputs : {input_vars}")
        print(f"  Current outputs: {output_vars}")
        rename_str = input("Enter comma-separated renames or press Enter to skip: ").strip()

    if not rename_str:
        return False

    input_map, output_map = parse_rename_string(rename_str)
    input_map = _validate_map(input_map, input_vars, "input")
    output_map = _validate_map(output_map, output_vars, "output")

    if not input_map and not output_map:
        print("  [rename] Nothing to rename.")
        return False

    for sim in data_dict.values():
        if input_map:
            items = list(sim.items())          # rebuild to preserve key order
            sim.clear()
            sim.update({input_map.get(k, k): v for k, v in items})
        if output_map and isinstance(sim.get("U"), dict):
            u_items = list(sim["U"].items())
            sim["U"].clear()
            sim["U"].update({output_map.get(k, k): v for k, v in u_items})

    for old, new in input_map.items():
        print(f"  [rename] input  '{old}' -> '{new}'")
    for old, new in output_map.items():
        print(f"  [rename] output 'U[{old}]' -> 'U[{new}]'")
    return True


def clean_keys(raw):
    """Strip whitespace and non-printable characters from a comma-separated key string."""
    keys = [k.strip() for k in raw.split(",")]
    return [''.join(ch for ch in k if ch.isprintable()) for k in keys]


def plot_all(data_dict, input_keys, output_keys, trace_input_keys, filename,
             ylog, xlog, color='green', scatter=True):
    """Produce the scatter (plotdata) and trace (plottraces) plots for a dataset."""
    if scatter:
        smp.plotdata(data_dict, input_keys=input_keys, output_keys=output_keys,
                     simulation_recording_interval=1, inc=-1, cmap='magma',
                     storepath="./output", filename=filename, dark=True)
    smp.plottraces(data_dict, input_keys=trace_input_keys, output_keys=output_keys,
                   ylog=ylog, xlog=xlog,
                   storepath="./output", filename=filename,
                   color=color, dark=True)


def concat_and_histogram(TRAIN_DATA):
    """Concatenate the training data, check consistency and plot the histogram."""
    TRAIN_DATA = smd.concatenate_data(TRAIN_DATA, remove_initial_incs=0)
    smd.check_data_consistency(TRAIN_DATA['data'])
    smp.datahist(
        TRAIN_DATA['data'],
        4,
        storepath="./output",
        filename="physical",
        input_label_keys=[
            r"$\varepsilon_p$",
            r"$\sigma_\mathrm{VM}$",
            "T",
            r"$\rho_\mathrm{cell}$",
        ],
    )
    return TRAIN_DATA


# ======================================================================
# Parse command‑line arguments
# ======================================================================
parser = argparse.ArgumentParser(description="\nLoad and curate dataset with optional down‑sampling.")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--max_length",
    type=int,
    help="Maximum number of timesteps per simulation after down‑sampling.",
)
group.add_argument(
    "--factor",
    type=float,
    help="Down‑sampling factor (e.g., 1.3 reduces points by ~30%%).",
)
parser.add_argument(
    "--split_data",
    type=float,
    default=None,
    help="Percentage (0‑100) of curated simulations to use for concatenated training. "
         "The remainder will be kept as separate testing data."
)
parser.add_argument(
    "--input_keys",
    type=str,
    default=None,
    help='Comma-separated input variable keys for plots (e.g., "vmJ2,temperature").'
)
parser.add_argument(
    "--output_keys",
    type=str,
    default=None,
    help='Comma-separated output variable keys for plots (e.g., "rhoc,U[evm]").'
)
parser.add_argument(
    "--rename",
    type=str,
    default=None,
    help='Comma-separated label renames "old:new". Inputs are plain names; outputs '
         '(nested U dict) use U[old]:new. Example: "vmJ2:stress,U[rhoc]:drhoc_dt". '
         'Subsequent key arguments must use the NEW names. If omitted, you are prompted.'
)
parser.add_argument('--anchor_points',
                    type=str,
                    default=None,
                    help='Comma-separated list of anchor points for optional data augmentation.')
parser.add_argument('--correct_rhoc',
                    action='store_true',
                    help='Apply Karl‑specific correction to U[rhoc] rates.')
parser.add_argument('--downsample_key',
                    type=str,
                    default=None,
                    help="Per-simulation input array used as the monotonic axis for down‑sampling. "
                    "Defaults to 'evm' (strain) for creep datasets, auto-falling back to 't' "
                    "(time) when 'evm' is absent, e.g. for the Cantera kinetics example.")
parser.add_argument('--concat',
                    action='store_true',
                    help='If set, concatenate the curated/augmented dataset and generate the histogram plot.')
parser.add_argument('--pickle_files',
                    type=str,
                    default=None,
                    help='Comma‑separated list of pickle files (relative to the references folder) to load and merge.')
parser.add_argument('--ylog',
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Use logarithmic scale on the y‑axis of the trace plots. '
                    'Use --no-ylog to disable (default: enabled).')
parser.add_argument('--xlog',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Use logarithmic scale on the x‑axis of the trace plots. '
                    'Use --xlog to enable (default: disabled).')
parser.add_argument('--x_trim',
                    type=str,
                    default=None,
                    help='Supply x-variable to trim arrays by (max value --x_max).')
parser.add_argument('--x_max',
                    type=float,
                    default=None,
                    help='Max value in --x_trim var to trim arrays by.')
parser.add_argument('--y_filter',
                    type=str,
                    default=None,
                    help='Supply y-variable to filter data by (max value --y_max).')
parser.add_argument('--y_max',
                    type=float,
                    default=None,
                    help='Max value in --y_filter var used to decide removal of simulation data.')
args = parser.parse_args()

# ----------------------------------------------------------------------
# Determine which pickle files to load
# ----------------------------------------------------------------------
if args.pickle_files:
    file_list = [fn.strip() for fn in args.pickle_files.split(',') if fn.strip()]
else:
    # No files provided via CLI → open an interactive file picker
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                      # hide the root window
        root.attributes('-topmost', True)    # bring dialog to front

        selected = filedialog.askopenfilenames(
            title="Select one or more pickle files",
            initialdir=data_path,
            filetypes=[("Pickle files", "*.pickle *.pkl"), ("All files", "*")]
        )
        root.destroy()

        if selected:
            file_list = [os.path.basename(p) for p in selected]
        else:
            print("No files selected. Falling back to default 'data.pickle'.")
            file_list = ['data.pickle']
    except Exception as e:
        print(f"Could not open file dialog ({e}). Falling back to default 'data.pickle'.")
        file_list = ['data.pickle']

datasets = []
for fn in file_list:
    full_path = os.path.join(data_path, fn)
    print(f"Loading data from {full_path} ...")
    with open(full_path, "rb") as f:
        datasets.append(pickle.load(f))

DATA = merge_datasets(datasets)
data = DATA["data"]
meta_data = DATA["meta_data"]

# Base name for output files derived from the loaded pickle filenames
base_filename = "_".join([os.path.splitext(os.path.basename(fn))[0] for fn in file_list])

# ----------------------------------------------------------------------
# Optional correction (uses original label names, so it runs before renaming)
# ----------------------------------------------------------------------
if args.correct_rhoc:
    print("\nCorrecting output data in U[rhoc] to be rates. Specific to some legacy datasets.")
    for sim in data.keys():
        data[sim]['U']['rhoc'] = np.diff(data[sim]['rhoc']) / data[sim]['dt']
        data[sim]['U']['rhoc'] = np.concatenate(
            ([data[sim]['U']['rhoc'][0]], data[sim]['U']['rhoc'])
        )

# ----------------------------------------------------------------------
# Dataset summary, then optional label renaming
# ----------------------------------------------------------------------
print_summary(data)
print(f"Loaded {len(data)} simulations")
avg_len = sum(meta_data['sim_lengths']) / len(meta_data['sim_lengths'])
print(f"Average simulation length: {avg_len:.1f}\n")

if rename_labels(DATA, args.rename):
    print("\nLabels after renaming:")
    print_summary(data)

# ----------------------------------------------------------------------
# Interactive prompts for optional flags
# ----------------------------------------------------------------------
if not args.input_keys:
    inp = input("\nEnter input variable keys for plots (comma-separated) or press Enter for defaults: ").strip()
    if inp:
        args.input_keys = inp
if not args.output_keys:
    outp = input("\nEnter output variable keys for plots (comma-separated) or press Enter for defaults: ").strip()
    if outp:
        args.output_keys = outp

# If neither max length nor factor was given, ask for max length first.
# If the user skips (presses Enter), then ask for a down‑sampling factor.
if args.max_length is None and args.factor is None:
    downsample_max_length = None
    downsample_factor = None
    max_input = input("\nEnter maximum length for down‑sampling (or press Enter to set a factor): ").strip()
    if max_input:
        try:
            downsample_max_length = int(max_input)
        except ValueError:
            print("Invalid max length input, proceeding to factor prompt.")
            max_input = ""
    if not max_input:
        factor_input = input("Enter down‑sampling factor (e.g., 1.5) or press Enter to skip: ").strip()
        if factor_input:
            try:
                downsample_factor = float(factor_input)
            except ValueError:
                print("Invalid factor input, proceeding without down‑sampling.")
else:
    downsample_max_length = args.max_length
    downsample_factor = args.factor

# Prompt for concatenation if not supplied via CLI
if not args.concat:
    concat_input = input("\nCONCATENATE for Training? (True/False) ").strip().lower()
    args.concat = concat_input in ("true", "t", "yes", "y", "1")

# ----------------------------------------------------------------------
# Curate / down‑sample the dataset
# ----------------------------------------------------------------------
first_key = next(iter(data))
print("\nCurating dataset (down‑sampling)...")

# Decide which per-step array defines the down-sampling axis. Creep datasets use
# 'evm' (strain); the Cantera kinetics example has no 'evm' and uses 't' (time).
if args.downsample_key:
    downsample_key = args.downsample_key
elif 'evm' in data[first_key]:
    downsample_key = 'evm'
else:
    downsample_key = 't'
print(f"  down‑sampling axis: '{downsample_key}'")

DATA_new = smd.curate_dataset(
    copy.deepcopy(DATA),
    max_input_for_trim_key=args.x_trim,
    max_input_for_trim_value=args.x_max,
    max_output_for_sim_removal_key=args.y_filter,
    max_output_for_sim_removal_value=args.y_max,
    downsample_max_length=downsample_max_length,
    downsample_factor=downsample_factor,
    downsample_key=downsample_key,
)

# ----------------------------------------------------------------------
# Split data into training (concatenated) and testing (unconcatenated) subsets
# ----------------------------------------------------------------------
if args.split_data is not None:
    if not (0.0 <= args.split_data <= 100.0):
        raise ValueError("split_data must be between 0 and 100")
    split_frac = args.split_data / 100.0
    all_keys = list(DATA_new['data'].keys())
    n_total = len(all_keys)
    n_train = int(n_total * split_frac)
    train_keys = set(all_keys[:n_train])
    test_keys = set(all_keys[n_train:])

    def subset_data(keys):
        subset = {k: DATA_new['data'][k] for k in keys}
        # Preserve corresponding sim_lengths
        lengths = [l for k, l in zip(all_keys, DATA_new['meta_data']['sim_lengths']) if k in keys]
        return {'data': subset, 'meta_data': {'sim_lengths': lengths}}

    TRAIN_DATA = subset_data(train_keys)
    TEST_DATA = subset_data(test_keys)
    print(f"Split {n_total} simulations: {len(train_keys)} for training, {len(test_keys)} for testing.")
else:
    # No split requested – treat the whole set as training data
    TRAIN_DATA = DATA_new
    TEST_DATA = None

avg_len = sum(TRAIN_DATA['meta_data']['sim_lengths']) / len(TRAIN_DATA['meta_data']['sim_lengths'])
print(f"Updated average training simulation length: {avg_len:.1f}")

# ----------------------------------------------------------------------
# Determine which input and output keys to plot (defaults if not provided)
# ----------------------------------------------------------------------
input_keys = clean_keys(args.input_keys) if args.input_keys else ["t", "evm"]
raw_output_keys = clean_keys(args.output_keys) if args.output_keys else ["rhoc", "U[evm]"]

# Representative simulation used to inspect where keys live
sample_sim = next(iter(TRAIN_DATA["data"].values()), None)
if sample_sim is None and TEST_DATA is not None:
    sample_sim = next(iter(TEST_DATA["data"].values()), None)
if sample_sim is None:
    sample_sim = next(iter(DATA_new["data"].values()))

# Convert plain output key names into the 'U[<key>]' syntax expected by the plotting utilities
adjusted_output_keys = []
for key in raw_output_keys:
    if key.startswith("U["):                                   # already in required form
        adjusted_output_keys.append(key)
    elif key in sample_sim:                                    # top-level entry, keep
        adjusted_output_keys.append(key)
    else:                                                      # inside 'U' (or last resort: error later)
        adjusted_output_keys.append(f"U[{key}]")

output_keys = [
    k for k in adjusted_output_keys
    if (k.startswith("U[") and k[2:-1] in sample_sim.get("U", {})) or (k in sample_sim)
]
print(f"Filtered output_keys: {output_keys}")
dropped = [k for k in adjusted_output_keys if k not in output_keys]
if dropped:
    print(f"  WARNING: dropped output keys not found in data: {dropped}")

# Input (x-axis) keys for the per-simulation trace plots
trace_input_keys = [k for k in ["t", "evm"] if k in sample_sim]
if not trace_input_keys:
    if "t" in sample_sim:
        trace_input_keys = ["t"]
    else:
        trace_input_keys = [next((k for k in sample_sim if k not in PROTECTED_KEYS), "t")]
print(f"Trace input (x-axis) keys: {trace_input_keys}")

# ----------------------------------------------------------------------
# Plot the curated (downsampled/balanced) dataset
# ----------------------------------------------------------------------
plot_kwargs = dict(input_keys=input_keys, output_keys=output_keys,
                   trace_input_keys=trace_input_keys,
                   ylog=args.ylog, xlog=args.xlog)

if args.split_data is not None:
    plot_all(TRAIN_DATA['data'], filename="curated_train", **plot_kwargs)
    if TEST_DATA is not None:
        plot_all(TEST_DATA['data'], filename="curated_testing", **plot_kwargs)
else:
    plot_all(DATA_new['data'], filename="curated", **plot_kwargs)

# ----------------------------------------------------------------------
# Augment the data (optional) – training data only
# ----------------------------------------------------------------------
if not args.anchor_points:
    anchor_input = input("\nAUGMENTATION (Optional).\nEnter comma-separated anchor points for optional augmentation (or press Enter to skip): ").strip()
    if anchor_input:
        args.anchor_points = anchor_input

if args.anchor_points:
    try:
        anchor_points = [float(x) for x in args.anchor_points.split(",")]
    except ValueError:
        print("Invalid anchor_points format. Expecting comma-separated numbers.")
        anchor_points = None

    if anchor_points:
        TRAIN_DATA = smd.augment_piecewise_data(copy.deepcopy(TRAIN_DATA),
                                                nodes=anchor_points,
                                                min_points=10,
                                                max_points_per_sim=200,
                                                remove_initial_below_max_key="vmJ2")
        plot_all(TRAIN_DATA['data'], filename="curated_and_augmented_training",
                 color='purple', **plot_kwargs)

# ----------------------------------------------------------------------
# Concatenate training data (always when --split_data is used, else if --concat)
# ----------------------------------------------------------------------
if args.split_data is not None or args.concat:
    print("\nConcatenating the curated/augmented training data for model building...")
    TRAIN_DATA = concat_and_histogram(TRAIN_DATA)

# ----------------------------------------------------------------------
# Store the curated (and optionally augmented) datasets
# ----------------------------------------------------------------------
if args.split_data is not None:
    train_output_name = f"Curated_{int(args.split_data)}pct_train_{base_filename}.pickle"
    print(f"\nSaving training dataset to {train_output_name} ...")
    smd.store_output(TRAIN_DATA, "./output", train_output_name)

    if TEST_DATA is not None:
        test_output_name = f"Curated_{int(100 - args.split_data)}pct_test_{base_filename}.pickle"
        print(f"Saving testing dataset to {test_output_name} ...")
        smd.store_output(TEST_DATA, "./output", test_output_name)
else:
    output_name = f"Curated_{base_filename}.pickle"
    print(f"\nSaving curated dataset to {output_name} ...")
    smd.store_output(TRAIN_DATA, "./output", output_name)

print("Done.")