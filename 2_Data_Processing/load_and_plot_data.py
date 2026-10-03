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
- (Optionally) de‑concatenate it.
- Curate / down‑sample the data.
- Store the curated dataset back to disk.
- Plot the data in scatter plots and time history traces

"""

import os, copy, pickle, argparse, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from utils import sm_data as smd
from utils import sm_plot as smp

# Directory containing the reference pickle files (relative to this script)
data_path = os.path.join(os.path.dirname(__file__), "references")

# NOTE: Data loading and merging now occurs after argument parsing (see later in the file).

# ----------------------------------------------------------------------
# Parse command‑line arguments for down‑sampling parameters
# ----------------------------------------------------------------------
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
    help="Down‑sampling factor (e.g., 1.3 reduces points by ~30%).",
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
parser.add_argument('--anchor_points', type=str, default=None,
    help='Comma-separated list of anchor points for optional data augmentation.')
parser.add_argument('--correct_rhoc', action='store_true',
    help='Apply Karl‑specific correction to U[rhoc] rates.')
parser.add_argument('--concat', action='store_true',
    help='If set, concatenate the curated/augmented dataset and generate the histogram plot.')
# Accept a comma‑separated list of pickle files to merge. If omitted the script
# retains its original behaviour of loading the single hard‑coded `filename`.
parser.add_argument('--pickle_files', type=str, default=None,
    help='Comma‑separated list of pickle files (relative to the references folder) to load and merge.')
args = parser.parse_args()
# Prompt for input and output variable keys if not supplied via CLI
if not args.input_keys:
    inp = input("\nEnter input variable keys for plots (comma-separated) or press Enter for defaults: ").strip()
    if inp:
        args.input_keys = inp
if not args.output_keys:
    outp = input("\nEnter output variable keys for plots (comma-separated) or press Enter for defaults: ").strip()
    if outp:
        args.output_keys = outp

# ----------------------------------------------------------------------
# Interactive prompts for optional flags
# ----------------------------------------------------------------------
# If no arguments provided, interactively ask for maximum length first.
# If the user skips (presses Enter), then ask for a down‑sampling factor.
if args.max_length is None and args.factor is None:
    # Ask for max length
    max_input = input("\nEnter maximum length for down‑sampling (or press Enter to set a factor): ").strip()
    if max_input:
        try:
            downsample_max_length = int(max_input)
            downsample_factor = None
        except ValueError:
            print("Invalid max length input, proceeding to factor prompt.")
            downsample_max_length = None
            downsample_factor = None
    else:
        downsample_max_length = None
        # Ask for factor
        factor_input = input("Enter down‑sampling factor (e.g., 1.5) or press Enter to skip: ").strip()
        if factor_input:
            try:
                downsample_factor = float(factor_input)
            except ValueError:
                print("Invalid factor input, proceeding without down‑sampling.")
                downsample_factor = None
        else:
            downsample_factor = None
else:
    downsample_max_length = args.max_length
    downsample_factor = args.factor

# Prompt for concatenation if not supplied via CLI
if not args.concat:
    concat_input = input("\nCONCATENATE for Training? (True/False) ").strip().lower()
    if concat_input in ("true", "t", "yes", "y", "1"):
        args.concat = True
    else:
        args.concat = False

# ----------------------------------------------------------------------
# Load and optionally merge pickle files now that command‑line arguments have
# been processed.  This replaces the original single‑file loading that occurred
# at the top of the script.
# ----------------------------------------------------------------------
# ``data_path`` points to the ``references`` directory (defined near the top).
datasets = []
if args.pickle_files:
    file_list = [fn.strip() for fn in args.pickle_files.split(',') if fn.strip()]
else:
    # Preserve original behaviour – load the single default file.
    file_list = [
        'ht9_0_implicit.pickle',
    ]
for fn in file_list:
    full_path = os.path.join(data_path, fn)
    print(f"Loading data from {full_path} ...")
    with open(full_path, "rb") as f:
        datasets.append(pickle.load(f))

# Merge datasets if more than one was provided.
if len(datasets) == 1:
    DATA = datasets[0]
else:
    merged_data = {}
    merged_sim_lengths = []
    id_offset = 0
    for d in datasets:
        data_dict = d["data"]
        meta = d.get("meta_data", {})
        sim_lengths = meta.get("sim_lengths", [len(v["t"]) for v in data_dict.values()])
        for old_id, sim in data_dict.items():
            new_id = old_id + id_offset
            merged_data[new_id] = sim
        merged_sim_lengths.extend(sim_lengths)
        if merged_data:
            id_offset = max(merged_data.keys()) + 1
    merged_meta = {
        "sim_lengths": merged_sim_lengths,
        "job_number": list(range(len(merged_data))),
    }
    DATA = {"data": merged_data, "meta_data": merged_meta}

# NOTE: Data loading and merging now occurs after argument parsing (see later in the file).
# Extract convenience references and print basic information about the loaded data
# ----------------------------------------------------------------------
data = DATA["data"]
meta_data = DATA["meta_data"]
print(f"Loaded {len(data)} simulations")
# Compute and print average simulation length
avg_len = sum(meta_data['sim_lengths']) / len(meta_data['sim_lengths'])
print(f"Average simulation length: {avg_len:.1f}")
first_key = list(data.keys())[0]
print("\nOutput variable labels: {}".format(list(data[first_key]['U'].keys())))
print("Input variable labels: {}".format(list(data[first_key].keys())))

# Create a base name for output files derived from the loaded pickle filenames.
# This replaces the old single ``filename`` variable.
base_filename = "_".join([os.path.splitext(os.path.basename(fn))[0] for fn in file_list])
print("Input variable labels: {}".format(list(data[first_key].keys())))

if args.correct_rhoc:
    print("\nCorrecting output data in U[rhoc] to be rates. Specific to Karl's dataset.")
    data_tmp = DATA['data']
    for sim in data.keys():
        data_tmp[sim]['U']['rhoc'] = np.diff(data_tmp[sim]['rhoc']) / data_tmp[sim]['dt']
        data_tmp[sim]['U']['rhoc'] = np.concatenate(
            ([data_tmp[sim]['U']['rhoc'][0]], data_tmp[sim]['U']['rhoc'])
        )
    DATA['data'] = data_tmp

# ----------------------------------------------------------------------
# Curate / down‑sample the dataset
# ----------------------------------------------------------------------
print("\nCurating dataset (down‑sampling)...")
DATA_new = smd.curate_dataset(
                            copy.deepcopy(DATA),
                            downsample_max_length=downsample_max_length,
                            downsample_factor=downsample_factor,
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
        meta = {'sim_lengths': lengths}
        return {'data': subset, 'meta_data': meta}

    TRAIN_DATA = subset_data(train_keys)
    TEST_DATA = subset_data(test_keys)
    print(f"Split {n_total} simulations: {len(train_keys)} for training, {len(test_keys)} for testing.")
else:
    # No split requested – treat the whole set as training data
    TRAIN_DATA = DATA_new
    TEST_DATA = None
# Compute and print average simulation length for training data
avg_len = sum(TRAIN_DATA['meta_data']['sim_lengths']) / len(TRAIN_DATA['meta_data']['sim_lengths'])
print(f"Updated average training simulation length: {avg_len:.1f}")
# Determine which input and output keys to plot (defaults if not provided)
# Input keys are taken directly; output keys may need to be mapped to the 'U' sub‑dictionary.
# Clean any stray whitespace or non‑printable characters (e.g., accidental escape sequences) from the keys.
input_keys = args.input_keys.split(",") if args.input_keys else ["t", "evm"]
input_keys = [k.strip() for k in input_keys]
input_keys = [''.join(ch for ch in k if ch.isprintable()) for k in input_keys]

raw_output_keys = args.output_keys.split(",") if args.output_keys else ["rhoc", "U[evm]"]
raw_output_keys = [k.strip() for k in raw_output_keys]
raw_output_keys = [''.join(ch for ch in k if ch.isprintable()) for k in raw_output_keys]

# Choose a representative simulation to inspect where keys live.
# Prefer training data, otherwise testing data, otherwise the full curated dataset.
sample_sim = None
if TRAIN_DATA and isinstance(TRAIN_DATA, dict) and "data" in TRAIN_DATA:
    # Grab the first simulation dict
    for sim in TRAIN_DATA["data"].values():
        sample_sim = sim
        break
elif TEST_DATA and isinstance(TEST_DATA, dict) and "data" in TEST_DATA:
    for sim in TEST_DATA["data"].values():
        sample_sim = sim
        break
else:
    for sim in DATA_new["data"].values():
        sample_sim = sim
        break

# Transform plain output key names into the 'U[<key>]' syntax when they reside under the 'U' dict.
# Ensure that any plain output keys are converted to the ``U[<key>]`` syntax expected by the
# plotting utilities.  This handles cases where the user supplies ``evm`` instead of
# ``U[evm]`` while still allowing keys that are already correctly formatted.
adjusted_output_keys = []
if sample_sim is not None:
    for key in raw_output_keys:
        key = key.strip()
        # Already in the required ``U[…]`` form – keep as is
        if key.startswith("U["):
            adjusted_output_keys.append(key)
        # If the key exists as a top‑level entry in the simulation dict, keep it
        elif key in sample_sim:
            adjusted_output_keys.append(key)
        # If the key is present inside the ``U`` sub‑dictionary, convert it
        elif isinstance(sample_sim.get("U", None), dict) and key in sample_sim["U"]:
            adjusted_output_keys.append(f"U[{key}]")
        # As a last resort, try to prefix with ``U[…]`` – this will raise a clear
        # error later if the key does not exist, making debugging easier.
        else:
            adjusted_output_keys.append(f"U[{key}]")
else:
    # No sample simulation available – fall back to converting everything
    adjusted_output_keys = [f"U[{k}]" for k in raw_output_keys]

output_keys = [k for k in adjusted_output_keys if (k.startswith("U[") and k[2:-1] in sample_sim.get("U", {})) or (k in sample_sim)]
print(f"Filtered output_keys: {output_keys}")
# Use the same list for plottraces (no further transformation needed)
plottraces_output_keys = output_keys

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Plot the curated (downsampled/balanced) dataset 
# ---------------------------------------------------------------------
if args.split_data is not None:
    # Plot training subset
    smp.plotdata(TRAIN_DATA['data'], input_keys=input_keys, output_keys=output_keys,
                 simulation_recording_interval=1, inc=-1, cmap='magma',
                 storepath="./output", filename="_Curated_train_scatter_",
                 dark=True)
    smp.plottraces(TRAIN_DATA['data'], input_keys=["t", "evm"], output_keys=plottraces_output_keys,
                   ylog=True,
                   storepath="./output", filename="_Curated_train_trace_",
                   color='green',
                   dark=True)
    # Plot testing subset (if available)
    if TEST_DATA is not None:
        smp.plotdata(TEST_DATA['data'], input_keys=input_keys, output_keys=output_keys,
                     simulation_recording_interval=1, inc=-1, cmap='magma',
                     storepath="./output", filename="_Curated_test_scatter_",
                     dark=True)
        smp.plottraces(TEST_DATA['data'], input_keys=["t", "evm"], output_keys=plottraces_output_keys,
                       ylog=True,
                       storepath="./output", filename="_Curated_test_trace_",
                       color='green',
                       dark=True)
else:
    # Original behavior – plot the full curated dataset
    smp.plotdata(DATA_new['data'], input_keys=input_keys, output_keys=output_keys,
                 simulation_recording_interval=1, inc=-1, cmap='magma',
                 storepath="./output", filename="_Curated_",
                 dark=True)

    smp.plottraces(DATA_new['data'], input_keys=["t", "evm"], output_keys=plottraces_output_keys,
                   ylog=True,
                   storepath="./output", filename="_Curated_",
                   color='green',
                   dark=True)

# ---------------------------------------------------------------------
# Augment the data (optional)
# ---------------------------------------------------------------------
# Prompt for anchor points if not supplied via CLI
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
        # Apply augmentation only to training data
        TRAIN_DATA = smd.augment_piecewise_data(copy.deepcopy(TRAIN_DATA),
                                                 nodes=anchor_points,
                                                 min_points=10,
                                                 max_points_per_sim=200,
                                                 remove_initial_below_max_key="vmJ2")
        # Plot the augmented training dataset
        smp.plotdata(TRAIN_DATA['data'], input_keys=input_keys, output_keys=output_keys,
                     simulation_recording_interval=1, inc=-1, cmap='magma',
                     storepath="./output", filename="_Curated_and_Augmented_scatter_",
                     dark=True)
        smp.plottraces(TRAIN_DATA['data'], input_keys=["t", "evm"], output_keys=output_keys,
                       ylog=True,
                       storepath="./output", filename="_Curated_and_Augmented_trace_",
                       color='purple',
                       dark=True)

# ----------------------------------------------------------------------
# Concatenate the curated (and optionally augmented) dataset for training
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Concatenate training data (override any --concat flag when --split_data is used)
# ----------------------------------------------------------------------
if args.split_data is not None:
    print("\nConcatenating training data for model building...")
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
elif args.concat:
    # Original behavior – concatenate full dataset
    print("\nConcatenating the curated/augmented dataset for training...")
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

# ----------------------------------------------------------------------
# Store the curated and optionally augmented dataset
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Store the curated (training) and testing datasets
# ----------------------------------------------------------------------
if args.split_data is not None:
    # Save training dataset (concatenated)
    train_output_name = f"Curated_{int(args.split_data)}pct_train_{base_filename}.pickle"
    print(f"\nSaving training dataset to {train_output_name} ...")
    smd.store_output(TRAIN_DATA, "./output", train_output_name)

    if TEST_DATA is not None:
        # Save testing dataset (unconcatenated)
        test_output_name = f"Curated_{int(100-args.split_data)}pct_test_{base_filename}.pickle"
        print(f"Saving testing dataset to {test_output_name} ...")
        smd.store_output(TEST_DATA, "./output", test_output_name)
else:
    # Original behavior – save the (potentially concatenated) full dataset
    output_name = f"Curated_{base_filename}"
    print(f"\nSaving curated dataset to {output_name} ...")
    smd.store_output(TRAIN_DATA, "./output", output_name)

print("Done.")
