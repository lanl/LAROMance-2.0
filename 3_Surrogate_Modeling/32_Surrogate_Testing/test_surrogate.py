"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Testing script for creep simulations. 

"""

import pickle, os, copy, sys
import numpy as np
import argparse
import itertools

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from utils import sm_build as smb
from utils import sm_plot as smp

# ----------------------------------------------------------------------
# Load dataset and surrogate model
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Command‑line interface for oracle (truth‑injection) runs
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Surrogate model testing with optional truth‑injection (oracle) runs."
)

# --oracle-mode: no default – will be prompted if missing
parser.add_argument(
    "--oracle-mode",
    choices=["coupled", "single", "pairwise", "all"],
    help=(
        "Select which runs to execute: "
        "'coupled' – only the default run with no forced outputs; "
        "'single' – coupled plus one‑output‑forced runs; "
        "'pairwise' – run baseline plus all configurations where exactly one output is *not* forced (i.e., inject truth for all but one output); "
        "'all' – execute both the 'single' and 'pairwise' configurations (baseline plus all single‑output and pairwise‑output runs)."
    ),
)

# Other optional arguments retain defaults
parser.add_argument(
    "--force",
    nargs="*",
    default=[],
    help="Explicitly force specific outputs (e.g., --force rhoc rhow). Overrides --oracle-mode.",
)
parser.add_argument(
    "--output-directory",
    default="./output",
    help="Directory where result files will be written."
)
parser.add_argument(
    "--store-name-addons",
    default="",
    help="Optional suffix added to result filenames."
 )

# Allow the user to specify an alternate data pickle (or a list of them, mirroring the data‑processing script).
parser.add_argument(
    "--pickle_files",
    type=str,
    default=None,
    help="Comma‑separated list of pickle files (relative to the './references' folder) to load as the data set. If omitted the script falls back to the default 'HT9FY26_20pct_test.pickle'."
)

args_cli = parser.parse_args()

# ----------------------------------------------------------------------
# Load the dataset (and surrogate) based on the provided --pickle_files argument
# ----------------------------------------------------------------------
if args_cli.pickle_files:
    # Use the first file from the comma‑separated list, matching the behaviour of the data‑processing utilities.
    file_list = [fn.strip() for fn in args_cli.pickle_files.split(',') if fn.strip()]
    data_filename = file_list[0]
else:
    print("No pickle files specified. Using default data file if present (may not exist).")
    data_filename = 'Test_Data.pickle'

# Load the selected data pickle
with open(os.path.join('./references', data_filename), 'rb') as f:
    DATA = pickle.load(f)

# Load the surrogate model (always from SM.pickle in the references folder)
with open(os.path.join('./references', "SM.pickle"), 'rb') as f:
    SM = pickle.load(f)

# Prompt for oracle-mode if not provided on the command line
if args_cli.oracle_mode is None:
    while True:
        user_input = input(
            "Select oracle mode (coupled/single/pairwise/all) [coupled]: "
        ).strip()
        if not user_input:
            user_input = "coupled"
        if user_input in ("coupled", "single", "pairwise", "all"):
            args_cli.oracle_mode = user_input
            break
        else:
            print(
                "Invalid choice. Please enter 'coupled', 'single', 'pairwise', or 'all'."
            )

# ----------------------------------------------------------------------
# Base argument dictionary used for each validation run
# ----------------------------------------------------------------------
base_args = {}
base_args['output_directory'] = args_cli.output_directory
base_args['use_rtree'] = True
base_args['debug'] = False
base_args['correct_large_evm'] = True
base_args['correct_large_rhoc'] = True
base_args['correct_large_rhow'] = True
base_args['correct_negatives'] = True
base_args['sim_range'] = np.array(list(DATA['data'].keys()))[0:]

# ----------------------------------------------------------------------
# Identify available coupled outputs in the surrogate model
# ----------------------------------------------------------------------
output_keys = list(SM.get('nodal_values', {}).keys())
if not output_keys:
    raise RuntimeError("No outputs found in SM['nodal_values'].")

def run_testing(forced_outputs):
    """
    Execute sm_test for a specific set of forced outputs and generate plots.
    """
    run_args = base_args.copy()

    # Set force flags for requested outputs
    for key in output_keys:
        flag = f'force_{key}'
        if key in forced_outputs:
            run_args[flag] = True
            run_args[f'{flag}_end_point'] = 1e16   # default end‑point, can be tuned
        else:
            run_args.pop(flag, None)
            run_args.pop(f'{flag}_end_point', None)

    # Build a unique name addon for this run
    if forced_outputs:
        forced_suffix = "_".join(sorted(forced_outputs))
        run_args['store_name_addon'] = f"{forced_suffix}-oracle"
    else:
        run_args['store_name_addon'] = "coupled"

    # Include any user‑provided extra suffix
    if args_cli.store_name_addons:
        run_args['store_name_addon'] += f"_{args_cli.store_name_addons}"

    # ------------------------------------------------------------------
    # Run validation and plot for each output
    # ------------------------------------------------------------------
    test_results = smb.sm_test(SM, run_args, DATA=copy.deepcopy(DATA))
  

    # Determine which keys to plot:
    #   • In coupled mode (no forced outputs) plot all available outputs.
    #   • In forced (truth‑injection) mode, plot all outputs except those forced.
    if not forced_outputs:
        keys_to_plot = output_keys
        print("Running coupled validation (no forced outputs). Plotting all available outputs. {}".format(keys_to_plot))
    else:
        keys_to_plot = [k for k in output_keys if k not in forced_outputs]
    for key in keys_to_plot:
        # Skip plotting for outputs that were forced (truth‑injected) in this run
        if key in forced_outputs:
            continue
        # Inform the user which output is being plotted for this configuration
        print(f"Generating plot for output '{key}' (forced set: {sorted(forced_outputs)})")
        try:
                smp.test_report_plots(
                test_results,
                z_key=key,
                max_time_in_maps=1e9,
                plot_surf=True,
                path=run_args['output_directory'],
                name_addon=run_args['store_name_addon'],
                dark=True
                )
        except Exception as e:
                print(f"Error generating plots for output '{key}': {e}. Try running without surface plots (plot_surf=False).")
                smp.test_report_plots(
                test_results,
                z_key=key,
                max_time_in_maps=1e9,
                plot_surf=False,
                path=run_args['output_directory'],
                name_addon=run_args['store_name_addon'],
                dark=True
                )
                
# ----------------------------------------------------------------------
# Determine which configurations to execute
# ----------------------------------------------------------------------
if args_cli.force:
    # User explicitly specified which outputs to force (single set)
    forced_sets = [set(args_cli.force)]
else:
    mode = args_cli.oracle_mode
    if mode == "coupled":
        forced_sets = [set()]                                 # no forced outputs
    elif mode == "single":
        forced_sets = [set()] + [{key} for key in output_keys]  # include coupled and each single forced run
    elif mode == "pairwise":
        # Baseline + each configuration where exactly one output is omitted (i.e., force all others)
        forced_sets = [set()]                                 # baseline (no forced outputs)
        for key in output_keys:
            forced_sets.append(set(output_keys) - {key})
    elif mode == "all":
        # Combine single‑output and pairwise (all‑but‑one) configurations, baseline only once
        single_sets = [{key} for key in output_keys]
        pairwise_sets = [set(output_keys) - {key} for key in output_keys]
        forced_sets = [set()] + single_sets + pairwise_sets
    else:
        forced_sets = [set()]  # safety fallback

# ----------------------------------------------------------------------
# Execute validation for each configuration
# ----------------------------------------------------------------------
for forced in forced_sets:
    run_testing(forced)