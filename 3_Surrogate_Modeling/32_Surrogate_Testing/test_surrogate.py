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
# Command‑line interface
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Surrogate model testing with optional truth‑injection (oracle) runs."
)
parser.add_argument(
    "--oracle-mode",
    choices=["coupled", "single", "pairwise", "all"],
    help="Select which runs to execute.",
)
parser.add_argument(
    "--force",
    nargs="*",
    default=[],
    help="Explicitly force specific outputs.",
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
parser.add_argument(
    "--pickle_files",
    type=str,
    default=None,
    help="Comma‑separated list of pickle files."
)
args_cli = parser.parse_args()

# ----------------------------------------------------------------------
# Load data and surrogate model
# ----------------------------------------------------------------------
if args_cli.pickle_files:
    file_list = [fn.strip() for fn in args_cli.pickle_files.split(',') if fn.strip()]
    data_filename = file_list[0]
else:
    data_filename = 'Test_Data.pickle'

with open(os.path.join('./references', data_filename), 'rb') as f:
    DATA = pickle.load(f)

with open(os.path.join('./references', "SM.pickle"), 'rb') as f:
    SM = pickle.load(f)

# ----------------------------------------------------------------------
# Prompt for oracle-mode if needed
# ----------------------------------------------------------------------
if args_cli.oracle_mode is None:
    while True:
        user_input = input("Select oracle mode (coupled/single/pairwise/all) [coupled]: ").strip()
        if not user_input:
            user_input = "coupled"
        if user_input in ("coupled", "single", "pairwise", "all"):
            args_cli.oracle_mode = user_input
            break
        else:
            print("Invalid choice.")

# ----------------------------------------------------------------------
# Base arguments
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
# Determine output keys (handle both Grid-based and MLP models)
# ----------------------------------------------------------------------
if "mlp_model_path" in SM or "mlp_model" in SM:
    print("[test_surrogate] MLP model detected.")
    output_keys = SM.get("mlp_output_vars", [])
    is_mlp = True
else:
    output_keys = list(SM.get('nodal_values', {}).keys())
    is_mlp = False

if not output_keys:
    raise RuntimeError("No outputs found in SM (neither 'nodal_values' nor 'mlp_output_vars').")

# ----------------------------------------------------------------------
# Main testing function
# ----------------------------------------------------------------------
def run_testing(forced_outputs):
    run_args = base_args.copy()

    # Set force flags
    for key in output_keys:
        flag = f'force_{key}'
        if key in forced_outputs:
            run_args[flag] = True
            run_args[f'{flag}_end_point'] = 1e16
        else:
            run_args.pop(flag, None)
            run_args.pop(f'{flag}_end_point', None)

    # Name addon
    if forced_outputs:
        forced_suffix = "_".join(sorted(forced_outputs))
        run_args['store_name_addon'] = f"{forced_suffix}-oracle"
    else:
        run_args['store_name_addon'] = "coupled"

    if args_cli.store_name_addons:
        run_args['store_name_addon'] += f"_{args_cli.store_name_addons}"

    # === Detect inference path ===
    if is_mlp or "mlp_model_path" in SM or "mlp_model" in SM:
        print("[test_surrogate] Using MLP inference path")
        run_args["use_mlp"] = True
        run_args["mlp_model"] = SM.get("mlp_model")
        run_args["mlp_input_vars"] = SM.get("mlp_input_vars", [])
        run_args["mlp_output_vars"] = SM.get("mlp_output_vars", [])
    else:
        print("[test_surrogate] Using Grid-based inference path")

    # Run testing
    test_results = smb.sm_test(SM, run_args, DATA=copy.deepcopy(DATA))

    # Determine which outputs to plot
    if not forced_outputs:
        keys_to_plot = output_keys
        print(f"Running coupled validation. Plotting all outputs: {keys_to_plot}")
    else:
        keys_to_plot = [k for k in output_keys if k not in forced_outputs]

    for key in keys_to_plot:
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
            print(f"Error with surface plots for '{key}': {e}. Retrying without surface plots.")
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
# Determine which configurations to run
# ----------------------------------------------------------------------
if args_cli.force:
    forced_sets = [set(args_cli.force)]
else:
    mode = args_cli.oracle_mode
    if mode == "coupled":
        forced_sets = [set()]
    elif mode == "single":
        forced_sets = [set()] + [{key} for key in output_keys]
    elif mode == "pairwise":
        forced_sets = [set()] + [set(output_keys) - {key} for key in output_keys]
    elif mode == "all":
        single_sets = [{key} for key in output_keys]
        pairwise_sets = [set(output_keys) - {key} for key in output_keys]
        forced_sets = [set()] + single_sets + pairwise_sets
    else:
        forced_sets = [set()]

# ----------------------------------------------------------------------
# Execute all configurations
# ----------------------------------------------------------------------
for forced in forced_sets:
    run_testing(forced)