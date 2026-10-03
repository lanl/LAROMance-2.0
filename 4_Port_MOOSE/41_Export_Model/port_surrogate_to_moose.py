"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Script to export a surrogate model to MOOSE C++ code.

"""

import argparse
import importlib as imlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle, os, copy,  sys
# import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from utils import sm_plot as smp
from utils import sm_publish as smpub

# --------------------------------------
# Load the surrogate model
parser = argparse.ArgumentParser(description='Export surrogate model to MOOSE')
parser.add_argument('model_name', type=str, nargs='?', help='Name for the exported MOOSE model')
args = parser.parse_args()
if args.model_name:
    model_name = args.model_name
else:
    # Prompt the user if no model name was supplied via the command line
    # Prompt for a model name. If the user provides no input (e.g., when the script
    # is run in a non‑interactive environment), fall back to a sensible default.
    model_name = input("Enter a name for the exported MOOSE model: ").strip()
if not model_name:
    # Use a generic placeholder name to avoid NoneType errors downstream.
    model_name = "exported_model"
model_name = args.model_name

# Load surrogate model from pickle
path = os.path.join("./references", "SM.pickle")
file = open(path, 'rb')
SM= pickle.load(file)
file.close()

# transform the roi lists to lists (from numpy arrays) for MOOSE compatibility
for i, key in enumerate(SM['roi'].keys()):
    SM['roi'][key][0] = SM['input_maps'][key].inverse_transform(SM['mesh']['nodes'][:,i].min().reshape(-1,1))[0][0]
    SM['roi'][key][1] = SM['input_maps'][key].inverse_transform(SM['mesh']['nodes'][:,i].max().reshape(-1,1))[0][0]
    SM['roi'][key] = list(SM['roi'][key])

# # Add extrapolation elements for temperature and stress using generic wrappers
# SM_extrapolated_temperature = smpub.add_temperature_extrapolation_6d_element(copy.deepcopy(SM), new_lower_bound_temperature=293.0, evm_output=1e-28)
# SM_extrapolated_stress_temperature = smpub.add_stress_extrapolation_6d_element(copy.deepcopy(SM_extrapolated_temperature), new_upper_bound_vmJ2=2000.0, evm_output=1.0e-14)

SM_extrapolated_stress_temperature = SM  # Use the original SM without extrapolation for now

# plot the mesh of the extrapolated surrogate model
smp.visualize_mesh(copy.deepcopy(SM_extrapolated_stress_temperature['mesh']['nodes']), SM_extrapolated_stress_temperature['mesh']['conn'], 2, SM_extrapolated_stress_temperature['input_maps'], show_node_numbers=False, show_elem_numbers=False)
fig = plt.gcf()          # get current Figure
fig.savefig(os.path.join("./output", "mesh_visualization.png"), dpi=96, bbox_inches='tight')
print(f"\n2D Mesh layout saved to ./output")

# Store the extrapolated surrogate
SM_path = os.path.join('./output', "SM.pickle")
with open(SM_path, "wb") as file:
        pickle.dump(SM_extrapolated_stress_temperature, file)

# Publish the model under the name using the generic wrapper.
print("Writing MOOSE model files...")
textC = smpub.write_moose_C_4D_cached(SM_extrapolated_stress_temperature, name = model_name)
fileC = open(os.path.join('./output', str(model_name) +'.C'), "w")
fileC.write(textC)
fileC.close()
texth = smpub.write_moose_h_cached(SM_extrapolated_stress_temperature, name = model_name)
fileh = open(os.path.join('./output', str(model_name) +'.h'), "w")
fileh.write(texth)
fileh.close()
print("\tDone.")

# -------------------------------------------------------------------
# Generate verification table (use original SM, not SM_extrapolated)
# -------------------------------------------------------------------
print("Generating verification table...")
filename = "benchmarks.csv"
# Use generic verification table generation; number of inputs can be any size.
csv_data = smpub.generate_verification_table(SM, num_inputs = 1000,
                                             output_directory='./output', output_filename=filename)


# # Run the generation of the verification input file
# print("Generating verification input...")
# smpub.generate_verification_input("./output/benchmarks.csv", "./references/verification.i", "./output/verification.i")