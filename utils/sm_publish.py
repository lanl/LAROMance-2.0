"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

MOOSE publication scripts

"""
import numpy as np
import copy, random, os, csv

from utils import sm_data as smd
from utils import sm_build as smb


# -------------------------------------------------------------------
# Generate a verification input file for MOOSE (verification.i)
# -------------------------------------------------------------------
def generate_verification_input(
    csv_path: str = os.path.join("./output", "benchmarks.csv"),
    # The template is a copy of the reference file placed in the output folder.
    # Do NOT modify the original reference under ./references.
    template_path: str = os.path.join("./output", "verification.i"),
    output_path: str = os.path.join("./output", "verification.i"),
) -> None:
    """Adapt the ``verification.i`` file to the columns generated in ``benchmarks.csv``.

    This reads the CSV header, builds a ``PiecewiseConstant`` ``Function`` entry for each
    column (skipping the time column and any ``dt`` column), and replaces the existing
    ``[Functions]`` block in the template file with the new definitions.  The result is
    written back to ``output_path`` (in‑place modification of the copied template).

    Parameters
    ----------
    csv_path : str
        Path to the generated CSV benchmark file.
    template_path : str
        Path to the ``verification.i`` template copy in the *output* directory.
    output_path : str
        Destination path for the adapted ``verification.i`` file.
    """
    import csv

    # -------------------------------------------------------------------
    # 1️⃣  Read CSV header
    # -------------------------------------------------------------------
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)

    # -------------------------------------------------------------------
    # 2️⃣  Load template file lines (template_path points to the copy in ./output)
    # -------------------------------------------------------------------
    with open(template_path, "r") as f:
        lines = f.readlines()

    # -------------------------------------------------------------------
    # 3️⃣  Locate the [Functions] block (handles empty block case)
    # -------------------------------------------------------------------
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "[Functions]":
            start_idx = i
        # The block ends with a standalone [] line after the start tag.
        elif start_idx is not None and line.strip() == "[]":
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise RuntimeError("Could not locate the [Functions] block in the template file.")

    # -------------------------------------------------------------------
    # 4️⃣  Build new function definitions based on the CSV headers
    # -------------------------------------------------------------------
    function_lines: list[str] = []
    for col_idx, header in enumerate(headers):
        # Skip the time column (index 0) and any "dt" column
        if col_idx == 0 or header.lower() == "dt":
            continue
        # Derive a clean function name: remove trailing "_in"/"_out" if present and add "_fcn"
        base_name = header
        if base_name.endswith("_in") or base_name.endswith("_out"):
            base_name = base_name.rsplit("_", 1)[0]
        func_name = f"{base_name}_fcn"
        function_lines.extend(
            [
                f"  [{func_name}]\n",
                "    type = PiecewiseConstant\n",
                "    data_file = benchmarks.csv\n",
                "    x_index_in_file = 0\n",
                f"    y_index_in_file = {col_idx}\n",
                "    format = columns\n",
                "    xy_in_file_only = false\n",
                "    direction = LEFT_INCLUSIVE\n",
                "  []\n",
            ]
        )

    # Insert the new function definitions after the opening [Functions] line
    # and before the terminating [] line.
    new_lines = (
        lines[: start_idx + 1] + function_lines + lines[end_idx:]
    )

    # -------------------------------------------------------------------
    # 4️⃣  Locate the [Postprocessors] block (handles empty block case)
    # -------------------------------------------------------------------
    pp_start_idx = None
    pp_end_idx   = None
    for i, line in enumerate(lines):
        # Find the opening tag
        if line.strip() == "[Postprocessors]":
            pp_start_idx = i
        # After the opening tag, the block ends with a standalone [] line
        elif pp_start_idx is not None and line.strip() == "[]":
            pp_end_idx = i
            break

    if pp_start_idx is None or pp_end_idx is None:
        raise RuntimeError(
            "Could not locate the [Postprocessors] block in the template file."
        )
    
    lines = new_lines
    new_lines = (
        lines[: pp_start_idx + 1]          # keep everything up to the opening tag
        + postprocessor_lines             # your generated entries
        + lines[pp_end_idx:]              # keep the closing [] and the rest of the file
    )
    

    # -------------------------------------------------------------------
    # 5️⃣  Write the updated file back to output_path
    # -------------------------------------------------------------------
    with open(output_path, "w") as f:
        f.writelines(new_lines)

def generate_verification_5d_table(SM, roi, num_inputs = 1000, output_directory = None, output_filename = "verification_ref"):
    """
    Generate verification .csv table for MOOSE verification.

    Inputs:
    - SM : dict
        Surrogate model dict
    - num_inputs : int
        Number of verification points
    - output_directory : str
        Path to store .csv file

    """

    evm_rand = [random.uniform(roi['evm'][0], roi['evm'][1]) for _ in range(num_inputs)]
    rhoc_rand = [random.uniform(roi['rhoc'][0], roi['rhoc'][1]) for _ in range(num_inputs)]
    rhow_rand = [random.uniform(roi['rhow'][0], roi['rhow'][1]) for _ in range(num_inputs)]
    temp_rand = [random.uniform(roi['temperature'][0], roi['temperature'][1]) for _ in range(num_inputs)]
    vmJ2_rand = [random.uniform(roi['vmJ2'][0], roi['vmJ2'][1]) for _ in range(num_inputs)]

    # create an input deck: more than one input deck can be provided, each in a new key input[0], input[1], etc.
    input = {}
    for i in range(num_inputs):
            input[i] = {'vmJ2' :[vmJ2_rand[i]],
                    'temperature' : [temp_rand[i]],
                    'rhoc' : [rhoc_rand[i]],
                    'rhow' : [rhow_rand[i]],
                    'evm' : [evm_rand[i]],
                    'dt' : np.ones(2)*1e-1,
                    't' : [0.0],
                    }


    # some additional simulation specs
    args = {'map_output' : True,
            "output_directory" : output_directory,
            'max_length' : None,
            'concat' : False,
            'force_rhoc' : True,
            'force_rhow' : True,
            'force_evm' : True,
            'force_rhoc_end_point' : 3e-1,
            'force_rhow_end_point' : 3e-1,
            'force_evm_end_point' : 3e-1,
            'adaptive_time' : False,
            'debug' : False
            }

    # run a creep simulation
    creep_output = smb.run_creep(input, SM, sim_range=range(len(input)), args = args)

    ## ============================
    ## STORE RESULT IN .CSV FILE
    ## ----------------------------
    # Define headers
    csv_headers = ['time', 'rhom_in', 'rhoi_in', 'vmJ2_in', 'evm_in', 'temperature_in', 'dt_in', 'rhom_rate_out', 'rhoi_rate_out', 'creep_rate_out']

    # raise IOError("debug")
    # Fill the csv_data from the creep_output dict, define time steps and time
    csv_data = []
    # csv_dt = list(np.ones(num_inputs)*0.1)
    csv_time = 0.0
    csv_dt = 0.1
    for i in range(len(creep_output)):
        condition = (creep_output[i]['U']['evm'][-1] > 0.0 and np.abs(creep_output[i]['U']['evm'][-1]) != np.inf) and np.abs(creep_output[i]['U']['rhoc'][-1]) != np.inf and np.abs(creep_output[i]['U']['rhow'][-1]) != np.inf
        if condition:
            if input[i]['vmJ2'][-1] > 2000:
                raise IOError("debug")
            if np.abs(creep_output[i]['U']['rhoc'][-1]) < 1e-300:
                print("Subnormal value in rhoc output: {} normalized to 0.0".format(creep_output[i]['U']['rhoc'][-1]))
                creep_output[i]['U']['rhoc'][-1] = 0.0
            if np.abs(creep_output[i]['U']['rhow'][-1]) < 1e-300:
                print("Subnormal value in rhow output: {} normalized to 0.0".format(creep_output[i]['U']['rhow'][-1]))
                creep_output[i]['U']['rhow'][-1] = 0.0
            csv_data.append([csv_time, input[i]['rhoc'][0], input[i]['rhow'][0], input[i]['vmJ2'][-1], input[i]['evm'][-1], input[i]['temperature'][-1], csv_dt, creep_output[i]['U']['rhoc'][-1], creep_output[i]['U']['rhow'][-1], creep_output[i]['U']['evm'][-1]
                       ])
            csv_time+=csv_dt
            # print("\nStrain rate: {:e}, sim: {}.".format(creep_output[i]['dgedt'][-1], i))
            print("\nStrain rate: {:e}, sim: {}.".format(creep_output[i]['U']['evm'][-1], i))
        else:
            print("Invalid output detected, e.g., inf. Row excluded from verification input file. \nSim ID#: {}.\nInput stress: {} [MPa].\nInput temperature: {} [K]".format( i, creep_output[i]['vmJ2'][-1], creep_output[i]['temperature'][-1]))

    # Desired formatting: scientific notation with 15 decimals
    formatted_data = [
        [f"{value:.18e}" if isinstance(value, (float, int)) else value for value in row]
        for row in csv_data
    ]

    # raise IOError("debug")

    # Create the directories if they don't exist
    if output_directory != None:
        os.makedirs(output_directory, exist_ok=True)
        # Create the file path
        file_path = os.path.join(output_directory, output_filename)

        # Open the file and write the headers and data
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the headers as the first row
            writer.writerow(csv_headers)
            
            # Write the data rows
            writer.writerows(formatted_data)
    
    return csv_data

def generate_verification_table(
    SM,
    num_inputs: int = 1000,
    output_directory: str | None = None,
    output_filename: str = "verification_ref",
) -> list[list]:
    """
    Generate a verification ``.csv`` table for MOOSE verification – **dimension‑agnostic**.

    The region‑of‑interest (ROI) is read from ``SM['roi']``; the function therefore
    adapts automatically to any set of input dimensions that the surrogate model
    contains.  Output dimensions are taken from ``SM['nodal_values']``.

    Parameters
    ----------
    SM : dict
        Surrogate‑model dictionary.
    num_inputs : int, optional
        Number of random verification points to generate (default 1000).
    output_directory : str | None, optional
        Path where the CSV file will be written (passed to the simulation routine).
    output_filename : str, optional
        Base name for the generated CSV file (default ``"verification_ref"``).

    Returns
    -------
    csv_data : list[list]
        A list of rows that can be written with ``csv.writer``.  The first row
        (header) can be obtained from ``csv_headers`` defined inside the function.
    """
    _ORANGE = "\033[38;5;208m"   # orange
    _RESET  = "\033[0m"

    # --------------------------------------------------------------------- #
    # 1️⃣  Random sampling of the ROI (taken from SM['roi'])
    # --------------------------------------------------------------------- #
    roi = SM["roi"]                     # e.g. {'evm': (0,1), 'rhoc': (0,1e12), ...}
    rand_vals: dict[str, list[float]] = {}
    for key, (low, high) in roi.items():
        rand_vals[key] = [random.uniform(low, high) for _ in range(num_inputs)]

    # --------------------------------------------------------------------- #
    # 2️⃣  Build the “input deck’’ expected by ``smb.run_creep``.
    # --------------------------------------------------------------------- #
    input_deck: dict[int, dict] = {}
    for i in range(num_inputs):
        entry = {k: [rand_vals[k][i]] for k in roi.keys()}
        entry["dt"] = np.ones(2) * 1e-1   # time‑step vector (kept identical to the 5‑D version)
        entry["t"] = [0.0]               # initial time
        input_deck[i] = entry

    # --------------------------------------------------------------------- #
    # 3️⃣  Fixed simulation arguments (unchanged from the original implementation)
    # --------------------------------------------------------------------- #
    args = {
        "map_output": True,
        "output_directory": output_directory,
        "max_length": None,
        "concat": False,
        "force_rhoc": True,
        "force_rhow": True,
        "force_evm": True,
        "force_rhoc_end_point": 100,
        "force_rhow_end_point": 100,
        "force_evm_end_point": 100,
        "adaptive_time": False,
        "debug": False,
    }

    # --------------------------------------------------------------------- #
    # 4️⃣  Run the creep simulation for all generated inputs
    # --------------------------------------------------------------------- #
    creep_output = smb.run_creep(input_deck, SM, sim_range=range(len(input_deck)), args=args)

    # --------------------------------------------------------------------- #
    # 5️⃣  Assemble CSV headers – time, all inputs, dt, then all outputs
    # --------------------------------------------------------------------- #
    input_keys = list(roi.keys())                     # e.g. ['evm','rhoc','rhow','temperature','vmJ2','defect_rate']
    output_keys = list(SM["nodal_values"].keys())    # e.g. ['evm','rhoc','rhow', ...]

    csv_headers = (
        ["time"]
        + [f"{k}_in" for k in input_keys]
        + ["dt"]
        + [f"{k}_out" for k in output_keys]
    )

    # --------------------------------------------------------------------- #
    # 6️⃣  Populate CSV rows
    # --------------------------------------------------------------------- #
    csv_data: list[list] = []
    csv_time = 0.0
    csv_dt = 0.1
    negative_flag = 0
    infinite_flag = 0
    for i in range(len(creep_output)):
        # Keep the original sanity checks for the three core outputs (evm, rhoc, rhow)
        # If any of them are missing in ``output_keys`` the condition simply drops them.
        condition = True

        for out_key in ("evm"):
            if out_key in output_keys:
                val = creep_output[i]["U"][out_key][-1]
                condition = condition and (val >= 0)
                if condition != True:
                    negative_flag += 1

        for out_key in ("evm", "rhoc", "rhow"):
            if out_key in output_keys:
                val = creep_output[i]["U"][out_key][-1]
                condition = condition and (abs(val) != np.inf)
                if condition != True:
                    infinite_flag += 1

        if not condition:
            continue

        # Guard against sub‑normal values (kept from the original version)
        for out_key in ("rhoc", "rhow"):
            if out_key in output_keys:
                val = creep_output[i]["U"][out_key][-1]
                if abs(val) < 1e-300:
                    print(
                        f"Subnormal value in {out_key} output: {val} normalized to 0.0"
                    )
                    creep_output[i]["U"][out_key][-1] = 0.0

        # Build a row respecting the header order
        row = [csv_time]                                   # time column
        row += [input_deck[i][k][0] for k in input_keys]   # all inputs
        row.append(csv_dt)                                 # dt column
        row += [creep_output[i]["U"][k][-1] for k in output_keys]  # all outputs
        csv_data.append(row)

        csv_time += csv_dt

    # Desired formatting: scientific notation with 15 decimals
    formatted_data = [
        [f"{value:.18e}" if isinstance(value, (float, int)) else value for value in row]
        for row in csv_data
    ]
    
    if negative_flag > 0:
        print(
            f"{_ORANGE}Warning: {negative_flag} effective von Mises strain rate is "
            f"negative! Not added to .csv file.{_RESET}"
        )
    if infinite_flag > 0:
        # orange warning for infinite values
        print(
            f"{_ORANGE}Warning: {infinite_flag} values in output are infinite! "
            f"Not added to .csv file.{_RESET}"
        )

    # --------------------------------------------------------------------- #
    # 7️⃣  Return data (the caller can write the CSV using the header above)
    # --------------------------------------------------------------------- #\
    # Create the directories if they don't exist
    if output_directory != None:
        os.makedirs(output_directory, exist_ok=True)
        # Create the file path
        file_path = os.path.join(output_directory, output_filename)

        # Open the file and write the headers and data
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the headers as the first row
            writer.writerow(csv_headers)
            
            # Write the data rows
            writer.writerows(formatted_data)

    return csv_headers, csv_data


def generate_verification_6d_table(SM, roi, num_inputs = 1000, output_directory = None, output_filename = "verification_ref"):
    """
    Generate verification .csv table for MOOSE verification.

    Inputs:
    - SM : dict
        Surrogate model dict
    - num_inputs : int
        Number of verification points
    - output_directory : str
        Path to store .csv file

    """

    evm_rand = [random.uniform(roi['evm'][0], roi['evm'][1]) for _ in range(num_inputs)]
    rhoc_rand = [random.uniform(roi['rhoc'][0], roi['rhoc'][1]) for _ in range(num_inputs)]
    rhow_rand = [random.uniform(roi['rhow'][0], roi['rhow'][1]) for _ in range(num_inputs)]
    temp_rand = [random.uniform(roi['temperature'][0], roi['temperature'][1]) for _ in range(num_inputs)]
    vmJ2_rand = [random.uniform(roi['vmJ2'][0], roi['vmJ2'][1]) for _ in range(num_inputs)]
    defect_rate_rand = [random.uniform(roi['defect_rate'][0], roi['defect_rate'][1]) for _ in range(num_inputs)]

    # create an input deck: more than one input deck can be provided, each in a new key input[0], input[1], etc.
    input = {}
    for i in range(num_inputs):
            input[i] = {'vmJ2' :[vmJ2_rand[i]],
                    'temperature' : [temp_rand[i]],
                    'rhoc' : [rhoc_rand[i]],
                    'rhow' : [rhow_rand[i]],
                    'defect_rate' : [defect_rate_rand[i]],
                    'evm' : [evm_rand[i]],
                    'dt' : np.ones(2)*1e-1,
                    't' : [0.0],
                    }


    # some additional simulation specs
    args = {'map_output' : True,
            "output_directory" : output_directory,
            'max_length' : None,
            'concat' : False,
            'force_rhoc' : True,
            'force_rhow' : True,
            'force_evm' : True,
            'force_rhoc_end_point' : 3e-1,
            'force_rhow_end_point' : 3e-1,
            'force_evm_end_point' : 3e-1,
            'adaptive_time' : False,
            'debug' : False
            }

    # run a creep simulation
    creep_output = smb.run_creep(input, SM, sim_range=range(len(input)), args = args)

    ## ============================
    ## STORE RESULT IN .CSV FILE
    ## ----------------------------
    # Define headers
    csv_headers = ['time', 'rhom_in', 'rhoi_in', 'vmJ2_in', 'evm_in', 'temperature_in', 'prec_dec_in', 'dt_in', 'rhom_rate_out', 'rhoi_rate_out', 'creep_rate_out']

    # raise IOError("debug")
    # Fill the csv_data from the creep_output dict, define time steps and time
    csv_data = []
    # csv_dt = list(np.ones(num_inputs)*0.1)
    csv_time = 0.0
    csv_dt = 0.1
    for i in range(len(creep_output)):
        condition = (creep_output[i]['U']['evm'][-1] > 0.0 and np.abs(creep_output[i]['U']['evm'][-1]) != np.inf) and np.abs(creep_output[i]['U']['rhoc'][-1]) != np.inf and np.abs(creep_output[i]['U']['rhow'][-1]) != np.inf
        if condition:
            if input[i]['vmJ2'][-1] > 2000:
                raise IOError("debug")
            if np.abs(creep_output[i]['U']['rhoc'][-1]) < 1e-300:
                print("Subnormal value in rhoc output: {} normalized to 0.0".format(creep_output[i]['U']['rhoc'][-1]))
                creep_output[i]['U']['rhoc'][-1] = 0.0
            if np.abs(creep_output[i]['U']['rhow'][-1]) < 1e-300:
                print("Subnormal value in rhow output: {} normalized to 0.0".format(creep_output[i]['U']['rhow'][-1]))
                creep_output[i]['U']['rhow'][-1] = 0.0
            csv_data.append([csv_time, input[i]['rhoc'][0], input[i]['rhow'][0], input[i]['vmJ2'][-1], input[i]['evm'][-1], input[i]['temperature'][-1], input[i]['defect_rate'][-1], csv_dt, creep_output[i]['U']['rhoc'][-1], creep_output[i]['U']['rhow'][-1], creep_output[i]['U']['evm'][-1]
                       ])
            csv_time+=csv_dt
            # print("\nStrain rate: {:e}, sim: {}.".format(creep_output[i]['dgedt'][-1], i))
            print("\nStrain rate: {:e}, sim: {}.".format(creep_output[i]['U']['evm'][-1], i))
        else:
            print("Invalid output detected, e.g., inf. Row excluded from verification input file. \nSim ID#: {}.\nInput stress: {} [MPa].\nInput temperature: {} [K]".format(creep_output[i]['U']['evm'][-1], i, creep_output[i]['vmJ2'][-1], creep_output[i]['temperature'][-1]))

    # Desired formatting: scientific notation with 15 decimals
    formatted_data = [
        [f"{value:.18e}" if isinstance(value, (float, int)) else value for value in row]
        for row in csv_data
    ]

    # raise IOError("debug")

    # Create the directories if they don't exist
    if output_directory != None:
        os.makedirs(output_directory, exist_ok=True)
        # Create the file path
        file_path = os.path.join(output_directory, output_filename)

        # Open the file and write the headers and data
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the headers as the first row
            writer.writerow(csv_headers)
            
            # Write the data rows
            writer.writerows(formatted_data)
    
    return csv_data


def generate_verification_4d_table(SM, roi, num_inputs = 1000, output_directory = None, output_filename = "verification_ref"):
    """
    Generate verification .csv table for MOOSE verification.

    Inputs:
    - SM : dict
        Surrogate model dict
    - num_inputs : int
        Number of verification points
    - output_directory : str
        Path to store .csv file

    """

    evm_rand = [random.uniform(roi['evm'][0], roi['evm'][1]) for _ in range(num_inputs)]
    rhoc_rand = [random.uniform(roi['rhoc'][0], roi['rhoc'][1]) for _ in range(num_inputs)]
    rhow_rand = [random.uniform(roi['rhow'][0], roi['rhow'][1]) for _ in range(num_inputs)]
    temp_rand = [random.uniform(roi['temperature'][0], roi['temperature'][1]) for _ in range(num_inputs)]
    vmJ2_rand = [random.uniform(roi['vmJ2'][0], roi['vmJ2'][1]) for _ in range(num_inputs)]

    # create an input deck: more than one input deck can be provided, each in a new key input[0], input[1], etc.
    input = {}
    for i in range(num_inputs):
            input[i] = {'vmJ2' :[vmJ2_rand[i]],
                    'temperature' : [temp_rand[i]],
                    'rhoc' : [rhoc_rand[i]],
                    'rhow' : [rhow_rand[i]],
                    'evm' : [evm_rand[i]],
                    'dt' : np.ones(2)*1e-1,
                    't' : [0.0],
                    }


    # some additional simulation specs
    args = {'map_output' : True,
            "output_directory" : output_directory,
            'max_length' : None,
            'concat' : False,
            'force_rhoc' : True,
            'force_rhow' : True,
            'force_evm' : True,
            'force_rhoc_end_point' : 3e-1,
            'force_evm_end_point' : 3e-1,
            'adaptive_time' : False,
            'debug' : False
            }

    # run a creep simulation
    creep_output = smb.run_creep(input, SM, sim_range=range(len(input)), args=args)

    ## ============================
    ## STORE RESULT IN .CSV FILE
    ## ----------------------------
    # Define headers
    csv_headers = ['time', 'rhom_in', 'rhoi_in', 'vmJ2_in', 'evm_in', 'temperature_in', 'dt_in', 'rhom_rate_out', 'creep_rate_out']

    # raise IOError("debug")
    # Fill the csv_data from the creep_output dict, define time steps and time
    csv_data = []
    # csv_dt = list(np.ones(num_inputs)*0.1)
    csv_time = 0.0
    csv_dt = 0.1
    for i in range(len(creep_output)):
        # if creep_output[i]['dgedt'][-1] > 0.0:
        if creep_output[i]['U']['evm'][-1] > 0.0 and np.abs(creep_output[i]['U']['evm'][-1]) != np.inf:
            if input[i]['vmJ2'][-1] > 2000:
                raise IOError("debug")
            
            # csv_data.append([csv_time, input[i]['rhoc'][0], input[i]['rhow'][0], input[i]['vmJ2'][-1], input[i]['evm'][-1], input[i]['temperature'][-1], input[i]['prec_dens'][-1], csv_dt, creep_output[i]['drhocdt'][-1], creep_output[i]['drhowdt'][-1], creep_output[i]['dgedt'][-1]
            #            ])
            if np.abs(creep_output[i]['U']['rhoc'][-1]) < 1e-300:
                print("Subnormal value in rhoc output: {} normalized to 0.0".format(creep_output[i]['U']['rhoc'][-1]))
                creep_output[i]['U']['rhoc'][-1] = 0.0
                
            csv_data.append([csv_time, input[i]['rhoc'][0], input[i]['vmJ2'][-1], input[i]['evm'][-1], input[i]['temperature'][-1], csv_dt, creep_output[i]['U']['rhoc'][-1], creep_output[i]['U']['evm'][-1]
                       ])
            csv_time+=csv_dt
            # print("\nStrain rate: {:e}, sim: {}.".format(creep_output[i]['dgedt'][-1], i))
            print("\nStrain rate: {:e}, sim: {}.".format(creep_output[i]['U']['evm'][-1], i))
        else:

            # print("(Less than) zero output detected: excluded from verification input file. \nStrain rate: {:e}, sim: {}.".format(creep_output[i]['dgedt'][-1], i))
            if creep_output[i]['U']['evm'][-1] < 0.0:
                print("Negative strain rate output detected: excluded from verification input file. \nStrain rate: {:e}, sim: {}.\nInput stress: {}.\nInput temperature: {}".format(creep_output[i]['U']['evm'][-1], i, creep_output[i]['vmJ2'][-1], creep_output[i]['temperature'][-1]))
            elif np.abs(creep_output[i]['U']['evm'][-1]) == np.inf:
                print("Infinite strain rate output detected: excluded from verification input file. \nStrain rate: {:e}, sim: {}.".format(creep_output[i]['U']['evm'][-1], i))

    # Desired formatting: scientific notation with 15 decimals
    formatted_data = [
        [f"{value:.18e}" if isinstance(value, (float, int)) else value for value in row]
        for row in csv_data
    ]

    # raise IOError("debug")

    # Create the directories if they don't exist
    if output_directory != None:
        os.makedirs(output_directory, exist_ok=True)
        # Create the file path
        file_path = os.path.join(output_directory, output_filename)

        # Open the file and write the headers and data
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the headers as the first row
            writer.writerow(csv_headers)
            
            # Write the data rows
            writer.writerows(formatted_data)
    
    return csv_data

# OLD only for quads
def add_temperature_extrapolation_element(SM, new_lower_bound_temperature= 300.0, evm_output=1e-14):
    """
    Adds an element to an existing mesh as stored in SM dictionary.

    Inputs;
    - SM : dict
        Dictionary containing surrogate model, including mesh.
    - new_upper_bound_vmJ2: float
        New upper bound for stress in MPa
    - evm_output: float 
        new strain rate output in physiscal units

    Outputs:
    - SM : dict
        Dictionary containing surrogate model with appended mesh

    """
    # Check if mesh exists
    if 'mesh' not in SM.keys():
        return None
    else:

        # Define a helper function for comparing values within a tolerance
        def is_close(value1, value2, tol=1e-9):
            return np.abs(value1 - value2) < tol

        # Find the node-numbers at max stress boundary of SM ROI:
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])   # physical units

        for k in range(2):
            for l in range(2):
                for m in range(2):                    
                    roi_x = SM['roi']['vmJ2'][m]
                    roi_y = SM['roi']['temperature'][0]
                    roi_z = SM['roi']['evm'][l]
                    roi_w = SM['roi']['rhoc'][k]

                    indices = np.where((is_close(nodes[:,0], roi_x)) &
                                    (is_close(nodes[:,1], roi_y)) &
                                    (is_close(nodes[:,2], roi_z)) &
                                    (is_close(nodes[:,3], roi_w, tol=1e2)))[0] # & # larger tolerance, due to larger values for rhoc, in the order of O(10^12)

                    if indices.size > 0:
                        boundary_nodes.append(list(indices)[0])
                            
        # Add new nodal coordinates in stress dimension

        added_nodal_coordinates = copy.deepcopy(nodes[boundary_nodes])                              # nodal coordinate should here be in physical units 
        added_nodal_coordinates[:,1] = new_lower_bound_temperature                                  # change coordinate for temperature to new coordinate
        added_nodal_coordinates = smd.transform_nodes(added_nodal_coordinates, SM['input_maps'])
        new_nodes = np.concatenate((SM['mesh']['nodes'], added_nodal_coordinates), axis=0)          # add new node coordinates to node dict in SM 

        new_elem = SM['mesh']['conn'][0].copy()
        new_elem[0] = SM['mesh']['nodes'].shape[0] - 1  # index of the new node
        new_conn = np.vstack([SM['mesh']['conn'], new_elem])

        # Add value to new nodes for all outputs (evm, rhoc, rhom)
        
        added_nodal_values_evm = np.ones(len(SM['nodal_values']['evm'][boundary_nodes])) * SM['output_maps']['evm'].transform(evm_output) # make it some value (no addition)
        new_nodal_values_evm = np.concatenate((SM['nodal_values']['evm'], added_nodal_values_evm), axis=0)
        added_nodal_values_rhoc = SM['nodal_values']['rhoc'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhoc = np.concatenate((SM['nodal_values']['rhoc'], added_nodal_values_rhoc), axis=0)

        # change SM
        if 'extrapolation_element_ids' in SM['mesh_specs'].keys() and len(SM['mesh_specs']['extrapolation_element_ids']) > 0:
            SM['mesh_specs']['extrapolation_element_ids'].append(len(SM['mesh']['conn']))
        else:
            SM['mesh_specs']['extrapolation_element_ids'] = [len(SM['mesh']['conn'])]
        SM['roi']['temperature'][0] = new_lower_bound_temperature
        SM['mesh']['nodes'] = new_nodes
        SM['mesh']['conn'] = new_conn
        SM['nodal_values']['evm'] = new_nodal_values_evm
        SM['nodal_values']['rhoc'] = new_nodal_values_rhoc

            
        return SM

def add_temperature_extrapolation_6d_element(SM, new_lower_bound_temperature= 300.0, evm_output=1e-14):
    """
    Adds an element to an existing mesh as stored in SM dictionary.

    Inputs;
    - SM : dict
        Dictionary containing surrogate model, including mesh.
    - new_upper_bound_vmJ2: float
        New upper bound for stress in MPa
    - evm_output: float 
        new strain rate output in physiscal units

    Outputs:
    - SM : dict
        Dictionary containing surrogate model with appended mesh

    """
    # Check if mesh exists
    if 'mesh' not in SM.keys():
        return None
    else:

        # Define a helper function for comparing values within a tolerance
        def is_close(value1, value2, tol=1e-9):
            return np.abs(value1 - value2) < tol

        # Find the node-numbers at max stress boundary of SM ROI:
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])   # physical units
        print(nodes[:,1].min())
        print(nodes[:,1].max())
        print(SM['roi']['temperature'][0])
        for i in range(2):
          for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):                    
                        roi_x = SM['roi']['vmJ2'][m]
                        roi_y = SM['roi']['temperature'][0]
                        roi_z = SM['roi']['evm'][l]
                        roi_w = SM['roi']['rhoc'][k]
                        roi_v = SM['roi']['rhow'][j]
                        roi_u = SM['roi']['defect_rate'][i]
                        # roi_u = SM['roi']['flux'][i]

                        indices = np.where((is_close(nodes[:,0], roi_x, tol=1e-2)) &
                                        (is_close(nodes[:,1], roi_y, tol=1e-2)) &
                                        (is_close(nodes[:,2], roi_z, tol=1e-9)) &
                                        (is_close(nodes[:,3], roi_w, tol=1e2)) & # larger tolerance, due to larger values for rhoc, in the order of O(10^12)
                                        (is_close(nodes[:,4], roi_v, tol=1e2)) & # larger tolerance, due to larger values for rhow, in the order of O(10^12)
                                        (is_close(nodes[:,5], roi_u, tol=1e-9)))[0] # larger tolerance, due to larger values for prec_dens, in the order of O(10^20)

                        if indices.size > 0:
                            boundary_nodes.append(list(indices)[0])

        print("Boundary nodes identified: {}, count: {}".format(boundary_nodes, len(boundary_nodes)))
        # Add new nodal coordinates in stress dimension
        # raise IOError("debug")
        added_nodal_coordinates = copy.deepcopy(nodes[boundary_nodes])                              # nodal coordinate should here be in physical units 
        if len(added_nodal_coordinates) == 0:
            raise IOError("no added nodal coordinates")
        added_nodal_coordinates[:,1] = new_lower_bound_temperature                                  # change coordinate for temperature to new coordinate
        added_nodal_coordinates = smd.transform_nodes(added_nodal_coordinates, SM['input_maps'])
        new_nodes = np.concatenate((SM['mesh']['nodes'], added_nodal_coordinates), axis=0)          # add new node coordinates to node dict in SM 

        # Append element to connectivity matrix
        # Create a new element connectivity by copying the first existing element
        # and substituting its first node index with the newly added node.
        new_elem = SM['mesh']['conn'][0].copy()
        new_elem[0] = SM['mesh']['nodes'].shape[0] - 1  # index of the new node
        new_conn = np.vstack([SM['mesh']['conn'], new_elem])

        # Add value to new nodes for all outputs (evm, rhoc, rhom)
        
        added_nodal_values_evm = np.ones(len(SM['nodal_values']['evm'][boundary_nodes])) * SM['output_maps']['evm'].transform(evm_output) # make it some value (no addition)
        new_nodal_values_evm = np.concatenate((SM['nodal_values']['evm'], added_nodal_values_evm), axis=0)
        added_nodal_values_rhoc = SM['nodal_values']['rhoc'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhoc = np.concatenate((SM['nodal_values']['rhoc'], added_nodal_values_rhoc), axis=0)
        added_nodal_values_rhow = SM['nodal_values']['rhow'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhow = np.concatenate((SM['nodal_values']['rhow'], added_nodal_values_rhow), axis=0)

        # change SM
        if 'extrapolation_element_ids' in SM['mesh_specs'].keys() and len(SM['mesh_specs']['extrapolation_element_ids']) > 0:
            SM['mesh_specs']['extrapolation_element_ids'].append(len(SM['mesh']['conn']))
        else:
            SM['mesh_specs']['extrapolation_element_ids'] = [len(SM['mesh']['conn'])]
        SM['roi']['temperature'][0] = new_lower_bound_temperature
        SM['mesh']['nodes'] = new_nodes
        SM['mesh']['conn'] = new_conn
        SM['nodal_values']['evm'] = new_nodal_values_evm
        SM['nodal_values']['rhoc'] = new_nodal_values_rhoc
        SM['nodal_values']['rhow'] = new_nodal_values_rhow
            
        return SM

def add_stress_extrapolation_element(SM, new_upper_bound_vmJ2=2000.0, evm_output=1e3):
    """
    Adds an element to an existing mesh as stored in SM dictionary.

    Inputs;
    - SM : dict
        Dictionary containing surrogate model, including mesh.
    - new_upper_bound_vmJ2: float
        New upper bound for stress in MPa
    - addition_evm_output: float 
        Upper bound for strain rate in log10 space

    Outputs:
    - SM : dict
        Dictionary containing surrogate model with appended mesh

    """
    # Check if mesh exists
    if 'mesh' not in SM.keys():
        return None
    else:

        # Define a helper function for comparing values within a tolerance
        def is_close(value1, value2, tol=1e-9):
            return np.abs(value1 - value2) < tol

        # Find the node numbers at the max boundary in the vmJ2 dimension
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])  # Physical units

        # Find the node-numbers at max stress boundary of SM ROI:
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])   # physical units

        # for i in range(2):
        #   for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):                    
                    roi_x = SM['roi']['vmJ2'][1]
                    # roi_y = SM['roi']['temperature'][m]
                    if m==0:
                        roi_y = nodes[np.argmin(np.abs(nodes[:,1] - 293))][1] # manual
                    elif m==1:
                        roi_y = nodes[np.argmin(np.abs(nodes[:,1] - 1000))][1] # manual
                    roi_z = SM['roi']['evm'][l]
                    roi_w = SM['roi']['rhoc'][k]

                    indices = np.where((is_close(nodes[:,0], roi_x)) &
                                        (is_close(nodes[:,1], roi_y)) &
                                        (is_close(nodes[:,2], roi_z)) &
                                        (is_close(nodes[:,3], roi_w, tol=1e2)))[0] # & # larger tolerance, due to larger values for rhoc, in the order of O(10^12)
                                        # (is_close(nodes[:,4], roi_v, tol=1e2)) & # larger tolerance, due to larger values for rhow, in the order of O(10^12)
                                        # (is_close(nodes[:,5], roi_u, tol=1e2)))[0] # larger tolerance, due to larger values for prec_dens, in the order of O(10^20)

                    if indices.size > 0:
                        boundary_nodes.append(list(indices)[0])
                            
        # Add new nodal coordinates in stress dimension
        # raise IOError("debug")
        added_nodal_coordinates = copy.deepcopy(nodes[boundary_nodes])  # nodal coordinate should here be in physical units 
        added_nodal_coordinates[:,0] = new_upper_bound_vmJ2             
        added_nodal_coordinates = smd.transform_nodes(added_nodal_coordinates, SM['input_maps'])
        new_nodes = np.concatenate((SM['mesh']['nodes'], added_nodal_coordinates), axis=0)   # add new node coordinates to node dict in SM 

        # Append element to connectivity matrix (same logic as above)
        new_elem = SM['mesh']['conn'][0].copy()
        new_elem[0] = SM['mesh']['nodes'].shape[0] - 1
        new_conn = np.vstack([SM['mesh']['conn'], new_elem])

        # Add value to new nodes for all outputs (evm, rhoc, rhom)
        added_nodal_values_evm = np.ones(len(SM['nodal_values']['evm'][boundary_nodes])) * SM['output_maps']['evm'].transform(evm_output) # add some value so that there is a gradient dgedt/dstress
        new_nodal_values_evm = np.concatenate((SM['nodal_values']['evm'], added_nodal_values_evm), axis=0)
        added_nodal_values_rhoc = SM['nodal_values']['rhoc'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhoc = np.concatenate((SM['nodal_values']['rhoc'], added_nodal_values_rhoc), axis=0)
        # raise IOError("debug")
    
        # change SM
        if 'extrapolation_element_ids' in SM['mesh_specs'].keys() and len(SM['mesh_specs']['extrapolation_element_ids']) > 0:
            SM['mesh_specs']['extrapolation_element_ids'].append(len(SM['mesh']['conn']))
        else:
            SM['mesh_specs']['extrapolation_element_ids'] = [len(SM['mesh']['conn'])]
        SM['roi']['vmJ2'][1] = new_upper_bound_vmJ2
        SM['mesh']['nodes'] = new_nodes
        SM['mesh']['conn'] = new_conn
        SM['nodal_values']['evm'] = new_nodal_values_evm
        SM['nodal_values']['rhoc'] = new_nodal_values_rhoc
            
        return SM
    
def add_stress_extrapolation_6d_element(SM, new_upper_bound_vmJ2=2000.0, evm_output=1e3):
    """
    Adds an element to an existing mesh as stored in SM dictionary.

    Inputs;
    - SM : dict
        Dictionary containing surrogate model, including mesh.
    - new_upper_bound_vmJ2: float
        New upper bound for stress in MPa
    - addition_evm_output: float 
        Upper bound for strain rate in log10 space

    Outputs:
    - SM : dict
        Dictionary containing surrogate model with appended mesh

    """
    # Check if mesh exists
    if 'mesh' not in SM.keys():
        return None
    else:

        # Define a helper function for comparing values within a tolerance
        def is_close(value1, value2, tol=1e-9):
            return np.abs(value1 - value2) < tol

        # Find the node numbers at the max boundary in the vmJ2 dimension
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])  # Physical units

        # Find the node-numbers at max stress boundary of SM ROI:
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])   # physical units

        # Identify all nodes on the max‑stress face (upper vmJ2 bound) across the remaining five dimensions.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            roi_x = SM['roi']['vmJ2'][1]
                            roi_y = SM['roi']['temperature'][m]
                            roi_z = SM['roi']['evm'][l]
                            roi_w = SM['roi']['rhoc'][k]
                            roi_v = SM['roi']['rhow'][j]
                            roi_u = SM['roi']['defect_rate'][i]
                            # roi_u = SM['roi']['flux'][i]

                            indices = np.where(
                                (is_close(nodes[:, 0], roi_x, tol=1e0)) &
                                (is_close(nodes[:, 1], roi_y, tol=1e0)) &
                                (is_close(nodes[:, 2], roi_z, tol=1e-9)) &
                                (is_close(nodes[:, 3], roi_w, tol=1e2)) &  # larger tolerance for rhoc (~1e12)
                                (is_close(nodes[:, 4], roi_v, tol=1e2)) &  # larger tolerance for rhow (~1e12)
                                (is_close(nodes[:, 5], roi_u, tol=1e-9)))[0]
                            if indices.size > 0:
                                boundary_nodes.append(int(indices[0]))
                            
        # Add new nodal coordinates in stress dimension
        # raise IOError("debug")
        added_nodal_coordinates = copy.deepcopy(nodes[boundary_nodes])  # nodal coordinate should here be in physical units 
        added_nodal_coordinates[:,0] = new_upper_bound_vmJ2             
        added_nodal_coordinates = smd.transform_nodes(added_nodal_coordinates, SM['input_maps'])
        new_nodes = np.concatenate((SM['mesh']['nodes'], added_nodal_coordinates), axis=0)   # add new node coordinates to node dict in SM 

        # Append element to connectivity matrix
        added_conn = np.concatenate((np.array(boundary_nodes), np.arange(len(SM['mesh']['nodes']), len(new_nodes))), axis=0)
        new_conn = np.concatenate((SM['mesh']['conn'], added_conn[np.newaxis,:]), axis=0)

        # Add value to new nodes for all outputs (evm, rhoc, rhom)
        added_nodal_values_evm = np.ones(len(SM['nodal_values']['evm'][boundary_nodes])) * SM['output_maps']['evm'].transform(evm_output) # add some value so that there is a gradient dgedt/dstress
        new_nodal_values_evm = np.concatenate((SM['nodal_values']['evm'], added_nodal_values_evm), axis=0)
        added_nodal_values_rhoc = SM['nodal_values']['rhoc'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhoc = np.concatenate((SM['nodal_values']['rhoc'], added_nodal_values_rhoc), axis=0)
        added_nodal_values_rhow = SM['nodal_values']['rhow'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhow = np.concatenate((SM['nodal_values']['rhow'], added_nodal_values_rhow), axis=0)

        # raise IOError("debug")
    
        # change SM
        if 'extrapolation_element_ids' in SM['mesh_specs'].keys() and len(SM['mesh_specs']['extrapolation_element_ids']) > 0:
            SM['mesh_specs']['extrapolation_element_ids'].append(len(SM['mesh']['conn']))
        else:
            SM['mesh_specs']['extrapolation_element_ids'] = [len(SM['mesh']['conn'])]
        SM['roi']['vmJ2'][1] = new_upper_bound_vmJ2
        SM['mesh']['nodes'] = new_nodes
        SM['mesh']['conn'] = new_conn
        SM['nodal_values']['evm'] = new_nodal_values_evm
        SM['nodal_values']['rhoc'] = new_nodal_values_rhoc
        SM['nodal_values']['rhow'] = new_nodal_values_rhow
            
        return SM

# OLD only for quads. 4D
def add_temperature_extrapolation_element(SM, new_lower_bound_temperature= 300.0, evm_output=1e-14):
    """
    Adds an element to an existing mesh as stored in SM dictionary.

    Inputs;
    - SM : dict
        Dictionary containing surrogate model, including mesh.
    - new_upper_bound_vmJ2: float
        New upper bound for stress in MPa
    - evm_output: float 
        new strain rate output in physiscal units

    Outputs:
    - SM : dict
        Dictionary containing surrogate model with appended mesh

    """
    # Check if mesh exists
    if 'mesh' not in SM.keys():
        return None
    else:

        # Define a helper function for comparing values within a tolerance
        def is_close(value1, value2, tol=1e0):
            return np.abs(value1 - value2) < tol

        # Find the node-numbers at max stress boundary of SM ROI:
        boundary_nodes = []
        nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps'])   # physical units
        print("nodes", nodes)
        print("ROI", SM['roi'])

        for k in range(2):
            for l in range(2):
                for m in range(2):                    
                    roi_x = SM['roi']['vmJ2'][m]
                    roi_y = SM['roi']['temperature'][0]
                    roi_z = SM['roi']['evm'][l]
                    roi_w = SM['roi']['rhoc'][k]


                    indices = np.where((is_close(nodes[:,0], roi_x)) &
                                    (is_close(nodes[:,1], roi_y)) &
                                    (is_close(nodes[:,2], roi_z)) &
                                    (is_close(nodes[:,3], roi_w, tol=1e2)))[0] # & # larger tolerance, due to larger values for rhoc, in the order of O(10^12)

                    if indices.size > 0:
                        boundary_nodes.append(list(indices)[0])
                            
        # Add new nodal coordinates in stress dimension
        # raise IOError("debug")
        print("boundary_nodes", boundary_nodes)
        added_nodal_coordinates = copy.deepcopy(nodes[boundary_nodes])                              # nodal coordinate should here be in physical units 
        added_nodal_coordinates[:,1] = new_lower_bound_temperature                                  # change coordinate for temperature to new coordinate
        added_nodal_coordinates = smd.transform_nodes(added_nodal_coordinates, SM['input_maps'])
        new_nodes = np.concatenate((SM['mesh']['nodes'], added_nodal_coordinates), axis=0)          # add new node coordinates to node dict in SM 

        # Append element to connectivity matrix
        added_conn = np.concatenate((np.array(boundary_nodes), np.arange(len(SM['mesh']['nodes']), len(new_nodes))), axis=0)
        new_conn = np.concatenate((SM['mesh']['conn'], added_conn[np.newaxis,:]), axis=0)

        added_nodal_values_evm = np.ones(len(SM['nodal_values']['evm'][boundary_nodes])) * SM['output_maps']['evm'].transform(evm_output) # make it some value (no addition)
        new_nodal_values_evm = np.concatenate((SM['nodal_values']['evm'], added_nodal_values_evm), axis=0)
        added_nodal_values_rhoc = SM['nodal_values']['rhoc'][boundary_nodes] # copy values at boundary without per se adding a gradient
        new_nodal_values_rhoc = np.concatenate((SM['nodal_values']['rhoc'], added_nodal_values_rhoc), axis=0)


        # change SM
        if 'extrapolation_element_ids' in SM['mesh_specs'].keys() and len(SM['mesh_specs']['extrapolation_element_ids']) > 0:
            SM['mesh_specs']['extrapolation_element_ids'].append(len(SM['mesh']['conn']))
        else:
            SM['mesh_specs']['extrapolation_element_ids'] = [len(SM['mesh']['conn'])]
        SM['roi']['temperature'][0] = new_lower_bound_temperature
        SM['mesh']['nodes'] = new_nodes
        SM['mesh']['conn'] = new_conn
        SM['nodal_values']['evm'] = new_nodal_values_evm
        SM['nodal_values']['rhoc'] = new_nodal_values_rhoc
            
        return SM
    

def write_moose_h(SM: dict, name : str):
    """ 
    Write MOOSE h-file for HT9 models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for vmJ2, temperature, evm, rhoc, rhow, prec_dens (in that order)
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
   
    """

# Generate the content of the .h-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/************************************************************************************/
#pragma once

#include "LAesStressUpdateBaseNew.h"

template <bool is_ad>
class {0}Templ : public LAesStressUpdateBaseNewTempl<is_ad>
{{
public:
  static InputParameters validParams();

  {0}Templ(const InputParameters & parameters);

protected:
  virtual std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>> getInputTransform() override;
  virtual std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>> getOutputTransform() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getInputLimits() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodes() override;
  virtual std::vector<std::vector<unsigned int>> getConnectivityMatrix() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodalValues() override;
  virtual std::vector<unsigned int> getElementNumbers() override;
  virtual std::vector<unsigned int> getTriangularDimensions() override;
  virtual std::string getShapeFunctionDegree() override;
  virtual std::vector<unsigned int> getExtrapolationElement() override;
  virtual void initializeKeyMaps() override;
}};

typedef {0}Templ<false> {0};
typedef {0}Templ<true> AD{0};

""".format(name)
    return txt

def write_moose_h_cached(SM: dict, name : str):
    """ 
    Write MOOSE h-file for LAROMance models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for vmJ2, temperature, evm, rhoc, rhow, prec_dens (in that order)
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
   
    """

# Generate the content of the .h-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#pragma once

#include "LAROManceStressUpdateBaseUniversal.h"

template <bool is_ad>
class {0}Templ : public LAROManceStressUpdateBaseUniversalTempl<is_ad>
{{
public:
  static InputParameters validParams();

  {0}Templ(const InputParameters & parameters);

protected:
  virtual std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>> getInputTransform() override;
  virtual std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>> getOutputTransform() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getInputLimits() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodes() override;
  virtual std::vector<std::vector<unsigned int>> getConnectivityMatrix() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodalValues() override;
  virtual std::vector<unsigned int> getElementNumbers() override;
  virtual std::vector<unsigned int> getTriangularDimensions() override;
  virtual std::string getShapeFunctionDegree() override;
  virtual std::vector<unsigned int> getExtrapolationElement() override;
  virtual void initializeKeyMaps() override;
  virtual std::vector<Eigen::MatrixXd> getMinValues() const override;
  virtual std::vector<Eigen::MatrixXd> getMaxValues() const override;
}};

typedef {0}Templ<false> {0};
typedef {0}Templ<true> AD{0};

""".format(name)
    return txt

def write_moose_h(SM: dict, name : str):
    """ 
    Write MOOSE h-file for HT9 models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for vmJ2, temperature, evm, rhoc, rhow, prec_dens (in that order)
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
  
    """

# Generate the content of the .h-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/
#pragma once

#include "LAesStressUpdateBaseNew.h"

template <bool is_ad>
class {0}Templ : public LAesStressUpdateBaseNewTempl<is_ad>
{{
public:
  static InputParameters validParams();

  {0}Templ(const InputParameters & parameters);

protected:
  virtual std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>> getInputTransform() override;
  virtual std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>> getOutputTransform() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getInputLimits() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodes() override;
  virtual std::vector<std::vector<unsigned int>> getConnectivityMatrix() override;
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodalValues() override;
  virtual std::vector<unsigned int> getElementNumbers() override;
  virtual std::vector<unsigned int> getTriangularDimensions() override;
  virtual std::string getShapeFunctionDegree() override;
  virtual std::vector<unsigned int> getExtrapolationElement() override;
  virtual void initializeKeyMaps() override;
}};

typedef {0}Templ<false> {0};
typedef {0}Templ<true> AD{0};

""".format(name)
    return txt
    
def write_moose_C(SM: dict, name: str):
    """ 
    Write MOOSE C-file for surrogate models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for surrogate model inputs
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
    """

    # Generate some strings from the surrogate model dict "SM" 
    nodes_rows = ['{' + ', '.join(f"{x:2.18e}" for x in row) + '}' for row in SM['mesh']['nodes']]
    nodes_string = '{\n' + ',\n'.join(nodes_rows) + '\n}'

    conn_rows = ['{' + ', '.join(map(str, row)) + '}' for row in SM['mesh']['conn']]
    conn_string = '{\n' + ',\n'.join(conn_rows) + '\n}'

    nodal_values = copy.deepcopy(SM['nodal_values'])
    nodal_values_combined = ['{' + ', '.join(f"{x:2.18e}" for x in nodal_values[key]) + '}'for key in ['evm', 'rhoc']]
    nodal_values_string = '{\n' + ',\n'.join(map(str, nodal_values_combined)) + '\n}'

    shape_func_degree = SM['mesh_specs']['shapefunc_degree']
    
    # Handle extrapolation element IDs
    if 'extrapolation_element_ids' in SM['mesh_specs']:
        extrapolation_element = SM['mesh_specs']['extrapolation_element_ids']
        extrapolation_element_string = '{' + ', '.join(map(str, extrapolation_element)) + '}'
    else:
        extrapolation_element_string = '{}'

    # Get triangular dimensions if available
    triangular_dimensions = []
    if 'tri_elements' in SM['mesh_specs']:
        tri_elements = SM['mesh_specs']['tri_elements']
        
        # Extract the actual dimension indices
        if isinstance(tri_elements, list):
            # For a list, use the indices of elements that are True or 1
            for i, value in enumerate(tri_elements):
                if value:  # This will work for both boolean True and integer 1
                    triangular_dimensions.append(i)
        elif isinstance(tri_elements, dict):
            # For a dictionary, the keys might be dimension names and values might be booleans
            # or the values themselves might be the dimension indices
            for i, (key, value) in enumerate(tri_elements.items()):
                if isinstance(value, bool) or isinstance(value, int):
                    if value:  # If True or non-zero
                        triangular_dimensions.append(i)
                else:
                    # If the value is something else, assume it's the dimension index
                    triangular_dimensions.append(value)

    triangular_dimensions_string = '{' + ', '.join(map(str, triangular_dimensions)) + '}'


    # Element numbers
    MOOSE_input_keys = ["vmJ2", "temperature", "evm", "rhoc"]
    element_numbers = np.zeros(len(MOOSE_input_keys), int)
    for i, key in enumerate(MOOSE_input_keys):
        if isinstance(SM['mesh_specs']['element_numbers'][key], list): 
            element_numbers[i] = len(SM['mesh_specs']['element_numbers'][key]) - 1
        elif isinstance(SM['mesh_specs']['element_numbers'][key], int):
            element_numbers[i] = SM['mesh_specs']['element_numbers'][key]

    element_numbers_string = '{' + ', '.join(map(str, element_numbers)) + '}' 

    # Input limits based on ROI
    input_limits = ['{' + ', '.join(f"{value:2.18e}" for value in np.array(SM['roi'][key])) + '}' for key in MOOSE_input_keys]
    input_limits_string = '{\n' + ',\n'.join(map(str, input_limits)) + '\n}'
    rhoc_min = str(min(SM['roi']['rhoc']))
    rhoc_max = str(max(SM['roi']['rhoc']))
    rhoc_median = str(np.median([float(rhoc_min), float(rhoc_max)]))

    # Input scalers
    input_scalers = []
    for key in MOOSE_input_keys:
        if 'Compress' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"COMPRESS\", typename LAesStressUpdateBaseNewTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].compressor, SM['input_maps'][key].original_min))
        elif 'LogScaler' in str(SM['input_maps'][key]) and 'sym' not in str(SM['input_maps']):
            input_scalers.append("{{\"LOG10BOUNDED\", typename LAesStressUpdateBaseNewTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].b, SM['input_maps'][key].a, SM['input_maps'][key].logmin, SM['input_maps'][key].logmax))
        elif 'MinMax' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"MINMAX\", typename LAesStressUpdateBaseNewTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].data_min_[0], SM['input_maps'][key].data_max_[0], SM['input_maps'][key].feature_range[0], SM['input_maps'][key].feature_range[1]))

    input_scalers_string = '{' + ',\n'.join(input_scalers) + '\n}'

    # Output scalers
    MOOSE_output_keys = ['evm', 'rhoc']
    output_scalers = []
    for key in MOOSE_output_keys:
        if 'Compress' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"DECOMPRESS\", typename LAesStressUpdateBaseNewTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].compressor, SM['output_maps'][key].original_min))
        elif 'LogScaler' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"EXP10BOUNDED\", typename LAesStressUpdateBaseNewTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].b, SM['output_maps'][key].a, SM['output_maps'][key].logmin, SM['output_maps'][key].logmax))
        elif 'MinMax' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"INVMINMAX\", typename LAesStressUpdateBaseNewTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].data_min_[0], SM['output_maps'][key].data_max_[0], SM['output_maps'][key].feature_range[0], SM['output_maps'][key].feature_range[1]))

    output_scalers_string = '{' + ',\n'.join(output_scalers) + '\n}'

    # Generate the content of the .C-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "{0}.h"

registerMooseObject("PolecatApp", {0});
registerADMooseObject("PolecatApp", AD{0});

template <bool is_ad>
InputParameters
{0}Templ<is_ad>::validParams()
{{
  InputParameters params = LAesStressUpdateBaseNewTempl<is_ad>::validParams();
  params.addClassDescription("LAes creep update model for {0}");

  // Override defaults for material specific parameters below
  params.addRangeCheckedParam<GenericReal<is_ad>>("initial_cell_dislocation_density",
                                    1.0e12,
                                    "initial_cell_dislocation_density >= {2} &"
                                    "initial_cell_dislocation_density <= {3}",
                                    "Initial density of mobile (glissile) dislocations (1/m^2).");

  params.addRangeCheckedParam<GenericReal<is_ad>>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of mobile (glissile) dislocations.");

  return params;
}}

template <bool is_ad>
{0}Templ<is_ad>::{0}Templ(
    const InputParameters &parameters)
    : LAesStressUpdateBaseNewTempl<is_ad>(parameters)
{{
}}

template <bool is_ad>
void
{0}Templ<is_ad>::initializeKeyMaps()
{{
  // First call the base class implementation to set up default mappings
  LAesStressUpdateBaseNewTempl<is_ad>::initializeKeyMaps();
  
  // Override with specific ordering for this material
  // Input keys
  this->setInputIndexByKey("stress", 0);
  this->setInputIndexByKey("temperature", 1);
  this->setInputIndexByKey("old_strain", 2);
  this->setInputIndexByKey("cell", 3);

  // Output keys
  this->setOutputIndexByKey("strain", 0);
  this->setOutputIndexByKey("cell", 1);
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getElementNumbers()
{{
  return {1};
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getTriangularDimensions()
{{
  return {4};
}}

template <bool is_ad>
std::string
{0}Templ<is_ad>::getShapeFunctionDegree()
{{
  return "{5}";
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getExtrapolationElement()
{{
  return {6};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getInputTransform()
{{
  return {7};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getOutputTransform()
{{
  return {8};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getInputLimits()
{{
  return {9};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodes()
{{
  return {10};
}}

template <bool is_ad>
std::vector<std::vector<unsigned int>>
{0}Templ<is_ad>::getConnectivityMatrix()
{{
  return {11};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodalValues()
{{
  return {12};
}}
""".format(name, element_numbers_string, rhoc_min, rhoc_max, 
           triangular_dimensions_string, shape_func_degree, 
           extrapolation_element_string, input_scalers_string, 
           output_scalers_string, input_limits_string, 
           nodes_string, conn_string, nodal_values_string)

    return txt

def write_moose_C_4D_cached(SM: dict, name: str):
    """ 
    Write MOOSE C-file for surrogate models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for surrogate model inputs
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
    """

    # Generate some strings from the surrogate model dict "SM" 
    nodes_rows = ['{' + ', '.join(f"{x:2.18e}" for x in row) + '}' for row in SM['mesh']['nodes']]
    nodes_string = '{\n' + ',\n'.join(nodes_rows) + '\n}'

    conn_rows = ['{' + ', '.join(map(str, row)) + '}' for row in SM['mesh']['conn']]
    conn_string = '{\n' + ',\n'.join(conn_rows) + '\n}'

    nodal_values = copy.deepcopy(SM['nodal_values'])
    nodal_values_combined = ['{' + ', '.join(f"{x:2.18e}" for x in nodal_values[key]) + '}'for key in ['evm', 'rhoc']]
    nodal_values_string = '{\n' + ',\n'.join(map(str, nodal_values_combined)) + '\n}'

    shape_func_degree = SM['mesh_specs']['shapefunc_degree']
    
    # Handle extrapolation element IDs
    if 'extrapolation_element_ids' in SM['mesh_specs']:
        extrapolation_element = SM['mesh_specs']['extrapolation_element_ids']
        extrapolation_element_string = '{' + ', '.join(map(str, extrapolation_element)) + '}'
    else:
        extrapolation_element_string = '{}'

    # Get triangular dimensions if available
    triangular_dimensions = []
    if 'tri_elements' in SM['mesh_specs']:
        tri_elements = SM['mesh_specs']['tri_elements']
        
        # Extract the actual dimension indices
        if isinstance(tri_elements, list):
            # For a list, use the indices of elements that are True or 1
            for i, value in enumerate(tri_elements):
                if value:  # This will work for both boolean True and integer 1
                    triangular_dimensions.append(i)
        elif isinstance(tri_elements, dict):
            # For a dictionary, the keys might be dimension names and values might be booleans
            # or the values themselves might be the dimension indices
            for i, (key, value) in enumerate(tri_elements.items()):
                if isinstance(value, bool) or isinstance(value, int):
                    if value:  # If True or non-zero
                        triangular_dimensions.append(i)
                else:
                    # If the value is something else, assume it's the dimension index
                    triangular_dimensions.append(value)

    triangular_dimensions_string = '{' + ', '.join(map(str, triangular_dimensions)) + '}'


    # Element numbers
    MOOSE_input_keys = ["vmJ2", "temperature", "evm", "rhoc"]
    element_numbers = np.zeros(len(MOOSE_input_keys), int)
    for i, key in enumerate(MOOSE_input_keys):
        if isinstance(SM['mesh_specs']['element_numbers'][key], list): 
            element_numbers[i] = len(SM['mesh_specs']['element_numbers'][key]) - 1
        elif isinstance(SM['mesh_specs']['element_numbers'][key], int):
            element_numbers[i] = SM['mesh_specs']['element_numbers'][key]

    element_numbers_string = '{' + ', '.join(map(str, element_numbers)) + '}' 

    # Input limits based on ROI
    input_limits = ['{' + ', '.join(f"{value:2.18e}" for value in np.array(SM['roi'][key])) + '}' for key in MOOSE_input_keys]
    input_limits_string = '{\n' + ',\n'.join(map(str, input_limits)) + '\n}'
    rhoc_min = str(min(SM['roi']['rhoc']))
    rhoc_max = str(max(SM['roi']['rhoc']))
    rhoc_min = str(min(SM['roi']['rhoc']))
    rhoc_max = str(max(SM['roi']['rhoc']))

    # Input scalers
    input_scalers = []
    for key in MOOSE_input_keys:
        if 'Compress' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"COMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].compressor, SM['input_maps'][key].original_min))
        elif 'LogScaler' in str(SM['input_maps'][key]) and 'sym' not in str(SM['input_maps']):
            input_scalers.append("{{\"LOG10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].b, SM['input_maps'][key].a, SM['input_maps'][key].logmin, SM['input_maps'][key].logmax))
        elif 'MinMax' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"MINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].data_min_[0], SM['input_maps'][key].data_max_[0], SM['input_maps'][key].feature_range[0], SM['input_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"SYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].zmin, SM['input_maps'][key].zmax, SM['input_maps'][key].zbar, SM['input_maps'][key].lowerbound, SM['input_maps'][key].upperbound))


    input_scalers_string = '{' + ',\n'.join(input_scalers) + '\n}'

    # Output scalers
    MOOSE_output_keys = ['evm', 'rhoc']
    output_scalers = []
    for key in MOOSE_output_keys:
        if 'Compress' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"DECOMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].compressor, SM['output_maps'][key].original_min))
        elif 'LogScaler' in str(SM['output_maps'][key]) and 'Sym' not in str(SM['output_maps'][key]) :
            output_scalers.append("{{\"EXP10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].b, SM['output_maps'][key].a, SM['output_maps'][key].logmin, SM['output_maps'][key].logmax))
        elif 'MinMax' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"INVMINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].data_min_[0], SM['output_maps'][key].data_max_[0], SM['output_maps'][key].feature_range[0], SM['output_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"EXPSYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].zmin, SM['output_maps'][key].zmax, SM['output_maps'][key].zbar, SM['output_maps'][key].lowerbound, SM['output_maps'][key].upperbound))

    output_scalers_string = '{' + ',\n'.join(output_scalers) + '\n}'

    # Element boundaries
    num_inputs = len(SM['roi'])
    num_elements = len(SM['mesh']['conn'])
    nodal_coordinates = copy.deepcopy(SM['mesh']['nodes'])
    # nodal_coordinates_physical = np.around(smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps']), decimals=30)
    minValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    maxValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    # Mind the ordering here: in the newer versions we use initializeKeyMaps in the material .C and .h files. In older versions, we will need to use some dict here, like MOOSE_index_convertor = [2, 4, 3, 0, 1, 5] and then use enumerate in the following loop:
    for i in range(nodal_coordinates.shape[1]):
        for element in range(num_elements):
            minValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].min()
            maxValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].max() 
    
    minValues_string = []
    maxValues_string = []
    for i in range(num_inputs):
        minValues_string.append("minValues[{0}] = Eigen::MatrixXd(1, {1});\nminValues[{0}] << ".format(i, num_elements))
        maxValues_string.append("maxValues[{0}] = Eigen::MatrixXd(1, {1});\nmaxValues[{0}] << ".format(i, num_elements))
        
        # Format min and max values with scientific notation and 18 decimal places
        min_values_formatted = ", ".join(f"{x:2.18e}" for x in minValues[i])
        max_values_formatted = ", ".join(f"{x:2.18e}" for x in maxValues[i])
        
        # Append the formatted values
        minValues_string[-1] += min_values_formatted + ";"
        maxValues_string[-1] += max_values_formatted + ";"

        # Join all parts into a single string
        minValues_code = "\n".join(minValues_string)
        maxValues_code = "\n".join(maxValues_string)


    # Generate the content of the .C-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "{0}.h"

registerMooseObject("BisonApp", {0});
registerADMooseObject("BisonApp", AD{0});

template <bool is_ad>
InputParameters
{0}Templ<is_ad>::validParams()
{{
  InputParameters params = LAROManceStressUpdateBaseUniversalTempl<is_ad>::validParams();
  params.addClassDescription("LAes creep update model for {0}");

  // Override defaults for material specific parameters below
  params.addRangeCheckedParam<GenericReal<is_ad>>("initial_cell_dislocation_density",
                                    1.0e12,
                                    "initial_cell_dislocation_density >= {2} &"
                                    "initial_cell_dislocation_density <= {3}",
                                    "Initial density of mobile (glissile) dislocations (1/m^2).");

  params.addRangeCheckedParam<GenericReal<is_ad>>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of mobile (glissile) dislocations.");

  return params;
}}

template <bool is_ad>
{0}Templ<is_ad>::{0}Templ(
    const InputParameters &parameters)
    : LAROManceStressUpdateBaseUniversalTempl<is_ad>(parameters)
{{
}}

template <bool is_ad>
void
{0}Templ<is_ad>::initializeKeyMaps()
{{
  // First call the base class implementation to set up default mappings
  LAROManceStressUpdateBaseUniversalTempl<is_ad>::initializeKeyMaps();
  
  // Override with specific ordering for this material
  // Input keys
  this->setInputIndexByKey("stress", 0);
  this->setInputIndexByKey("temperature", 1);
  this->setInputIndexByKey("old_strain", 2);
  this->setInputIndexByKey("cell", 3);

  // Output keys
  this->setOutputIndexByKey("strain", 0);
  this->setOutputIndexByKey("cell", 1);
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getElementNumbers()
{{
  return {1};
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getTriangularDimensions()
{{
  return {4};
}}

template <bool is_ad>
std::string
{0}Templ<is_ad>::getShapeFunctionDegree()
{{
  return "{5}";
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getExtrapolationElement()
{{
  return {6};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getInputTransform()
{{
  return {7};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getOutputTransform()
{{
  return {8};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getInputLimits()
{{
  return {9};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodes()
{{
  // Store raw doubles first (safe), then convert to GenericReal<is_ad>
  static const std::vector<std::vector<double>> nodes_raw = {10};
  std::vector<std::vector<GenericReal<is_ad>>> result;
  result.reserve(nodes_raw.size());

  for (const auto& row : nodes_raw)
  {{
    std::vector<GenericReal<is_ad>> converted;
    converted.reserve(row.size());
    for (double v : row)
      converted.emplace_back(v);
    result.emplace_back(std::move(converted));
  }}
  return result;
}}

template <bool is_ad>
std::vector<std::vector<unsigned int>>
{0}Templ<is_ad>::getConnectivityMatrix()
{{
  return {11};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodalValues()
{{
  static const std::vector<std::vector<double>> nodal_values_raw = {12};
  
  std::vector<std::vector<GenericReal<is_ad>>> result;
  result.reserve(nodal_values_raw.size());

  for (const auto& row : nodal_values_raw)
  {{
    std::vector<GenericReal<is_ad>> converted;
    converted.reserve(row.size());
    for (double v : row)
      converted.emplace_back(v);
    result.emplace_back(std::move(converted));
  }}
  return result;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMinValues() const
{{
  // Container for minValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> minValues({13});

  {14}

  return minValues;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMaxValues() const
{{
  // Container for maxValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> maxValues({13});

  {15}
  
  return maxValues;
}}
""".format(name, element_numbers_string, rhoc_min, rhoc_max, 
           triangular_dimensions_string, shape_func_degree, 
           extrapolation_element_string, input_scalers_string, 
           output_scalers_string, input_limits_string, 
           nodes_string, conn_string, nodal_values_string, num_inputs, minValues_code, maxValues_code)

    return txt

def write_moose_C_5D_cached(SM: dict, name: str):
    """ 
    Write MOOSE C-file for surrogate models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for surrogate model inputs
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
    """

    MOOSE_output_keys = list(SM['nodal_values'].keys())

    # Generate some strings from the surrogate model dict "SM" 
    nodes_rows = ['{' + ', '.join(f"{x:2.18e}" for x in row) + '}' for row in SM['mesh']['nodes']]
    nodes_string = '{\n' + ',\n'.join(nodes_rows) + '\n}'

    conn_rows = ['{' + ', '.join(map(str, row)) + '}' for row in SM['mesh']['conn']]
    conn_string = '{\n' + ',\n'.join(conn_rows) + '\n}'

    nodal_values = copy.deepcopy(SM['nodal_values'])
    nodal_values_combined = ['{' + ', '.join(f"{x:2.18e}" for x in nodal_values[key]) + '}'for key in MOOSE_output_keys]
    nodal_values_string = '{\n' + ',\n'.join(map(str, nodal_values_combined)) + '\n}'

    shape_func_degree = SM['mesh_specs']['shapefunc_degree']
    
    # Handle extrapolation element IDs
    if 'extrapolation_element_ids' in SM['mesh_specs']:
        extrapolation_element = SM['mesh_specs']['extrapolation_element_ids']
        extrapolation_element_string = '{' + ', '.join(map(str, extrapolation_element)) + '}'
    else:
        extrapolation_element_string = '{}'

    # Get triangular dimensions if available
    triangular_dimensions = []
    if 'tri_elements' in SM['mesh_specs']:
        tri_elements = SM['mesh_specs']['tri_elements']
        
        # Extract the actual dimension indices
        if isinstance(tri_elements, list):
            # For a list, use the indices of elements that are True or 1
            for i, value in enumerate(tri_elements):
                if value:  # This will work for both boolean True and integer 1
                    triangular_dimensions.append(i)
        elif isinstance(tri_elements, dict):
            # For a dictionary, the keys might be dimension names and values might be booleans
            # or the values themselves might be the dimension indices
            for i, (key, value) in enumerate(tri_elements.items()):
                if isinstance(value, bool) or isinstance(value, int):
                    if value:  # If True or non-zero
                        triangular_dimensions.append(i)
                else:
                    # If the value is something else, assume it's the dimension index
                    triangular_dimensions.append(value)

    triangular_dimensions_string = '{' + ', '.join(map(str, triangular_dimensions)) + '}'


    # Element numbers
    MOOSE_input_keys = list(SM['input_maps'].keys())
    element_numbers = np.zeros(len(MOOSE_input_keys), int)
    for i, key in enumerate(MOOSE_input_keys):
        if isinstance(SM['mesh_specs']['element_numbers'][key], list): 
            element_numbers[i] = len(SM['mesh_specs']['element_numbers'][key]) - 1
        elif isinstance(SM['mesh_specs']['element_numbers'][key], int):
            element_numbers[i] = SM['mesh_specs']['element_numbers'][key]

    element_numbers_string = '{' + ', '.join(map(str, element_numbers)) + '}' 

    # Input limits based on ROI
    input_limits = ['{' + ', '.join(f"{value:2.18e}" for value in np.array(SM['roi'][key])) + '}' for key in MOOSE_input_keys]
    input_limits_string = '{\n' + ',\n'.join(map(str, input_limits)) + '\n}'
    if 'rhoc' in MOOSE_input_keys:
        rhoc_min = str(min(SM['roi']['rhoc']))
        rhoc_max = str(max(SM['roi']['rhoc']))
    else:
        rhoc_min = "0.0"
        rhoc_max = "1.0"
    if 'rhow' in MOOSE_input_keys:
        rhow_min = str(min(SM['roi']['rhow']))
        rhow_max = str(max(SM['roi']['rhow']))
    else:
        rhow_min = "0.0"
        rhow_max = "1.0"

    # Input scalers
    input_scalers = []
    for key in MOOSE_input_keys:
        if 'Compress' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"COMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].compressor, SM['input_maps'][key].original_min))
        elif 'LogScaler' in str(SM['input_maps'][key]) and 'Sym' not in str(SM['input_maps'][key]):
            input_scalers.append("{{\"LOG10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].b, SM['input_maps'][key].a, SM['input_maps'][key].logmin, SM['input_maps'][key].logmax))
        elif 'MinMax' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"MINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].data_min_[0], SM['input_maps'][key].data_max_[0], SM['input_maps'][key].feature_range[0], SM['input_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"SYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].zmin, SM['input_maps'][key].zmax, SM['input_maps'][key].zbar, SM['input_maps'][key].lowerbound, SM['input_maps'][key].upperbound))


    input_scalers_string = '{' + ',\n'.join(input_scalers) + '\n}'

    # Output scalers
    output_scalers = []
    for key in MOOSE_output_keys:
        if 'Compress' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"DECOMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].compressor, SM['output_maps'][key].original_min))
        elif 'LogScaler' in str(SM['output_maps'][key]) and 'Sym' not in str(SM['output_maps'][key]) :
            output_scalers.append("{{\"EXP10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].b, SM['output_maps'][key].a, SM['output_maps'][key].logmin, SM['output_maps'][key].logmax))
        elif 'MinMax' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"INVMINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].data_min_[0], SM['output_maps'][key].data_max_[0], SM['output_maps'][key].feature_range[0], SM['output_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"EXPSYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].zmin, SM['output_maps'][key].zmax, SM['output_maps'][key].zbar, SM['output_maps'][key].lowerbound, SM['output_maps'][key].upperbound))

    output_scalers_string = '{' + ',\n'.join(output_scalers) + '\n}'

    # Element boundaries
    num_inputs = len(SM['roi'])
    num_elements = len(SM['mesh']['conn'])
    nodal_coordinates = copy.deepcopy(SM['mesh']['nodes'])
    # nodal_coordinates_physical = np.around(smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps']), decimals=30)
    minValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    maxValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    # Mind the ordering here: in the newer versions we use initializeKeyMaps in the material .C and .h files. In older versions, we will need to use some dict here, like MOOSE_index_convertor = [2, 4, 3, 0, 1, 5] and then use enumerate in the following loop:
    for i in range(nodal_coordinates.shape[1]):
        for element in range(num_elements):
            minValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].min()
            maxValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].max() 
    
    minValues_string = []
    maxValues_string = []
    for i in range(num_inputs):
        minValues_string.append("minValues[{0}] = Eigen::MatrixXd(1, {1});\nminValues[{0}] << ".format(i, num_elements))
        maxValues_string.append("maxValues[{0}] = Eigen::MatrixXd(1, {1});\nmaxValues[{0}] << ".format(i, num_elements))
        
        # Format min and max values with scientific notation and 18 decimal places
        min_values_formatted = ", ".join(f"{x:2.18e}" for x in minValues[i])
        max_values_formatted = ", ".join(f"{x:2.18e}" for x in maxValues[i])
        
        # Append the formatted values
        minValues_string[-1] += min_values_formatted + ";"
        maxValues_string[-1] += max_values_formatted + ";"

        # Join all parts into a single string
        minValues_code = "\n".join(minValues_string)
        maxValues_code = "\n".join(maxValues_string)


    # Generate the content of the .C-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "{0}.h"

registerMooseObject("BisonApp", {0});
registerADMooseObject("BisonApp", AD{0});

template <bool is_ad>
InputParameters
{0}Templ<is_ad>::validParams()
{{
  InputParameters params = LAROManceStressUpdateBaseUniversalTempl<is_ad>::validParams();
  params.addClassDescription("LAes creep update model for {0}");

  // Override defaults for material specific parameters below
  params.addRangeCheckedParam<Real>("initial_cell_dislocation_density",
                                    1.0e12,
                                    "initial_cell_dislocation_density >= {2} &"
                                    "initial_cell_dislocation_density <= {3}",
                                    "Initial density of mobile (glissile) dislocations (1/m^2).");
  params.addRangeCheckedParam<Real>("initial_wall_dislocation_density",
                                     7.0e12,
                                     "initial_wall_dislocation_density >= {4} &"
                                     "initial_wall_dislocation_density <= {5}",
                                     "Initial density of immobile (trapped) dislocations (1/m^2).");

  params.addRangeCheckedParam<Real>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of mobile (glissile) dislocations.");
params.addRangeCheckedParam<Real>(
    "max_relative_wall_dislocation_increment",
    0.5,
    "max_relative_wall_dislocation_increment > 0.0",
    "Maximum increment of density of immobile (trapped) dislocations.");

  return params;
}}

template <bool is_ad>
{0}Templ<is_ad>::{0}Templ(
    const InputParameters &parameters)
    : LAROManceStressUpdateBaseUniversalTempl<is_ad>(parameters)
{{
}}

template <bool is_ad>
void
{0}Templ<is_ad>::initializeKeyMaps()
{{
  // First call the base class implementation to set up default mappings
  LAROManceStressUpdateBaseUniversalTempl<is_ad>::initializeKeyMaps();
  
  // Override with specific ordering for this material
  // Input keys
  this->setInputIndexByKey("stress", 0);
  this->setInputIndexByKey("temperature", 1);
  this->setInputIndexByKey("old_strain", 2);
  this->setInputIndexByKey("cell", 3);
  this->setInputIndexByKey("wall", 4);

  // Output keys
  this->setOutputIndexByKey("strain", 0);
  this->setOutputIndexByKey("cell", 1);
  this->setOutputIndexByKey("wall", 2);
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getElementNumbers()
{{
  return {1};
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getTriangularDimensions()
{{
  return {6};
}}

template <bool is_ad>
std::string
{0}Templ<is_ad>::getShapeFunctionDegree()
{{
  return "{7}";
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getExtrapolationElement()
{{
  return {8};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getInputTransform()
{{
  return {9};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getOutputTransform()
{{
  return {10};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getInputLimits()
{{
  return {11};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodes()
{{
  // Store raw doubles first (safe), then convert to GenericReal<is_ad>
  static const std::vector<std::vector<double>> nodes_raw = {12};
  std::vector<std::vector<GenericReal<is_ad>>> result;
  result.reserve(nodes_raw.size());

  for (const auto& row : nodes_raw)
  {{
    std::vector<GenericReal<is_ad>> converted;
    converted.reserve(row.size());
    for (double v : row)
      converted.emplace_back(v);
    result.emplace_back(std::move(converted));
  }}
  return result;
}}

template <bool is_ad>
std::vector<std::vector<unsigned int>>
{0}Templ<is_ad>::getConnectivityMatrix()
{{
  return {13};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodalValues()
{{
  static const std::vector<std::vector<double>> nodal_values_raw = {14};
  
  std::vector<std::vector<GenericReal<is_ad>>> result;
  result.reserve(nodal_values_raw.size());

  for (const auto& row : nodal_values_raw)
  {{
    std::vector<GenericReal<is_ad>> converted;
    converted.reserve(row.size());
    for (double v : row)
      converted.emplace_back(v);
    result.emplace_back(std::move(converted));
  }}
  return result;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMinValues() const
{{
  // Container for minValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> minValues({15});

  {16}

  return minValues;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMaxValues() const
{{
  // Container for maxValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> maxValues({15});

  {17}
  
  return maxValues;
}}
""".format(name, element_numbers_string, rhoc_min, 
           rhoc_max, rhow_min, rhow_max,
           triangular_dimensions_string, shape_func_degree, extrapolation_element_string, 
           input_scalers_string, output_scalers_string, input_limits_string, 
           nodes_string, conn_string, nodal_values_string, 
           num_inputs, minValues_code, maxValues_code)

    return txt

def write_moose_C_6D_cached(SM: dict, name: str):
    """ 
    Write MOOSE C-file for surrogate models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for surrogate model inputs
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
    """

    MOOSE_output_keys = list(SM['nodal_values'].keys())

    # Generate some strings from the surrogate model dict "SM" 
    nodes_rows = ['{' + ', '.join(f"{x:2.18e}" for x in row) + '}' for row in SM['mesh']['nodes']]
    nodes_string = '{\n' + ',\n'.join(nodes_rows) + '\n}'

    conn_rows = ['{' + ', '.join(map(str, row)) + '}' for row in SM['mesh']['conn']]
    conn_string = '{\n' + ',\n'.join(conn_rows) + '\n}'

    nodal_values = copy.deepcopy(SM['nodal_values'])
    nodal_values_combined = ['{' + ', '.join(f"{x:2.18e}" for x in nodal_values[key]) + '}'for key in MOOSE_output_keys]
    nodal_values_string = '{\n' + ',\n'.join(map(str, nodal_values_combined)) + '\n}'

    shape_func_degree = SM['mesh_specs']['shapefunc_degree']
    
    # Handle extrapolation element IDs
    if 'extrapolation_element_ids' in SM['mesh_specs']:
        extrapolation_element = SM['mesh_specs']['extrapolation_element_ids']
        extrapolation_element_string = '{' + ', '.join(map(str, extrapolation_element)) + '}'
    else:
        extrapolation_element_string = '{}'

    # Get triangular dimensions if available
    triangular_dimensions = []
    if 'tri_elements' in SM['mesh_specs']:
        tri_elements = SM['mesh_specs']['tri_elements']
        
        # Extract the actual dimension indices
        if isinstance(tri_elements, list):
            # For a list, use the indices of elements that are True or 1
            for i, value in enumerate(tri_elements):
                if value:  # This will work for both boolean True and integer 1
                    triangular_dimensions.append(i)
        elif isinstance(tri_elements, dict):
            # For a dictionary, the keys might be dimension names and values might be booleans
            # or the values themselves might be the dimension indices
            for i, (key, value) in enumerate(tri_elements.items()):
                if isinstance(value, bool) or isinstance(value, int):
                    if value:  # If True or non-zero
                        triangular_dimensions.append(i)
                else:
                    # If the value is something else, assume it's the dimension index
                    triangular_dimensions.append(value)

    triangular_dimensions_string = '{' + ', '.join(map(str, triangular_dimensions)) + '}'


    # Element numbers
    MOOSE_input_keys = list(SM['input_maps'].keys())
    element_numbers = np.zeros(len(MOOSE_input_keys), int)
    for i, key in enumerate(MOOSE_input_keys):
        if isinstance(SM['mesh_specs']['element_numbers'][key], list): 
            element_numbers[i] = len(SM['mesh_specs']['element_numbers'][key]) - 1
        elif isinstance(SM['mesh_specs']['element_numbers'][key], int):
            element_numbers[i] = SM['mesh_specs']['element_numbers'][key]

    element_numbers_string = '{' + ', '.join(map(str, element_numbers)) + '}' 

    # Input limits based on ROI
    input_limits = ['{' + ', '.join(f"{value:2.18e}" for value in np.array(SM['roi'][key])) + '}' for key in MOOSE_input_keys]
    input_limits_string = '{\n' + ',\n'.join(map(str, input_limits)) + '\n}'
    if 'rhoc' in MOOSE_input_keys:
        rhoc_min = str(min(SM['roi']['rhoc']))
        rhoc_max = str(max(SM['roi']['rhoc']))
    else:
        rhoc_min = "0.0"
        rhoc_max = "1.0"
    if 'rhow' in MOOSE_input_keys:
        rhow_min = str(min(SM['roi']['rhow']))
        rhow_max = str(max(SM['roi']['rhow']))
    else:
        rhow_min = "0.0"
        rhow_max = "1.0"

    # Input scalers
    input_scalers = []
    for key in MOOSE_input_keys:
        if 'Compress' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"COMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].compressor, SM['input_maps'][key].original_min))
        elif 'LogScaler' in str(SM['input_maps'][key]) and 'Sym' not in str(SM['input_maps'][key]):
            input_scalers.append("{{\"LOG10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].b, SM['input_maps'][key].a, SM['input_maps'][key].logmin, SM['input_maps'][key].logmax))
        elif 'MinMax' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"MINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].data_min_[0], SM['input_maps'][key].data_max_[0], SM['input_maps'][key].feature_range[0], SM['input_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"SYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].zmin, SM['input_maps'][key].zmax, SM['input_maps'][key].zbar, SM['input_maps'][key].lowerbound, SM['input_maps'][key].upperbound))


    input_scalers_string = '{' + ',\n'.join(input_scalers) + '\n}'

    # Output scalers
    output_scalers = []
    for key in MOOSE_output_keys:
        if 'Compress' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"DECOMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].compressor, SM['output_maps'][key].original_min))
        elif 'LogScaler' in str(SM['output_maps'][key]) and 'Sym' not in str(SM['output_maps'][key]) :
            output_scalers.append("{{\"EXP10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].b, SM['output_maps'][key].a, SM['output_maps'][key].logmin, SM['output_maps'][key].logmax))
        elif 'MinMax' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"INVMINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].data_min_[0], SM['output_maps'][key].data_max_[0], SM['output_maps'][key].feature_range[0], SM['output_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"EXPSYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].zmin, SM['output_maps'][key].zmax, SM['output_maps'][key].zbar, SM['output_maps'][key].lowerbound, SM['output_maps'][key].upperbound))

    output_scalers_string = '{' + ',\n'.join(output_scalers) + '\n}'

    # Element boundaries
    num_inputs = len(SM['roi'])
    num_elements = len(SM['mesh']['conn'])
    nodal_coordinates = copy.deepcopy(SM['mesh']['nodes'])
    # nodal_coordinates_physical = np.around(smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps']), decimals=30)
    minValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    maxValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    # Mind the ordering here: in the newer versions we use initializeKeyMaps in the material .C and .h files. In older versions, we will need to use some dict here, like MOOSE_index_convertor = [2, 4, 3, 0, 1, 5] and then use enumerate in the following loop:
    for i in range(nodal_coordinates.shape[1]):
        for element in range(num_elements):
            minValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].min()
            maxValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].max() 
    
    minValues_string = []
    maxValues_string = []
    for i in range(num_inputs):
        minValues_string.append("minValues[{0}] = Eigen::MatrixXd(1, {1});\nminValues[{0}] << ".format(i, num_elements))
        maxValues_string.append("maxValues[{0}] = Eigen::MatrixXd(1, {1});\nmaxValues[{0}] << ".format(i, num_elements))
        
        # Format min and max values with scientific notation and 18 decimal places
        min_values_formatted = ", ".join(f"{x:2.18e}" for x in minValues[i])
        max_values_formatted = ", ".join(f"{x:2.18e}" for x in maxValues[i])
        
        # Append the formatted values
        minValues_string[-1] += min_values_formatted + ";"
        maxValues_string[-1] += max_values_formatted + ";"

        # Join all parts into a single string
        minValues_code = "\n".join(minValues_string)
        maxValues_code = "\n".join(maxValues_string)


    # Generate the content of the .C-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "{0}.h"

registerMooseObject("BisonApp", {0});
registerADMooseObject("BisonApp", AD{0});

template <bool is_ad>
InputParameters
{0}Templ<is_ad>::validParams()
{{
  InputParameters params = LAROManceStressUpdateBaseUniversalTempl<is_ad>::validParams();
  params.addClassDescription("LAes creep update model for {0}");

  // Override defaults for material specific parameters below
  params.addRangeCheckedParam<Real>("initial_cell_dislocation_density",
                                    1.0e12,
                                    "initial_cell_dislocation_density >= {2} &"
                                    "initial_cell_dislocation_density <= {3}",
                                    "Initial density of mobile (glissile) dislocations (1/m^2).");
  params.addRangeCheckedParam<Real>("initial_wall_dislocation_density",
                                     7.0e12,
                                     "initial_wall_dislocation_density >= {4} &"
                                     "initial_wall_dislocation_density <= {5}",
                                     "Initial density of immobile (trapped) dislocations (1/m^2).");

  params.addRangeCheckedParam<Real>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of mobile (glissile) dislocations.");
params.addRangeCheckedParam<Real>(
    "max_relative_wall_dislocation_increment",
    0.5,
    "max_relative_wall_dislocation_increment > 0.0",
    "Maximum increment of density of immobile (trapped) dislocations.");

  return params;
}}

template <bool is_ad>
{0}Templ<is_ad>::{0}Templ(
    const InputParameters &parameters)
    : LAROManceStressUpdateBaseUniversalTempl<is_ad>(parameters)
{{
}}

template <bool is_ad>
void
{0}Templ<is_ad>::initializeKeyMaps()
{{
  // First call the base class implementation to set up default mappings
  LAROManceStressUpdateBaseUniversalTempl<is_ad>::initializeKeyMaps();
  
  // Override with specific ordering for this material
  // Input keys
  this->setInputIndexByKey("stress", 0);
  this->setInputIndexByKey("temperature", 1);
  this->setInputIndexByKey("old_strain", 2);
  this->setInputIndexByKey("cell", 3);
  this->setInputIndexByKey("wall", 4);
  this->setInputIndexByKey("environmental", 5);

  // Output keys
  this->setOutputIndexByKey("strain", 0);
  this->setOutputIndexByKey("cell", 1);
  this->setOutputIndexByKey("wall", 2);
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getElementNumbers()
{{
  return {1};
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getTriangularDimensions()
{{
  return {6};
}}

template <bool is_ad>
std::string
{0}Templ<is_ad>::getShapeFunctionDegree()
{{
  return "{7}";
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getExtrapolationElement()
{{
  return {8};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getInputTransform()
{{
  return {9};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getOutputTransform()
{{
  return {10};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getInputLimits()
{{
  return {11};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodes()
{{
  // Store raw doubles first (safe), then convert to GenericReal<is_ad>
  static const std::vector<std::vector<double>> nodes_raw = {12};
  std::vector<std::vector<GenericReal<is_ad>>> result;
  result.reserve(nodes_raw.size());

  for (const auto& row : nodes_raw)
  {{
    std::vector<GenericReal<is_ad>> converted;
    converted.reserve(row.size());
    for (double v : row)
      converted.emplace_back(v);
    result.emplace_back(std::move(converted));
  }}
  return result;
}}

template <bool is_ad>
std::vector<std::vector<unsigned int>>
{0}Templ<is_ad>::getConnectivityMatrix()
{{
  return {13};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodalValues()
{{
  static const std::vector<std::vector<double>> nodal_values_raw = {14};
  
  std::vector<std::vector<GenericReal<is_ad>>> result;
  result.reserve(nodal_values_raw.size());

  for (const auto& row : nodal_values_raw)
  {{
    std::vector<GenericReal<is_ad>> converted;
    converted.reserve(row.size());
    for (double v : row)
      converted.emplace_back(v);
    result.emplace_back(std::move(converted));
  }}
  return result;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMinValues() const
{{
  // Container for minValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> minValues({15});

  {16}

  return minValues;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMaxValues() const
{{
  // Container for maxValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> maxValues({15});

  {17}
  
  return maxValues;
}}
""".format(name, element_numbers_string, rhoc_min, 
           rhoc_max, rhow_min, rhow_max,
           triangular_dimensions_string, shape_func_degree, extrapolation_element_string, 
           input_scalers_string, output_scalers_string, input_limits_string, 
           nodes_string, conn_string, nodal_values_string, 
           num_inputs, minValues_code, maxValues_code)

    return txt


def write_moose_C_cached_8D(SM: dict, name: str):
    """ 
    Write MOOSE C-file for surrogate models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for surrogate model inputs
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
    """

    MOOSE_output_keys = ['evm', 'rhoc', 'rhow']

    # Generate some strings from the surrogate model dict "SM" 
    nodes_rows = ['{' + ', '.join(f"{x:2.18e}" for x in row) + '}' for row in SM['mesh']['nodes']]
    nodes_string = '{\n' + ',\n'.join(nodes_rows) + '\n}'

    conn_rows = ['{' + ', '.join(map(str, row)) + '}' for row in SM['mesh']['conn']]
    conn_string = '{\n' + ',\n'.join(conn_rows) + '\n}'

    nodal_values = copy.deepcopy(SM['nodal_values'])
    nodal_values_combined = ['{' + ', '.join(f"{x:2.18e}" for x in nodal_values[key]) + '}'for key in MOOSE_output_keys]
    nodal_values_string = '{\n' + ',\n'.join(map(str, nodal_values_combined)) + '\n}'

    shape_func_degree = SM['mesh_specs']['shapefunc_degree']
    
    # Handle extrapolation element IDs
    if 'extrapolation_element_ids' in SM['mesh_specs']:
        extrapolation_element = SM['mesh_specs']['extrapolation_element_ids']
        extrapolation_element_string = '{' + ', '.join(map(str, extrapolation_element)) + '}'
    else:
        extrapolation_element_string = '{}'

    # Get triangular dimensions if available
    triangular_dimensions = []
    if 'tri_elements' in SM['mesh_specs']:
        tri_elements = SM['mesh_specs']['tri_elements']
        
        # Extract the actual dimension indices
        if isinstance(tri_elements, list):
            # For a list, use the indices of elements that are True or 1
            for i, value in enumerate(tri_elements):
                if value:  # This will work for both boolean True and integer 1
                    triangular_dimensions.append(i)
        elif isinstance(tri_elements, dict):
            # For a dictionary, the keys might be dimension names and values might be booleans
            # or the values themselves might be the dimension indices
            for i, (key, value) in enumerate(tri_elements.items()):
                if isinstance(value, bool) or isinstance(value, int):
                    if value:  # If True or non-zero
                        triangular_dimensions.append(i)
                else:
                    # If the value is something else, assume it's the dimension index
                    triangular_dimensions.append(value)

    triangular_dimensions_string = '{' + ', '.join(map(str, triangular_dimensions)) + '}'


    # Element numbers
    MOOSE_input_keys = ["vmJ2", "temperature", "evm", "rhoc", "rhow", "dloops_density_values", "dloops_size_values", "bd_density_values"]
    element_numbers = np.zeros(len(MOOSE_input_keys), int)
    for i, key in enumerate(MOOSE_input_keys):
        if isinstance(SM['mesh_specs']['element_numbers'][key], list): 
            element_numbers[i] = len(SM['mesh_specs']['element_numbers'][key]) - 1
        elif isinstance(SM['mesh_specs']['element_numbers'][key], int):
            element_numbers[i] = SM['mesh_specs']['element_numbers'][key]

    element_numbers_string = '{' + ', '.join(map(str, element_numbers)) + '}' 

    # Input limits based on ROI
    input_limits = ['{' + ', '.join(f"{value:2.18e}" for value in np.array(SM['roi'][key])) + '}' for key in MOOSE_input_keys]
    input_limits_string = '{\n' + ',\n'.join(map(str, input_limits)) + '\n}'
    rhoc_min = str(min(SM['roi']['rhoc']))
    rhoc_max = str(max(SM['roi']['rhoc']))
    rhow_min = str(min(SM['roi']['rhow']))
    rhow_max = str(max(SM['roi']['rhow']))

    # Input scalers
    input_scalers = []
    for key in MOOSE_input_keys:
        if 'Compress' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"COMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].compressor, SM['input_maps'][key].original_min))
        elif 'LogScaler' in str(SM['input_maps'][key]) and 'sym' not in str(SM['input_maps']):
            input_scalers.append("{{\"LOG10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].b, SM['input_maps'][key].a, SM['input_maps'][key].logmin, SM['input_maps'][key].logmax))
        elif 'MinMax' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"MINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].data_min_[0], SM['input_maps'][key].data_max_[0], SM['input_maps'][key].feature_range[0], SM['input_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"SYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].zmin, SM['input_maps'][key].zmax, SM['input_maps'][key].zbar, SM['input_maps'][key].lowerbound, SM['input_maps'][key].upperbound))


    input_scalers_string = '{' + ',\n'.join(input_scalers) + '\n}'

    # Output scalers
    MOOSE_output_keys = ['evm', 'rhoc', 'rhow']
    output_scalers = []
    for key in MOOSE_output_keys:
        if 'Compress' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"DECOMPRESS\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].compressor, SM['output_maps'][key].original_min))
        elif 'LogScaler' in str(SM['output_maps'][key]) and 'Sym' not in str(SM['output_maps'][key]) :
            output_scalers.append("{{\"EXP10BOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].b, SM['output_maps'][key].a, SM['output_maps'][key].logmin, SM['output_maps'][key].logmax))
        elif 'MinMax' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"INVMINMAX\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].data_min_[0], SM['output_maps'][key].data_max_[0], SM['output_maps'][key].feature_range[0], SM['output_maps'][key].feature_range[1]))
        elif 'SymLog' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"EXPSYMLOGBOUNDED\", typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::SymLogTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].zmin, SM['output_maps'][key].zmax, SM['output_maps'][key].zbar, SM['output_maps'][key].lowerbound, SM['output_maps'][key].upperbound))

    output_scalers_string = '{' + ',\n'.join(output_scalers) + '\n}'

    # Element boundaries
    num_inputs = len(SM['roi'])
    num_elements = len(SM['mesh']['conn'])
    nodal_coordinates = copy.deepcopy(SM['mesh']['nodes'])
    # nodal_coordinates_physical = np.around(smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps']), decimals=30)
    minValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    maxValues = np.zeros([num_inputs, len(SM['mesh']['conn'])])
    # Mind the ordering here: in the newer versions we use initializeKeyMaps in the material .C and .h files. In older versions, we will need to use some dict here, like MOOSE_index_convertor = [2, 4, 3, 0, 1, 5] and then use enumerate in the following loop:
    for i in range(nodal_coordinates.shape[1]):
        for element in range(num_elements):
            minValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].min()
            maxValues[i, element] = nodal_coordinates[SM['mesh']['conn'][element], i].max() 
    
    minValues_string = []
    maxValues_string = []
    for i in range(num_inputs):
        minValues_string.append("minValues[{0}] = Eigen::MatrixXd(1, {1});\nminValues[{0}] << ".format(i, num_elements))
        maxValues_string.append("maxValues[{0}] = Eigen::MatrixXd(1, {1});\nmaxValues[{0}] << ".format(i, num_elements))
        
        # Format min and max values with scientific notation and 18 decimal places
        min_values_formatted = ", ".join(f"{x:2.18e}" for x in minValues[i])
        max_values_formatted = ", ".join(f"{x:2.18e}" for x in maxValues[i])
        
        # Append the formatted values
        minValues_string[-1] += min_values_formatted + ";"
        maxValues_string[-1] += max_values_formatted + ";"

        # Join all parts into a single string
        minValues_code = "\n".join(minValues_string)
        maxValues_code = "\n".join(maxValues_string)


    # Generate the content of the .C-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "{0}.h"

registerMooseObject("PolecatApp", {0});
registerADMooseObject("PolecatApp", AD{0});

template <bool is_ad>
InputParameters
{0}Templ<is_ad>::validParams()
{{
  InputParameters params = LAROManceStressUpdateBaseUniversalTempl<is_ad>::validParams();
  params.addClassDescription("LAes creep update model for {0}");

  // Override defaults for material specific parameters below
  params.addRangeCheckedParam<GenericReal<is_ad>>("initial_cell_dislocation_density",
                                    1.0e12,
                                    "initial_cell_dislocation_density >= {2} &"
                                    "initial_cell_dislocation_density <= {3}",
                                    "Initial density of mobile (glissile) dislocations (1/m^2).");
  params.addRangeCheckedParam<GenericReal<is_ad>>("initial_wall_dislocation_density",
                                     7.0e12,
                                     "initial_wall_dislocation_density >= {4} &"
                                     "initial_wall_dislocation_density <= {5}",
                                     "Initial density of immobile (trapped) dislocations (1/m^2).");

  params.addRangeCheckedParam<GenericReal<is_ad>>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of mobile (glissile) dislocations.");
params.addRangeCheckedParam<GenericReal<is_ad>>(
    "max_relative_wall_dislocation_increment",
    0.5,
    "max_relative_wall_dislocation_increment > 0.0",
    "Maximum increment of density of immobile (trapped) dislocations.");

  return params;
}}

template <bool is_ad>
{0}Templ<is_ad>::{0}Templ(
    const InputParameters &parameters)
    : LAROManceStressUpdateBaseUniversalTempl<is_ad>(parameters)
{{
}}

template <bool is_ad>
void
{0}Templ<is_ad>::initializeKeyMaps()
{{
  // First call the base class implementation to set up default mappings
  LAROManceStressUpdateBaseUniversalTempl<is_ad>::initializeKeyMaps();
  
  // Override with specific ordering for this material
  // Input keys
  this->setInputIndexByKey("stress", 0);
  this->setInputIndexByKey("temperature", 1);
  this->setInputIndexByKey("old_strain", 2);
  this->setInputIndexByKey("cell", 3);
  this->setInputIndexByKey("wall", 4);
  this->setInputIndexByKey("microstruct_param_1", 5);
  this->setInputIndexByKey("microstruct_param_2", 6);
  this->setInputIndexByKey("microstruct_param_3", 7);

  // Output keys
  this->setOutputIndexByKey("strain", 0);
  this->setOutputIndexByKey("cell", 1);
  this->setOutputIndexByKey("wall", 2);
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getElementNumbers()
{{
  return {1};
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getTriangularDimensions()
{{
  return {6};
}}

template <bool is_ad>
std::string
{0}Templ<is_ad>::getShapeFunctionDegree()
{{
  return "{7}";
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getExtrapolationElement()
{{
  return {8};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getInputTransform()
{{
  return {9};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAROManceStressUpdateBaseUniversalTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getOutputTransform()
{{
  return {10};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getInputLimits()
{{
  return {11};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodes()
{{
  return {12};
}}

template <bool is_ad>
std::vector<std::vector<unsigned int>>
{0}Templ<is_ad>::getConnectivityMatrix()
{{
  return {13};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodalValues()
{{
  return {14};
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMinValues() const
{{
  // Container for minValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> minValues({15});

  {16}

  return minValues;
}}

template <bool is_ad>
std::vector<Eigen::MatrixXd>
{0}Templ<is_ad>::getMaxValues() const
{{
  // Container for maxValues for each domain (using Eigen)
  std::vector<Eigen::MatrixXd> maxValues({15});

  {17}
  
  return maxValues;
}}
""".format(name, element_numbers_string, rhoc_min, 
           rhoc_max, rhow_min, rhow_max,
           triangular_dimensions_string, shape_func_degree, extrapolation_element_string, 
           input_scalers_string, output_scalers_string, input_limits_string, 
           nodes_string, conn_string, nodal_values_string, 
           num_inputs, minValues_code, maxValues_code)

    return txt


def write_moose_6d_C(SM: dict, name: str):
    """ 
    Write MOOSE C-file for surrogate models.

    input:
    - SM : dict
        dictionary containing surrogate model variables, including:
            - nodes : ndarray
                node coordinates for surrogate model inputs
            - conn : ndarray
                connectivity matrix: one element per row
    - name : desired name for the model (used in the MOOSE code)
    """

    # Generate some strings from the surrogate model dict "SM" 
    nodes_rows = ['{' + ', '.join(f"{x:2.18e}" for x in row) + '}' for row in SM['mesh']['nodes']]
    nodes_string = '{\n' + ',\n'.join(nodes_rows) + '\n}'

    conn_rows = ['{' + ', '.join(map(str, row)) + '}' for row in SM['mesh']['conn']]
    conn_string = '{\n' + ',\n'.join(conn_rows) + '\n}'

    nodal_values = copy.deepcopy(SM['nodal_values'])
    nodal_values_combined = ['{' + ', '.join(f"{x:2.18e}" for x in nodal_values[key]) + '}'for key in ['evm', 'rhoc', 'rhow']]
    nodal_values_string = '{\n' + ',\n'.join(map(str, nodal_values_combined)) + '\n}'

    shape_func_degree = SM['mesh_specs']['shapefunc_degree']
    
    # Handle extrapolation element IDs
    if 'extrapolation_element_ids' in SM['mesh_specs']:
        extrapolation_element = SM['mesh_specs']['extrapolation_element_ids']
        extrapolation_element_string = '{' + ', '.join(map(str, extrapolation_element)) + '}'
    else:
        extrapolation_element_string = '{}'

    # Get triangular dimensions if available
    triangular_dimensions = []
    if 'tri_elements' in SM['mesh_specs']:
        tri_elements = SM['mesh_specs']['tri_elements']
        
        # Extract the actual dimension indices
        if isinstance(tri_elements, list):
            # For a list, use the indices of elements that are True or 1
            for i, value in enumerate(tri_elements):
                if value:  # This will work for both boolean True and integer 1
                    triangular_dimensions.append(i)
        elif isinstance(tri_elements, dict):
            # For a dictionary, the keys might be dimension names and values might be booleans
            # or the values themselves might be the dimension indices
            for i, (key, value) in enumerate(tri_elements.items()):
                if isinstance(value, bool) or isinstance(value, int):
                    if value:  # If True or non-zero
                        triangular_dimensions.append(i)
                else:
                    # If the value is something else, assume it's the dimension index
                    triangular_dimensions.append(value)

    triangular_dimensions_string = '{' + ', '.join(map(str, triangular_dimensions)) + '}'


    # Element numbers
    MOOSE_input_keys = ["vmJ2", "temperature", "evm", "rhoc", "rhow", "flux"]
    element_numbers = np.zeros(len(MOOSE_input_keys), int)
    for i, key in enumerate(MOOSE_input_keys):
        if isinstance(SM['mesh_specs']['element_numbers'][key], list): 
            element_numbers[i] = len(SM['mesh_specs']['element_numbers'][key]) - 1
        elif isinstance(SM['mesh_specs']['element_numbers'][key], int):
            element_numbers[i] = SM['mesh_specs']['element_numbers'][key]

    element_numbers_string = '{' + ', '.join(map(str, element_numbers)) + '}' 

    # Input limits based on ROI
    input_limits = ['{' + ', '.join(f"{value:2.18e}" for value in np.array(SM['roi'][key])) + '}' for key in MOOSE_input_keys]
    input_limits_string = '{\n' + ',\n'.join(map(str, input_limits)) + '\n}'
    rhoc_min = str(min(SM['roi']['rhoc']))
    rhoc_max = str(max(SM['roi']['rhoc']))
    rhow_min = str(min(SM['roi']['rhow']))
    rhow_max = str(max(SM['roi']['rhow']))

    # Input scalers
    input_scalers = []
    for key in MOOSE_input_keys:
        if 'Compress' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"COMPRESS\", typename LAesStressUpdateBaseNewTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].compressor, SM['input_maps'][key].original_min))
        elif 'LogScaler' in str(SM['input_maps'][key]) and 'sym' not in str(SM['input_maps']):
            input_scalers.append("{{\"LOG10BOUNDED\", typename LAesStressUpdateBaseNewTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].factor, SM['input_maps'][key].b, SM['input_maps'][key].a, SM['input_maps'][key].logmin, SM['input_maps'][key].logmax))
        elif 'MinMax' in str(SM['input_maps'][key]):
            input_scalers.append("{{\"MINMAX\", typename LAesStressUpdateBaseNewTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['input_maps'][key].data_min_[0], SM['input_maps'][key].data_max_[0], SM['input_maps'][key].feature_range[0], SM['input_maps'][key].feature_range[1]))

    input_scalers_string = '{' + ',\n'.join(input_scalers) + '\n}'

    # Output scalers
    MOOSE_output_keys = ['evm', 'rhoc', 'rhow']
    output_scalers = []
    for key in MOOSE_output_keys:
        if 'Compress' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"DECOMPRESS\", typename LAesStressUpdateBaseNewTempl<is_ad>::CompressTransform{{{:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].compressor, SM['output_maps'][key].original_min))
        elif 'LogScaler' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"EXP10BOUNDED\", typename LAesStressUpdateBaseNewTempl<is_ad>::Log10Transform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].factor, SM['output_maps'][key].b, SM['output_maps'][key].a, SM['output_maps'][key].logmin, SM['output_maps'][key].logmax))
        elif 'MinMax' in str(SM['output_maps'][key]):
            output_scalers.append("{{\"INVMINMAX\", typename LAesStressUpdateBaseNewTempl<is_ad>::MinMaxTransform{{{:2.18e}, {:2.18e}, {:2.18e}, {:2.18e}}}}}".format(SM['output_maps'][key].data_min_[0], SM['output_maps'][key].data_max_[0], SM['output_maps'][key].feature_range[0], SM['output_maps'][key].feature_range[1]))

    output_scalers_string = '{' + ',\n'.join(output_scalers) + '\n}'

    # Generate the content of the .C-file.
    txt = """
/************************************************************************************/
/*                        © 2026 Triad National Security, LLC                       */
/*                                ALL RIGHTS RESERVED                               */
/*                                                                                  */
/* This software was produced under U.S. Government contract 89233218CNA000001 for  */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. The U.S. Government has rights to use, reproduce, and distribute */
/* this software. NEITHER THE GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY */
/* WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS       */
/* SOFTWARE. If software is modified to produce derivative works, such modified     */
/* software should be clearly marked, so as not to confuse it with the version      */
/* available from LANL.                                                             */
/*                                                                                  */
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "{0}.h"

registerMooseObject("PolecatApp", {0});
registerADMooseObject("PolecatApp", AD{0});

template <bool is_ad>
InputParameters
{0}Templ<is_ad>::validParams()
{{
  InputParameters params = LAesStressUpdateBaseNewTempl<is_ad>::validParams();
  params.addClassDescription("LAes creep update model for {0}");

  // Override defaults for material specific parameters below
  params.addRangeCheckedParam<GenericReal<is_ad>>("initial_cell_dislocation_density",
                                    7.0e12,
                                    "initial_cell_dislocation_density >= {2} &"
                                    "initial_cell_dislocation_density <= {3}",
                                    "Initial density of mobile (glissile) dislocations (1/m^2).");
  params.addRangeCheckedParam<GenericReal<is_ad>>("initial_wall_dislocation_density",
                                     7.0e12,
                                     "initial_wall_dislocation_density >= {4} &"
                                     "initial_wall_dislocation_density <= {5}",
                                     "Initial density of immobile (trapped) dislocations (1/m^2).");

  params.addRangeCheckedParam<GenericReal<is_ad>>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of mobile (glissile) dislocations.");
  params.addRangeCheckedParam<GenericReal<is_ad>>(
      "max_relative_wall_dislocation_increment",
      0.5,
      "max_relative_wall_dislocation_increment > 0.0",
      "Maximum increment of density of immobile (trapped) dislocations.");

  return params;
}}

template <bool is_ad>
{0}Templ<is_ad>::{0}Templ(
    const InputParameters &parameters)
    : LAesStressUpdateBaseNewTempl<is_ad>(parameters)
{{
}}

template <bool is_ad>
void
{0}Templ<is_ad>::initializeKeyMaps()
{{
  // First call the base class implementation to set up default mappings
  LAesStressUpdateBaseNewTempl<is_ad>::initializeKeyMaps();
  
  // Override with specific ordering for this material
  // Input keys
  this->setInputIndexByKey("stress", 0);
  this->setInputIndexByKey("temperature", 1);
  this->setInputIndexByKey("old_strain", 2);
  this->setInputIndexByKey("cell", 3);
  this->setInputIndexByKey("wall", 4);
  this->setInputIndexByKey("environmental", 5);

  // Output keys
  this->setOutputIndexByKey("strain", 0);
  this->setOutputIndexByKey("cell", 1);
  this->setOutputIndexByKey("wall", 2);
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getElementNumbers()
{{
  return {1};
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getTriangularDimensions()
{{
  return {6};
}}

template <bool is_ad>
std::string
{0}Templ<is_ad>::getShapeFunctionDegree()
{{
  return "{7}";
}}

template <bool is_ad>
std::vector<unsigned int>
{0}Templ<is_ad>::getExtrapolationElement()
{{
  return {8};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getInputTransform()
{{
  return {9};
}}

template <bool is_ad>
std::vector<std::pair<std::string, typename LAesStressUpdateBaseNewTempl<is_ad>::Transform>>
{0}Templ<is_ad>::getOutputTransform()
{{
  return {10};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getInputLimits()
{{
  return {11};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodes()
{{
  return {12};
}}

template <bool is_ad>
std::vector<std::vector<unsigned int>>
{0}Templ<is_ad>::getConnectivityMatrix()
{{
  return {13};
}}

template <bool is_ad>
std::vector<std::vector<GenericReal<is_ad>>>
{0}Templ<is_ad>::getNodalValues()
{{
  return {14};
}}
""".format(name, element_numbers_string, rhoc_min, rhoc_max, 
           rhow_min, rhow_max,
           triangular_dimensions_string, shape_func_degree, 
           extrapolation_element_string, input_scalers_string, 
           output_scalers_string, input_limits_string, 
           nodes_string, conn_string, nodal_values_string)

    return txt