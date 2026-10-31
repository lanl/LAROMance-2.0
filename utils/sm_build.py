#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Finite element surrogate model build methods.

sm_train:      calibration of finite element nodal values
sm_test:   testing of calibrated model
rate_stepper: subprocess for creep simulation

@author: Andre Ruybalid
"""

import os, sys, copy, numpy as np, time, pickle, scipy.sparse.linalg
from scipy.optimize import lsq_linear
from scipy.optimize import newton
from scipy.sparse.linalg import spsolve
import scipy.optimize as opt
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from tqdm import tqdm
from multiprocessing import cpu_count

from utils.mlp_utils import PointDataset, train_mlp, collate_fn
from utils.mlp_models import get_model
import torch
from torch.utils.data import DataLoader

from utils import fe_shapes as fes
from utils import sm_analyze as sma
from utils import sm_data as smd
from utils import sm_plot as smp

# ============================================================
# Shared helper: pretty R² report (used by both grid and neural paths)
# ============================================================
def _print_r2_report(r2_dict, title="Training fit quality"):
    """Pretty-print R² values with color coding (green ≥0.99, yellow ≥0.95, red <0.95)."""
    GREEN, YEL, RED, BOLD, END = "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"
    print(f"\n{BOLD}===== {title} ====={END}")
    for key, val in r2_dict.items():
        if key == "M_condition":
            continue
        col = GREEN if val >= 0.99 else (YEL if val >= 0.95 else RED)
        print(f"\t{BOLD}R-squared {key}:\t{col}{val:4.3f}{END}")
    if "M_condition" in r2_dict:
        print(f"\t{BOLD}Condition number M:\t{r2_dict['M_condition']:4.2f}{END}")

# ============================================================
# MLP Inference Helper
# ============================================================

def prepare_mlp_args(SM, args=None):
    """
    Automatically populates the args dictionary with MLP-related keys
    needed for rate_stepper() when using an MLP model.

    Parameters
    ----------
    SM : dict
        Surrogate model dictionary. Must contain:
            - "mlp_model"
            - "mlp_input_vars"
            - "mlp_output_vars"
    args : dict, optional
        Existing args dictionary. If None, a new one is created.

    Returns
    -------
    args : dict
        Updated args dictionary ready for MLP inference.
    """
    if args is None:
        args = {}

    if "mlp_model" not in SM:
        raise KeyError("SM does not contain 'mlp_model'. "
                       "Make sure you stored the model after MLP training.")

    args["use_mlp"] = True
    args["mlp_model"] = SM["mlp_model"]
    args["mlp_input_vars"] = SM.get("mlp_input_vars", [])
    args["mlp_output_vars"] = SM.get("mlp_output_vars", [])

    return args

# ============================================================
# MLP Trainer
# ============================================================

def mlp_train(roi, mesh_specs, args, DATA):
    print("\n" + "="*60)
    print("MLP TRAINING FORK (using real scaler objects)")
    print("="*60)

    input_vars = list(mesh_specs.get("element_numbers", {}).keys())
    output_vars = list(args.get("map_output", {}).keys())


    if not input_vars or not output_vars:
        raise ValueError("No input or output variables found.")

    print(f"Input variables : {input_vars}")
    print(f"Output variables: {output_vars}")

    map_input = args.get("map_input", {})
    map_output = args.get("map_output", {})
    if not map_input:
        print("[WARNING] No input scalers found in args — using raw data")
    if not map_output:
        print("[WARNING] No output scalers found in args — using raw data")

    # Build raw data
    X_list = [np.asarray(DATA["data"][var], dtype=np.float64) for var in input_vars]
    X = np.column_stack(X_list)

    y_list = [np.asarray(DATA["data"]["U"][var], dtype=np.float64) for var in output_vars]
    y = np.column_stack(y_list)

    print(f"Total training points: {X.shape[0]}")

    # === Apply input scalers using the actual scaler objects ===
    X_scaled = X.copy()
    for i, var in enumerate(input_vars):
        if var in map_input:
            entry = map_input[var]
            # support both formats:
            # 1) direct fitted scaler (new)
            # 2) {"minmax": scaler} (old)
            if hasattr(entry, "transform"):
                scaler = entry
            else:
                scaler_dict = entry
                scaler_type = list(scaler_dict.keys())[0]
                scaler = scaler_dict[scaler_type]
            if hasattr(scaler, "transform"):
                X_scaled[:, i] = scaler.transform(X[:, i].reshape(-1, 1)).ravel()
            else:
                print("WARNING: fallback to apply_input_scaler for input variable:", var)
                X_scaled[:, i] = apply_input_scaler(X[:, i], scaler_type if 'scaler_type' in locals() else None, scaler)

    # === Apply output scalers using the actual scaler objects ===
    y_scaled = y.copy()
    for i, var in enumerate(output_vars):
        if var in map_output:
            entry = map_output[var]
            # support both formats:
            # 1) direct fitted scaler (new)
            # 2) {"minmax": scaler} (old)
            if hasattr(entry, "transform"):
                scaler = entry
            else:
                scaler_dict = entry
                scaler_type = list(scaler_dict.keys())[0]
                scaler = scaler_dict[scaler_type]
            if hasattr(scaler, "transform"):
                y_scaled[:, i] = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
            else:
                print("WARNING: fallback to apply_output_scaler for output variable:", var)
                y_scaled[:, i] = apply_output_scaler(y[:, i], scaler_type if 'scaler_type' in locals() else None, scaler)

    # Create model
    model_name = args.get("mlp_model", "pointnet")
    hidden = args.get("mlp_hidden", 128)
    # MoE
    if "moe" in model_name:
        num_experts = args.get("moe_num_experts", 4)
        top_k = args.get("moe_top_k", 2)
        model = get_model(model_name, input_dim=X.shape[1], hidden=hidden, output_dim=y.shape[1], num_experts=num_experts, top_k=top_k)
    else:
        model = get_model(model_name, input_dim=X.shape[1], hidden=hidden, output_dim=y.shape[1])

    print(f"Model: {model_name} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    epochs = args.get("mlp_epochs", 10)
    batch_size = args.get("mlp_batch_size", 1024)
    lr = args.get("mlp_lr", 1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.get("output_directory", "./output")

    trained_model = train_mlp(
        X=X_scaled,
        y=y_scaled,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        output_dir=output_dir,
        drop_last=True
    )

    # === Compute R² on scaled training data ===
    with torch.no_grad():
        y_pred_scaled = trained_model(torch.tensor(X_scaled, dtype=torch.float64)).cpu().numpy()

    r2_scaled = {var: sma.r2(y_scaled[:, j], y_pred_scaled[:, j])
                 for j, var in enumerate(output_vars)}


    # Pretty report (unified with grid path)
    _print_r2_report(r2_scaled, title="Training fit quality (Neural)")

    print("\n" + "="*60)
    print("MLP TRAINING COMPLETE")
    print("="*60 + "\n")

    # Attach metrics to the returned model for convenience
    trained_model._r2_scaled = r2_scaled

    return trained_model

# ============================================================
# Scaler Helper Functions (reused from grid-based path)
# ============================================================
def apply_input_scaler(x, scaler_type, config):
    x = np.asarray(x, dtype=np.float64)
    if scaler_type == "minmax":
        xmin, xmax = config.get("range") or (-1.0, 1.0)
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-12:
            return np.full_like(x, (xmin + xmax) / 2)
        return xmin + (x - x_min) * (xmax - xmin) / (x_max - x_min)

    elif scaler_type == "log10":
        factor = config.get("factor") or 1e-10
        return np.log10(np.maximum(x, factor))

    elif scaler_type == "symlog":
        return np.sign(x) * np.log1p(np.abs(x))

    elif scaler_type == "compress":
        factor = config.get("factor") or 1e-10
        compressor = config.get("compressor") or 0.01
        return np.sign(x) * np.log1p(np.abs(x) / factor) * compressor

    else:
        return x


def apply_output_scaler(y, scaler_type, config):
    y = np.asarray(y, dtype=np.float64)
    if scaler_type == "minmax":
        ymin, ymax = config.get("range") or (-1.0, 1.0)
        y_min, y_max = np.min(y), np.max(y)
        if y_max - y_min < 1e-12:
            return np.full_like(y, (ymin + ymax) / 2)
        return ymin + (y - y_min) * (ymax - ymin) / (y_max - y_min)

    elif scaler_type == "log10":
        factor = config.get("factor") or 1e-10
        return np.log10(np.maximum(y, factor))

    elif scaler_type == "symlog":
        return np.sign(y) * np.log1p(np.abs(y))

    elif scaler_type == "compress":
        factor = config.get("factor") or 1e-10
        compressor = config.get("compressor") or 0.01
        return np.sign(y) * np.log1p(np.abs(y) / factor) * compressor

    else:
        return y

def apply_output_scaler_inverse(y, scaler_type, config):
    """Inverse of apply_output_scaler (used during inference)."""
    y = np.asarray(y, dtype=np.float64)

    if scaler_type == "minmax":
        ymin, ymax = config.get("range") or (-1.0, 1.0)
        y_min, y_max = np.min(y), np.max(y)   # fallback if not stored
        if y_max - y_min < 1e-12:
            return np.full_like(y, (ymin + ymax) / 2)
        return y_min + (y - ymin) * (y_max - y_min) / (ymax - ymin)

    elif scaler_type == "log10":
        factor = config.get("factor") or 1e-10
        return np.power(10, y)

    elif scaler_type == "symlog":
        return np.sign(y) * (np.exp(np.abs(y)) - 1)

    elif scaler_type == "compress":
        factor = config.get("factor") or 1e-10
        compressor = config.get("compressor") or 0.01
        return np.sign(y) * (np.exp(np.abs(y) / compressor) - 1) * factor

    else:
        return y

def sm_train(roi: dict, mesh_specs: dict, args : dict, DATA : dict = None, data_files_common_name : str = None):
    """
    Calibrate the nodal values of a connected element surrogate model on training data-set
    from physics-based simulations.

    This method relies on the scripts in utils:
        - utils.fe_shapes     :    shape function methods, e.g., mesh generator, shape function builder, etc.
        - utils.sm_analyze    :    analysis methods, e.g., sizing 
        - utils.sm_check      :    checks during run
        - utils.sm_data       :    data loading, organizing/structuring, storing

    Inputs:
        roi: dict
            Region of interest for each input variables
        mesh_specs: dict
            node_numbers: number of nodes per input variable
            mesh_ratio: boundary ratio of elements, e.g., quadratically larger (quadraticvec) 
                        or larger near roi edges (boundratiovec)
        args: dict
            Dictionary of other arguments used in the function that are less frequently changed by the user.
                - data_dictionary : str
                    place where physics-based data is stored:
                - output_dictionary : str
                    place where surrogate model will be saved
                - max_length : int
                    limit of simulation length of the loaded physics-based simulations
                - map_output : bool
                    map output variable True/False
                - debug : bool
                    print more output for debuggin True/False
        data : dict (Default = None)
            data dictionary in case multiple surrogates need
            rapid testing on the same dataset. If no data dictionary is provided,
            the framework will try to load data. If data dictionary is available 
            from a previous run, it can be reloaded to the training framework here.
        data_files_common_name: string
            Common name part of the data-set containing training data from physics-based simulations.
            Needed when no DATA dict is provided, and data needs to be extracted

    Outputs:   
        output: dict
                - input space of utilized data-set
                - output from physics based data-set 
                - approximated solution for the training data-set
                - R-squared values, condition number of M matrix
                - meta_data related to data-set
        SM: dict
            contains the outputs in terms of 
                - nodal values (calibrated)
                - mesh : nodal coordinates, connectivity
                - mesh_specs : specification of how mesh was built
    
    The outputs are also stored in a .pickle-file within the dir: "../sm_workdir/output" or otherwise specified by args
    """
    
    print("\nInitializing Training of Piecewise-Polynomial Response Surface in ND")

    ## ============================
    # DATA COLLECTION
    # -----------------------------
    # print region of interest
    print("\nROI defined:")
    for key, value in roi.items():
        formatted_values = [f"{v:.2e}" for v in value]  # Format values with scientific notation
        formatted_value_str = ", ".join(formatted_values)
        print(f"\t{key}: \t[{formatted_value_str}]")
        sys.stdout.flush()

    # initialize SM dict for storing surrogate model and model details
    SM = {}

    # collect data from pickles with physics-based simulation data
    if DATA is None:
        if data_files_common_name is not None:
            if smd.check_data_location(args['data_directory'], data_files_common_name) == 0:
                raise IOError("\nCheck NOT OK. Break here during INITIALIZAION of sm_train().")
            data, meta_data = smd.load_data(data_files_common_name, roi, args)  
        else:
            raise IOError("No data_files_common_name provided. Not sure which data files to extract from data directory.")
    elif DATA is not None and 'data' in DATA:
        data = DATA['data']
        print("\nBalanced and transformed dataset provided.\nNo new data loaded.\nContinue...")
    else:
        raise IOError("ERROR: provided dataset not a dictionary. Type: {} ".format(type(data)))
    
    # Print some info about loaded data-set
    if 'meta_data' in DATA:
        meta_data = DATA['meta_data']
        print("\nLoaded {} simulations with a total of {:4.2e} observations.".format(len(meta_data['sim_lengths']), len(data['vmJ2'])))
        sys.stdout.flush()
        print("\tSimulation length average: {:4.2f}, and max: {:4.2f}.".format(np.mean(meta_data['sim_lengths']), np.max(meta_data['sim_lengths'])))
        sys.stdout.flush()
        print("\tNumber of ignored empty simulations: {}".format(len(meta_data['empty_sims'])))
        sys.stdout.flush()
        print("\tNaNs removed from: {} sims".format(len(meta_data['nan_sims'])))
        sys.stdout.flush()
        print("\tNegative strain rates removed from: {} sims".format(len(meta_data['negative_strainrate_sims'])))
        sys.stdout.flush()

    # Scale input
    if (args['map_input'] != False and args['map_input'] != None ):
        print("Transform input data")
        sys.stdout.flush()
        data, scalers_in = smd.fit_and_transform_input(data, maps = args['map_input'])
        SM['input_maps'] = scalers_in
    elif (args['map_input'] == False or args['map_input'] == None):
        print("No input transform applied")
        sys.stdout.flush()
        SM['input_maps'] = None

    # Scale output
    if (args['map_output'] != False and args['map_output'] != None ):
        print("Transform output data")
        sys.stdout.flush()
        data, scalers_out = smd.fit_and_transform_output(data, maps=args['map_output'])
        SM['output_maps'] = scalers_out
    elif (args['map_output'] == False or args['map_output'] == None ):
        print("No output transform applied")
        sys.stdout.flush()
        SM['output_maps'] = False
    
    # Double check on nans in mapped data
    nan_keys = smd.check_nans_recursive(data)
    if nan_keys:
        raise IOError(f"In sm_build: NaNs found in mapped data at keys: {nan_keys}")

    # Plot histograms (also if 'plot_histograms' is not provided)
    if args.get('plot_histograms', True):
        smp.datahist(
            copy.deepcopy(data),
            num_inputs=len(roi),
            storepath=args['output_directory'],
            filename="_training",
            dark=False
        )
    # Check if data is consistent: all arrays should be equal in size
    check = smd.check_data_consistency(data)
    if check[0] == 0:
        raise IOError("\nCheck NOT OK.\nBreak here during DATA COLLECTION of sm_train().")
     
    ## ========================================
    # MESH GENERATION
    ## -------------------------------------

    if mesh_specs.get("premade_mesh", None) is not None and len(mesh_specs['element_numbers']) > 1:  

        # --- Always infer dimensionality from node array ---
        ndim_base = np.array(mesh_specs['premade_mesh']['nodes']).shape[1]
        ndim_target = len(mesh_specs['element_numbers'])

        if ndim_target > ndim_base:
            if mesh_specs.get("tri_elements", []):
                print(f"\nExtrude the premade {ndim_base}D triangular mesh into {ndim_target - ndim_base} higher dimensions...")
                nodes, conn = fes.extrude_premade_simplex_mesh_to_nd(mesh_specs, roi)
            else:
                print(f"\nExtrude the premade {ndim_base}D quad mesh into {ndim_target - ndim_base} higher dimensions...")
                nodes, conn = fes.extrude_premade_hypercube_mesh_to_nd(mesh_specs, roi)
        else:
            print("\nLoad 2D premade mesh (no extrusion needed)")
            nodes = np.array(mesh_specs['premade_mesh']['nodes'])
            conn = np.array(mesh_specs['premade_mesh']['conn'])
    else:
        # Generate a mesh
        print("\nGenerate new mesh from specifications...")
        sys.stdout.flush()
        if not mesh_specs.get('tri_elements', []):
            nodes, conn = fes.build_hypercube_mesh(mesh_specs, roi, dim_names=None) # Keep hypercube-only option
        else:
            refined_nodes = mesh_specs.get('refined_nodes', None)
            if refined_nodes is not None and len(refined_nodes) > 0:
                refined_nodes = np.array(refined_nodes)
                # Build mesh using refined nodes directly (total node array provided and retrieved from lasso)
                nodes, conn = fes.build_simplex_extrusion_nd(mesh_specs, roi, nodes=refined_nodes)
            else:
                # Default: regular mesh
                nodes, conn = fes.build_simplex_extrusion_nd(mesh_specs, roi)

        # Empty element highlighting, followed by removal
        # TOFIX: this tries to find and highlight empty elements when highligh_elements is not present in mesh_refinemens, 
        # but it cannot do this in case of higher than 2 dimensionality (it could, but should only focus on the first two dimensions). 
        # else ndim > 2: highlight elements only based on first to dimensions in nodes. Or skip this highlighting of elemenents altogether 
        # for the case of not having a premade mesh...
        highlight_elements = []
        mesh_refinement = mesh_specs.get('mesh_refinement', {})
        if len(mesh_refinement.get('highlight_elements', [])) == 0 and len(mesh_specs['element_numbers']) > 1:
            ndim = nodes.shape[1]
            n_nodes_per_elem = conn.shape[1]

            if ndim == 1:
                # TOADD: density check for 1D and element removal method
                highlight_elements = []  # Skip for 1D
            elif ndim == 2:
                if n_nodes_per_elem == 3:
                    highlight_elements = fes.data_density_in_element_2d_simplex(data, nodes, conn, mesh_specs)
                elif n_nodes_per_elem == 4:
                    highlight_elements = fes.data_density_in_element_2d_hypercube(data, nodes, conn, mesh_specs)
                else:
                    raise ValueError(f"Unsupported 2D element with {n_nodes_per_elem} nodes.")
            else:
                raise ValueError(f"Unsupported mesh dimension: {ndim}")
            
        nodes, conn = fes.remove_elements_by_index(nodes, conn, highlight_elements=highlight_elements, enforce_diagonals=False)

    # Scale the nodes if input maps are supplied
    if (args['map_input'] is not False or args['map_input'] is not None) and SM['input_maps'] is not None:
        sys.stdout.flush()
        nodes = smd.transform_nodes(nodes, SM['input_maps'])
    
    # store mesh in SM dict
    SM['roi'] = roi
    SM['mesh'] = {'nodes' : nodes, 
                  'conn' : conn,
                  'data_density' : None}
    SM['mesh_specs'] = mesh_specs
    if args['map_input'] != False:
        SM['nodes_mapped'] = True
    elif args['map_input'] == False:
        SM['nodes_mapped'] = False
        
    # Build and store R-tree spatial index if available
    if SM['mesh']['nodes'].shape[1] > 1:
        try:
            from utils import sm_rtree
            if sm_rtree.RTREE_AVAILABLE:
                sys.stdout.flush()
                rtree_index = sm_rtree.create_mesh_rtree_index(nodes, conn, mesh_specs)
                SM['mesh']['rtree_index_data'] = rtree_index.to_dict()
                sys.stdout.flush()
            else:
                print("\nWarning: rtree package detected but spatial index not available.")
                sys.stdout.flush()
        except ImportError:
            print("\n Warning: R-tree indexing not available. Install rtree package for faster element lookup.")
            sys.stdout.flush()
    else:
        print("\nDimensions smaller than 2: no R-tree spatial indexing possible. Continue without R-tree indexing.")
    
    

    # # Return SM in case sparse elements have not been removed (only plotted)
    # if return_flag == True:
    #     smd.store_output(SM, args['output_directory'], filename="SM.pickle")
    #     return None, SM
    
    print("\nMesh generated with {} elements and {} nodes.".format(conn.shape[0], nodes.shape[0]))
    sys.stdout.flush()

    ## ========================================
    # BUILD INTERPOLATION MATRIX
    ## -------------------------------------
    start = time.time()

    # build global interpolation matrix
    if args['sparse'] == True:
        print("\nBuild global sparse interpolation matrix...")
        sys.stdout.flush()
        N = fes.fe_shapefunc_sparse_mixed(data, nodes, conn, mesh_specs, debug=args['debug'])
    else:
        raise IOError("No support for fully dense interpolation matrix N. Not sure how to build interpolation matrix (try using args['sparse']=true)")

    end = time.time()
    print("\nGlobal interpolation matrix constructed (in {:4.2f} s).".format(end-start))
    sys.stdout.flush()

    # print size of sparse interpolation matrix and check nans and inf
    if args['sparse'] == True:
        sma.get_sparse_matrix_size(N)
        flagnan = np.isnan(N.data)
        flaginf = np.isinf(N.data)
        if flagnan.any() is True or flaginf.any() is True:
            print("\n\t!WARNING: N matrix contains NaNs or inf!\n")
    else:
        size_N = sma.get_matrix_size(N)
        print("\tWith size: {:2.4f} GB. \t\t\nShape: {}".format(size_N, N.shape))
        sys.stdout.flush()


    ## ==========================
    # CALIBRATION
    ## --------------------------
    print("\nStart linear regression...")
    sys.stdout.flush()
    start = time.time()

    # solver matrix
    M = N.T @ N
    if np.isnan(M.data).any():
        print("WARNING: NaNs in M" )

    # Compute eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(M.toarray())  # For symmetric M
    D = np.diag(eigvals)
    Q = eigvecs
    SM['mesh']['M'] = M.toarray()
    SM['mesh']['Q'] = Q
    SM['mesh']['D'] = D
    if args['debug'] == True:
        SM['mesh']['N'] = N.toarray()


    # Collect output keys dynamically from data
    output_keys = list(data['U'].keys())

    SM['nodal_values'] = {}
    SM['constrained'] = {}
    uh = {}
    r2 = {}

    # Loop over all output keys
    for key in output_keys:
        rhs = N.T @ data['U'][key]

        # Solve
        if args['sparse']: 
            nodal_values = scipy.sparse.linalg.spsolve(M, rhs)  

            if any(np.isnan(nodal_values)):
                print("NaN in computed nodal values. Check density per element..")
                input_vars = list(mesh_specs['element_numbers'].keys())
                data_array = np.vstack([data[var] for var in input_vars]).T  # shape (N_points, D)
                sparse_elements, sparse_centroids = fes.data_density_per_element_mixed(data_array, nodes, conn, mesh_specs, density_thresh=50, verbose=True)
                print("Sparse elements: {}", sparse_elements)
                print("Sparse centroids:\n{}", smd.backtransform_nodes(sparse_centroids, SM['input_maps']))
                # # TOFIX: faster method to decide if elements are empty
                # sparse_elements2, sparse_centroids2 = fes.compute_sparse_elements_from_N(N, nodes, conn, threshold=50, verbose=True)
                # print("Sparse elements2: {}", sparse_elements2)
                # print("Sparse centroids2:\n{}", smd.backtransform_nodes(sparse_centroids2, SM['input_maps']))

            # Constrained regression step
            if args.get("constrain_regression"):
                constraint = args["constrain_regression"].get(key)

                # If constraint is None, skip constrained solve entirely
                if constraint is None:
                    print(f"No constraints set for {key}, skipping bounded solve.")
                else:
                    # Determine bounds
                    lowerbound = constraint.get("lowerbound")
                    upperbound = constraint.get("upperbound")

                    # Replace "min" and "max" tokens with computed defaults
                    if lowerbound == "min":
                        lowerbound = data["U"][key].min() - 0.1*np.abs(data["U"][key].min())
                    elif lowerbound is None:
                        lowerbound = -np.inf  # default if not set

                    if upperbound == "max":
                        upperbound = data["U"][key].max() + 0.1*np.abs(data["U"][key].max())
                    elif upperbound is None:
                        upperbound = np.inf  # default if not set

                    print(f"\nNodal values for {key} constrained between {lowerbound} and {upperbound}")
                    res = lsq_linear(M, rhs, bounds=(lowerbound, upperbound))
                    nodal_values = res.x
                    print(f"\nFinetuned Nodal values: {nodal_values}")


                    SM['constrained'][key] = {
                        "lowerbound": lowerbound,
                        "upperbound": upperbound
                    }

        else:
            if np.linalg.det(M) == 0.0:
                raise IOError("Abort: singular shape function matrix M. Try using sparse solvers.")
            nodal_values = np.linalg.solve(M, rhs)

        # Store nodal values in surrogate model
        SM['nodal_values'][key] = nodal_values

        # Compute approximated response
        uh[key] = N @ nodal_values

        # Compute R²
        r2[key] = sma.r2(data['U'][key], uh[key])

        # Debug prints of matrices
        if args['debug'] == True:
            print(f"\nDebug: rhs for key {key}")
            print(f"rhs shape: {rhs.shape}")
            print("Full rhs array:")
            print(rhs)

            print(f"\nN shape: {N.shape}")
            print("Full N array:")
            print(N.toarray())
            row_sums = N.toarray().sum(axis=1)
            tol = 1e-6
            rows_sum_to_one = np.isclose(row_sums, 1, atol=tol)
            print(f"\nNumber of rows that sum to 1 (within {tol} tolerance): {np.sum(rows_sum_to_one)}")
            print(f"Percentage of rows that sum to 1: {np.mean(rows_sum_to_one) * 100:.2f}%")

            print("\nFull M matrix:")
            print(M.toarray())

            # After calculating nodal_values
            print(f"\nnodal_values shape: {nodal_values.shape}")
            print("Full nodal_values array:")
            print(nodal_values)

            # After calculating uh[key]
            print(f"\nuh[{key}] shape: {uh[key].shape}")
            print(f"Full uh[{key}] array:")
            print(uh[key])

            # Compare uh[key] with original data
            print("\nFull difference array:")
            diff = uh[key] - data['U'][key]
            print(diff)

            # Print original data for comparison
            print(f"\nOriginal data shape: {data['U'][key].shape}")
            print("Full original data array:")
            print(data['U'][key])

            sys.stdout.flush()

    end = time.time()
    print(f"Completed linear regression (in {end-start:4.2f} s).")
    sys.stdout.flush()

    # Compute condition number
    condM = scipy.sparse.linalg.norm(M) if args['sparse'] else scipy.linalg.norm(M)

    print("\nApproximated response computed.")
    sys.stdout.flush()

    
    # Print R² for each output key (using unified helper)
    r2_report = dict(r2)
    r2_report["M_condition"] = condM
    _print_r2_report(r2_report, title="Training fit quality")
    sys.stdout.flush()

    ## ===================================
    # RESTRUCTURE DATA
    ## -----------------
    # deconcatenate data
    if args['deconcat'] is True:
        if 'meta_data' in DATA:
            train_output = smd.deconcatenate_data(data, meta_data, mesh_specs, SM['output_maps'], uh)
            train_output['meta_data'] = meta_data
        else:
            raise IOError("ERROR: Cannot deconcatenating data. No meta data available.")
    else:
        train_output = data
        train_output['Usm'] = {}
        train_output['unmapped']['Usm'] = {}
        for key in uh.keys():
            train_output['Usm'][key] = uh[key]
            train_output['unmapped']['Usm'][key] = smd.backtransform_output(uh[key], SM['output_maps'][key])


    # add meta_data and some other info to surrogate model dict
    train_output['r2'] = r2
    train_output['M_condition'] = condM
    
    # Store data_file path in SM dictionary if available in args
    if 'data_file' in args:
        SM['data_file'] = args['data_file']

    ## =============================
    ## store output dict in pickle:
    ## -----------------------------
    if args['output_directory'] is not None:
        print("\nStore training output and surrogate model in .pickle files...")
        sys.stdout.flush()

        # Specify the directory path (even if it does not exist)
        output_directory = args['output_directory']
        # Create the directories if they don't exist
        os.makedirs(output_directory, exist_ok=True)
        # Create the file path and surrogate model path
        file_path = os.path.join(output_directory, "train_output.pickle")
        SM_path = os.path.join(output_directory, "SM.pickle")
        # Open the file in binary write mode
        with open(file_path, "wb") as file:
            # Dump the dictionary into the file using pickle
            pickle.dump(train_output, file)
        # Open the file in binary write mode
        with open(SM_path, "wb") as file:
            # Dump the dictionary into the file using pickle
            pickle.dump(SM, file)

        print("\tTrain output stored in: {}".format(file_path))
        sys.stdout.flush()
        print("\tSurrogate model stored in: {}".format(SM_path))
        sys.stdout.flush()

    ## =====================================
    ## return output
    ## -------------------------------------
    print("\nTraining completed!")
    sys.stdout.flush()

    return train_output, SM

def sm_test(SM: dict, args : dict, DATA : dict = None, data_files_common_name : str = None):
    ## ===============================================================================
    ## INITIALIZAION: some prints
    ## ------------------------------------------------------------------------------
    print("\nTESTING: time-step a model through a loading history.")

    ## ============================
    # DATA COLLECTION
    # -----------------------------

    # collect data from pickles with physics-based simulation data
    if DATA is None:
        if data_files_common_name is not None:
            if smd.check_data_location(args['data_directory'], data_files_common_name) == 0:
                raise IOError("\nCheck NOT OK. Break here during INITIALIZAION of sm_train().")
            data, meta_data = smd.load_data(data_files_common_name, args['data_directory'], SM['roi'], args)  
        else:
            raise IOError("No data_files_common_name provided. Not sure which data files to extract from data directory.")
    elif DATA is not None and 'data' in DATA:
        data = DATA['data']
        print("\nBalanced and transformed dataset provided.\nNo new data loaded.\nContinue...")
    else:
        raise IOError("ERROR: provided dataset not a dictionary. Type: {} ".format(type(data))) 
    
    # raise IOError("debug")
    # some printed info about loaded data-set
    if 'meta_data' in DATA:
        meta_data = DATA['meta_data']
        print("\nLoaded {} simulations.".format(len(meta_data['sim_lengths'])))
        sys.stdout.flush()
        print("\tSimulation length average: {:4.2f}, and max: {:4.2f}.".format(np.mean(meta_data['sim_lengths']), np.max(meta_data['sim_lengths'])))
        sys.stdout.flush()
        # print("\tNumber of ignored empty simulations: {}".format(len(meta_data['empty_sims'])))
        # sys.stdout.flush()
        # print("\tNaNs removed from: {} sims".format(len(meta_data['nan_sims'])))
        # sys.stdout.flush()
        # print("\tNegative strain rates removed from: {} sims".format(len(meta_data['negative_strainrate_sims'])))
        # sys.stdout.flush()
    
    ## =================================
    ## RUN MODEL
    # ----------------------------------
    print("\nStart validation")
    sys.stdout.flush()
    start = time.time()

    # set sim_range=range(len(data)) or custom for debugging, e.g., sim_range=range(151,171)
    if 'all' in args['sim_range']:
        valid = rate_stepper(data, SM, sim_range=list(data.keys()), args=args)
    elif isinstance(args['sim_range'], np.ndarray):
        valid = rate_stepper(data, SM, sim_range=args['sim_range'], args=args)
    else:
        raise IOError("I do not know which validation runs to execute in smb.sm_test")

    end = time.time()
    print("\nValidation time: {:4.2f}".format(end-start))

    # add the output do a dict
    valid_output = {}
    valid_output['valid'] = valid
    valid_output['data'] = data


    ## =============================
    ## store output dict in pickle:
    ## -----------------------------
    if args['output_directory'] is not None:
        
        print("\nStore validation output in .pickle file...")
        sys.stdout.flush()
        # Specify the directory path (even if it does not exist)
        output_directory = args['output_directory']
        store_name_addon = args['store_name_addon']
        # Create the directories if they don't exist
        os.makedirs(output_directory, exist_ok=True)
        # Create the file path
        file_path = os.path.join(output_directory, "Testing_result_" + store_name_addon + ".pickle")
        # Open the file in binary write mode
        with open(file_path, "wb") as file:
            # Dump the dictionary into the file using pickle
            pickle.dump(valid_output, file)

        print("\tValidation output stored in: {}".format(file_path))
        sys.stdout.flush()

    print("\nValidation completed!")
    sys.stdout.flush()

    return valid_output

def rate_stepper(data: dict, SM: dict, sim_range: range, args: dict | None = None):
    """
    Runner for a thermo-mechanical/irradiation creep simulation.
    The initial conditions are provided to compute strain rate and
    dislocation density rates using a (surrogate) model, 
    by which the values for accumulated strain and 
    dislocation density are updated. 

    Time increments are either adaptively updated or directly provided
    as input.

    Not parallelized.
    """
    print("\nTime-step 'creep' runner...")
    sys.stdout.flush()

    args = args or {}
    use_mlp = args.get("use_mlp", False) or "mlp_model" in SM or "mlp_model_path" in SM

    # ============================================================
    # 1. Prepare rate provider (MLP vs Grid)
    # ============================================================
    if use_mlp:
        print("[rate_stepper] Using MLP inference path")
        import torch
        from utils.mlp_models import get_model

        # Load model
        if "mlp_model" in SM and not isinstance(SM["mlp_model"], dict):
            model = SM["mlp_model"]
        elif "mlp_model_path" in SM:
            model_path = SM["mlp_model_path"]
            checkpoint = torch.load(model_path, map_location="cpu")

            model_type = SM.get("mlp_model_type", "pointnet")   
            input_dim  = len(SM.get("mlp_input_vars", []))
            output_dim = len(SM.get("mlp_output_vars", []))
            hidden     = SM.get("mlp_hidden", 128)

            model = get_model(model_type, input_dim=input_dim, hidden=hidden, output_dim=output_dim)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
        else:
            model = args.get("mlp_model")

        input_vars   = SM.get("mlp_input_vars")  or args.get("mlp_input_vars")
        output_vars  = SM.get("mlp_output_vars") or args.get("mlp_output_vars")
        map_input    = SM.get("input_maps")      or args.get("input_maps", {})
        map_output   = SM.get("output_maps")     or args.get("output_maps", {})

        rate_provider = "mlp"
    else:
        print("[rate_stepper] Using grid-based inference path")
        nodes        = copy.deepcopy(SM['mesh']['nodes'])
        conn         = copy.deepcopy(SM['mesh']['conn'])
        mesh_specs   = copy.deepcopy(SM['mesh_specs'])
        input_maps   = SM.get('input_maps', {})
        output_maps  = SM.get('output_maps', {})
        output_keys  = list(SM['nodal_values'].keys())
        debug        = args.get("debug", False)

        # R-tree (unchanged)
        rtree_index = None
        try:
            from utils import sm_rtree
            if 'rtree_index_data' in SM['mesh']:
                rtree_index = sm_rtree.MeshRtreeIndex.from_dict(
                    SM['mesh']['rtree_index_data'], nodes, conn, mesh_specs)
            else:
                rtree_index = sm_rtree.create_mesh_rtree_index(nodes, conn, mesh_specs)
        except ImportError:
            pass

        rate_provider = "grid"

    # ============================================================
    # 2. Common simulation loop (grid-based logic kept intact)
    # ============================================================
    creep_output = {}
    total_iterations = sum(len(data[i]['t']) for i in data)

    with tqdm(total=total_iterations) as pbar:
        print("Begin simulations...")
        sys.stdout.flush()

        for i in sim_range:
            starttime = time.time()
            sim_id = i

            if rate_provider == "grid":
                input_keys = list(SM['mesh_specs']['element_numbers'].keys())
                output_keys = list(SM['nodal_values'].keys())
            else:
                input_keys = input_vars
                output_keys = output_vars

            this_simulation = smd.initialize_this_simulation(SM, data[i])

            clipped_rhoc = False
            t = 0
            while_condition = True

            while while_condition:
                pbar.update(1)
                pbar.set_description(
                    f"progress {i+1}/{sim_range[-1]} - "
                    f"{this_simulation['t'][-1]}/{np.cumsum(data[i]['dt'])[-1]}"
                )
                sys.stdout.flush()

                # ------------------------------------------------
                # RATE GENERATION (only difference between paths)
                # ------------------------------------------------
                if rate_provider == "mlp":
                    # --- MLP rate prediction ---
                    current_input = np.array(
                        [this_simulation[var][-1] for var in input_vars],
                        dtype=np.float64
                    )

                    # input scaling
                    for j, var in enumerate(input_vars):
                        if var in map_input:
                            scaler = map_input[var]
                            if hasattr(scaler, "transform"):
                                current_input[j] = scaler.transform(
                                    np.array([[current_input[j]]])
                                ).item()

                    # forward pass
                    with torch.no_grad():
                        X_tensor = torch.tensor(current_input.reshape(1, -1),
                                                dtype=torch.float64)
                        y_pred = model(X_tensor).cpu().numpy().flatten()

                    # inverse scaling of rates
                    outputs = {}
                    for j, var in enumerate(output_vars):
                        rate = y_pred[j]
                        if var in map_output:
                            scaler = map_output[var]
                            if hasattr(scaler, "inverse_transform"):
                                rate = scaler.inverse_transform(
                                    np.array([rate])
                                ).item()
                        outputs[var] = np.array([rate])

                else:
                    # --- Grid-based rate prediction ---
                    this_input_point = []
                    this_input_point_dict = {}
                    for key in input_keys:
                        this_input_point_dict[key] = np.array([this_simulation[key][-1]])
                        this_input_point.append(
                            input_maps[key].transform(
                                np.array(this_simulation[key][-1]).reshape(1, -1)
                            )[0][0]
                        )
                    this_input_point_transformed_dict = smd.transform_input(
                        copy.deepcopy(this_input_point_dict), input_maps
                    )

                    node_range, Iel = fes.find_element_mixed(
                        this_input_point, nodes, conn, mesh_specs=mesh_specs,
                        i=sim_id, verbose=debug, rtree_index=rtree_index
                    )
                    if node_range is False:
                        break

                    N, dN = fes.eval_shape_mixed_point(
                        this_input_point_transformed_dict,
                        nodes[conn[Iel, :], :],
                        mesh_specs,
                        debug
                    )

                    outputs = {}
                    for key in output_keys:
                        outputs[key] = N @ SM['nodal_values'][key][conn[Iel, :]]
                        if output_maps is not False:
                            outputs[key] = smd.backtransform_output(
                                outputs[key], output_maps[key]
                            )

                # ------------------------------------------------
                # FROM HERE ON: EXACT SAME ACCUMULATION LOGIC
                # ------------------------------------------------
                dt = this_simulation['dt'][-1]
                updates = {
                    key: outputs[key][0] * dt
                    for key in output_keys if key in input_keys
                }
                new_values = {}
                for key in output_keys:
                    if key in input_keys:
                        new_values[key] = this_simulation[key][-1] + updates[key]

                    else:
                        print("\nWARNING: No known input variable aligns with updates from output {}.".format(key))


                # adaptive time stepping, shrinking, bounds checking, etc.
                # (the rest of the original grid-based code stays unchanged)
                shrink = 0
                if args.get("adaptive_time", False) is not False:
                    
                    while (shrink < 5 and dt > 1e-12 and
                           any(new_values[key] > SM['roi'][key][1] or
                               new_values[key] < SM['roi'][key][0]
                               for key in new_values if key in SM['roi'])):
                        shrink += 1
                        dt *= args['adaptive_time']['shrink_factor']
                        this_simulation['dt'][-1] = dt
                        for key in updates:
                            updates[key] = outputs[key][0] * dt
                            if key in input_keys:
                                new_values[key] = this_simulation[key][-1] + updates[key]

                if args.get("adaptive_time", False) == False:
                    if t < (len(data[i]['dt']) - 1):
                        this_simulation['dt'].append(copy.deepcopy(data[i]['dt'][t + 1]))
                elif 'growth_factor' in args.get('adaptive_time', {}):
                    if shrink == 0:
                        growth_factor = args['adaptive_time']['growth_factor']
                        this_simulation['dt'].append(this_simulation['dt'][-1] * growth_factor)
                    else:
                        this_simulation['dt'].append(this_simulation['dt'][0])

                # Bounds checking / clipping (unchanged)
                clipped_rhoc = None
                if rate_provider == 'grid':
                    # only applicable for the grid-based method 
                    for key in updates.keys():
                        if key not in SM['roi'] or key not in output_keys:
                            continue
                        val = new_values[key]
                        if val < SM['roi'][key][0]:
                            if args.get("correct_negatives", True):
                                new_values[key] = SM['roi'][key][0]
                                clipped_rhoc = True
                        elif val > SM['roi'][key][1]:
                            if args.get(f'correct_large_{key}', False):
                                new_values[key] = SM['roi'][key][1]
                            else:
                                while_condition = False

                # Stop on infinite values
                for k in new_values:
                    if np.isinf(new_values[k]):
                        while_condition = False

                # update time, dt, counters, etc. (unchanged)
                while_condition = while_condition and (
                    this_simulation['t'][-1] < data[i]['t'][-1]
                )
                if args.get("adaptive_time", False) is False:
                    while_condition = while_condition and t < (len(data[i]['t']) - 1)
                elif 'max_inc' in args['adaptive_time']:
                    while_condition = while_condition and t < args['adaptive_time']['max_inc']

                try:
                    time_accum = this_simulation['t'][-1] + this_simulation['dt'][t]
                except IndexError:
                    break

                for key in new_values.keys():
                    this_simulation[key].append(new_values[key])
                for key in output_keys:
                    this_simulation['U'][key].append(outputs[key][0])
                if while_condition:
                    # update the other (not-coupled) inputs.
                    for key in input_keys:
                        if key not in new_values and key not in ['t']:
                            # make sure not to include 't' if it's also part of 'input_keys'. 't' will be added below separately.
                            this_simulation[key].append(data[i][key][t+1])
                this_simulation['t'].append(time_accum)
                this_simulation['shrink'] = shrink
                this_simulation['simulation_time'] = time.time() - starttime
                this_simulation['clipped_rhoc'] = clipped_rhoc
                t += 1

            creep_output[sim_id] = copy.deepcopy(this_simulation)
            print("Done simulating.")
            sys.stdout.flush()

    return creep_output
