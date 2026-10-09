#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Finite element surrogate model build methods.

sm_train:      calibration of finite element nodal values
sm_test:   testing of calibrated model
run_creep: subprocess for creep simulation

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

from utils import fe_shapes as fes
from utils import sm_analyze as sma
from utils import sm_data as smd
from utils import sm_plot as smp

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

        print(f"--> ndim_target: {ndim_target}, ndim_base: {ndim_base}")

        if ndim_target > ndim_base:
            if mesh_specs.get("tri_elements", []):
                print(f"\nExtrude the premade {ndim_base}D triangular mesh into {ndim_target - ndim_base} higher dimensions...")
                nodes, conn = fes.extrude_premade_simplex_mesh_to_nd(mesh_specs, roi)
            else:
                print(f"\nExtrude the premade {ndim_base}D quad mesh into {ndim_target - ndim_base} higher dimensions...")
                nodes, conn = fes.extrude_premade_hypercube_mesh_to_nd(mesh_specs, roi)
                # debug prints:
                print(mesh_specs['premade_mesh']['nodes'])
                print("nodes[:,0] min : {}, max :{}".format(nodes[:,0].min(), nodes[:,0].max()))
                print("nodes[:,1] min : {}, max :{}".format(nodes[:,1].min(), nodes[:,1].max()))
                print("nodes[:,2] min : {}, max :{}".format(nodes[:,2].min(), nodes[:,2].max()))
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
        print("\nTransform nodal coordinates.")
        sys.stdout.flush()
        nodes = smd.transform_nodes(nodes, SM['input_maps'])
    
    # debug prints
    print("nodes: {}".format(nodes))

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
                print("\nBuilding R-tree spatial index for future queries...")
                sys.stdout.flush()
                rtree_index = sm_rtree.create_mesh_rtree_index(nodes, conn, mesh_specs)
                SM['mesh']['rtree_index_data'] = rtree_index.to_dict()
                print("R-tree index built and stored in surrogate model.")
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

    
    # Print R² for each output key
    for key in output_keys:
        print(f"\tR-squared {key}: \t{r2[key]:4.2f}")

    print(f"\tCondition number M: \t{condM:4.2f}")
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
        valid = run_creep(data, SM, sim_range=list(data.keys()), args=args)
    elif isinstance(args['sim_range'], np.ndarray):
        valid = run_creep(data, SM, sim_range=args['sim_range'], args=args)
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

def run_creep(data : dict, SM : dict, sim_range : range, args : dict | None = None):
    """
    Runner for a thermo-mechanical/irradiation creep simulation.

    The initial conditions are provided to compute strain rate and
    dislocation density rates using a (surrogate) model, 
    by which the values for accumulated strain and 
    dislocation density are updated. 

    Time increments are either adapatively updated or directly provided
    as input.

    Not parallelized.
    """
    print("\nTime-step 'creep' runner...")
    sys.stdout.flush()


    # Prep some containers
    nodes = copy.deepcopy(SM['mesh']['nodes'])
    conn = copy.deepcopy(SM['mesh']['conn'])
    mesh_specs = copy.deepcopy(SM['mesh_specs'])
    debug = args.get("debug", False)
    creep_output = {}
    args = {} if args is None else args # make sure args is a dict if not provided
    
    # Create or restore R-tree spatial index if available
    rtree_index = None
    try:
        from utils import sm_rtree
        # Check if we have a stored index in the surrogate model
        if 'rtree_index_data' in SM['mesh']:
            print("Restoring R-tree spatial index from surrogate model...")
            sys.stdout.flush()
            rtree_index = sm_rtree.MeshRtreeIndex.from_dict(
                SM['mesh']['rtree_index_data'],
                nodes, conn, mesh_specs
            )
            print("R-tree index restored successfully.")
        else:
            print("Building R-tree spatial index for fast element lookup...")
            sys.stdout.flush()
            rtree_index = sm_rtree.create_mesh_rtree_index(nodes, conn, mesh_specs)
            print("R-tree index built successfully.")
        sys.stdout.flush()
    except ImportError:
        print("Warning: R-tree indexing not available. Using standard element lookup.")
        sys.stdout.flush()
    

    # total nr of simulations
    total_iterations = sum(len(data[i]['evm']) for i in data)
    # tqdm: wait bar update
    with tqdm(total=total_iterations) as pbar:
        # # empty temporary dict reset
        # this_valid = {}

        print("Begin simulations...")
        sys.stdout.flush()

        # loop over simulations

        for i in sim_range:      
            starttime = time.time()
            sim_id = i    
            
            # Extract input keys relevant to the surrogate model input domains
            input_keys = list(SM['mesh_specs']['element_numbers'].keys())
            output_keys = list(SM['nodal_values'].keys())

            # Dict of this current simulation only: pre-fill with initial values from data (the input provided to the simulation)
            this_simulation = smd.initialize_this_simulation(SM, data[i]) 

            clipped_rhoc = False
            # reset time increment counter for each new simulation i
            t = 0
            while_condition = True
            while while_condition:
                # Update the progress bar
                pbar.update(1)
                pbar.set_description(f"progress {i+1}/{sim_range[-1]} - {this_simulation['t'][-1]}/{np.cumsum(data[i]['dt'])[-1]}")  
                sys.stdout.flush()

                this_input_point = []
                this_input_point_dict = {}
                for key in input_keys:
                    this_input_point_dict[key] = np.array([this_simulation[key][-1]])
                    this_input_point.append(SM['input_maps'][key].transform(copy.deepcopy(np.array(this_simulation[key][-1])).reshape(1,-1))[0][0])
                this_input_point_transformed_dict = smd.transform_input(copy.deepcopy(this_input_point_dict), SM['input_maps'])
                # raise IOError("Debug")
            
                if args.get("use_rtree", True) == False:
                    node_range, Iel = fes.find_element_mixed(this_input_point, nodes, conn, mesh_specs=SM['mesh_specs'], 
                                                             i=sim_id, verbose=debug, rtree_index=None)
                else:
                    node_range, Iel = fes.find_element_mixed(this_input_point, nodes, conn, mesh_specs=SM['mesh_specs'], 
                                                             i=sim_id, verbose=debug, rtree_index=rtree_index)

                if args.get("debug", False) == True:
                    print("Active element: {}".format(Iel))

                if node_range == False:
                    this_input_point_backscaled = []
                    for i, key in enumerate(SM['input_maps'].keys()):
                        this_input_point_backscaled.append(smd.backtransform_input([copy.deepcopy(this_input_point[i])], SM['input_maps'][key])[0])
                    print("\nWARNING: No element at input:\n\tPhysical units: {}.\n Scaled units: {}. \nROI: {}. \nSTOP simulation #.\n".format(this_input_point_backscaled, this_input_point, SM['roi']))

                    break
                
                # New: much faster
                N, dN = fes.eval_shape_mixed_point(this_input_point_transformed_dict,
                           nodes[conn[Iel, :], :],
                           SM['mesh_specs'],
                           args.get("debug", False))
                
                if args.get("debug", False) == True:
                    print("Nonzero shape function entries (N):")
                    for ind, val in enumerate(N.flatten()):
                        if abs(val) > 1e-10:
                            print(f"  Index {ind}: N = {val}")
            
                # compute outputs only nodal values of the relevant nodes belonging to element in which data point falls
                outputs = {}
                doutputs = {}
                for key in output_keys:
                    outputs[key] = N @ SM['nodal_values'][key][conn[Iel, :]] 
                    # dy*/dx* in mapped space should be chain ruled: dy/y* * dy*/dx* * dx*/dx
                    doutputs[key] = dN @ SM['nodal_values'][key][conn[Iel, :]]           
                    if np.isnan(outputs[key]):
                        raise IOError("Debug: NaN detected in output (unscaled).")                   
                    
            
                # back-transform output to physical space
                if SM['output_maps'] is not False:
                    sys.stdout.flush()
                    dy_dy_scaled = {}       # dy/dy*
                    for key in outputs:
                        outputs[key] = smd.backtransform_output(outputs[key], SM['output_maps'][key])
                        if debug == True:
                            print("outputs['{}']: \n{}".format(key, outputs[key]))
                        dy_dy_scaled[key] = smd.backtransform_output(copy.deepcopy(outputs[key]), SM['output_maps'][key], derivative=True)
                    sys.stdout.flush()

                # compute dx*/dx:
                dvmJ2_scaled_dvmJ2 =  smd.backtransform_input(copy.deepcopy(this_input_point_transformed_dict['vmJ2']), SM['input_maps']['vmJ2'], derivative=True)

                # raise IOError("debug: break here to check outputs")

                # compute full derivative chain rule: dy/dx = dy/dy* * dy*/dx* * dx*/dx:
                doutputs_dvmj2 = {}
                for key in outputs:
                    doutputs_dvmj2[key] = dy_dy_scaled[key] * doutputs[key] * dvmJ2_scaled_dvmJ2

                    if args.get("debug", False) == True:
                        print("doutputs_dvmj2['{}']: \n{}".format(key, doutputs_dvmj2[key]))

                        print("\ndy_dy_scaled['{}']: \n{}".format(key, dy_dy_scaled[key]))
                        print("doutputs['{}']: \n{}".format(key,  doutputs[key]))
                        print("dvmJ2_scaled_dvmJ2: \n{}".format(dvmJ2_scaled_dvmJ2))
                # raise IOError("doutputs_dvmj2['evm']: \n{}".format(doutputs_dvmj2['evm']))
                sys.stdout.flush()

                if args.get('apply_discrepancy_correction', False) == True:
                    from discrepancy_model_predict import DiscrepancyModel
                    if "_discrepancy_model_obj" not in args:
                        disc_dir = args.get("discrepancy_model_dir", "discrepancy_model")
                        args["_discrepancy_model_obj"] = DiscrepancyModel(disc_dir)
                    model = args["_discrepancy_model_obj"]
                    corr_fac = model.predict_factor(
                        this_input_point_transformed_dict['unmapped']['temperature'][0],
                        this_input_point_transformed_dict['unmapped']['vmJ2'][0],
                        this_input_point_transformed_dict['unmapped']['evm'][0],
                        this_input_point_transformed_dict['unmapped']['rhoc'][0],
                        # float(outputs['evm'][0]),                       # wrong way
                        # float(outputs['rhoc'][0])                       # wrong way
                        evm_rate_sur=float(outputs['evm'][0]),        # right way
                        rhoc_rate_sur=float(outputs['rhoc'][0])       # right way
                    )
                    outputs['evm'] = outputs['evm'] * corr_fac
                    print("Correction Factor applied after strain rate computation by Discrepancy Model")
                    sys.stdout.flush()

                # compute updates using surrogate rates
                dt = this_simulation['dt'][-1]
                updates = {key: outputs[key][0] * dt for key in output_keys if key in input_keys}
                new_values = {}
                for key in output_keys:
                    if key in input_keys:
                        new_values[key] = this_simulation[key][-1] + updates[key]
                    else:
                        print("\nWARNING: No known input variable aligns with updates from outputs. Input update ignored.")
                
                # Force updates with reference values from input "data", only when no adaptive time-stepping is used.
                # raise IOError("debug")
                if args.get("adaptive_time", False) == False:
                    
                    for key in input_keys:
                        flag = f'force_{key}'
                        end_point = f'{flag}_end_point'
                        if args.get(flag, False) == True and this_simulation[key][-1] < args[end_point] and (t < len(data[i]['t']) - 1):
                            print(f"force {key} update")
                            new_values[key] = data[i][key][t+1]

                # Shrink time step if out of ROI bounds
                # (shrink max 5 times and never below a time step of 1e-12 (hardcoded here))
                shrink = 0
                while args.get("adaptive_time", False) is not False and shrink < 5 and dt > 1e-12 and any(new_values[key] > SM['roi'][key][1] or new_values[key] < SM['roi'][key][0] for key in new_values.keys() if key in SM['roi']):
                    shrink += 1
                    dt *= args['adaptive_time']['shrink_factor']
                    this_simulation['dt'][-1] = dt
                    print(f"\nCut time-step in half. \n\tNew time-step: {dt}.\n\tShrink = {shrink}.\n\tTime inc: {t}")
                    sys.stdout.flush()
                    for key in updates:
                        updates[key] = outputs[key][0] * dt
                        if key in input_keys:
                            new_values[key] = this_simulation[key][-1] + updates[key]
                        else:
                            print("\nWARNING: No known input variable aligns with updates from outputs. Input update ignored.")
                
                # bounds checking and corrections (surrogate cannot extrapolate)
                for key in updates.keys():
                    # only correct input variables from output rates (not all inputs)
                    if key not in SM['roi'] or key not in output_keys:
                        continue
                    val = new_values[key]
                    # handle negatives 
                    # if val < 0.0:
                    
                    if val < SM['roi'][key][0]:
                        if args.get("correct_negatives", True):
                            new_values[key] = SM['roi'][key][0]
                            clipped_rhoc = True
                            print(f"WARNING: negative {key} set to lowest roi value: {new_values[key]:2.1e}. Continue...")
                        else:
                            print(f"WARNING: Negative {key} computed. Will Fail next time-step...")
                    # handle overshoots 
                    elif val > SM['roi'][key][1]:
                        
                        if args.get(f'correct_large_{key}', False) == True:
                            new_values[key] = SM['roi'][key][1]
                            print(f"WARNING: Large {key} set to upper bound value: {new_values[key]:2.1e}. Continue...")
                        else:
                            # raise IOError("debiug")
                            print(f"WARNING: Large out of bounds {key} computed. Abort.")
                            while_condition = False

                # stop at infinite values
                for k in new_values:
                    if np.isinf(new_values[k]):
                        print("\nWARNING: Infinite value copmuted for {}. Stop simulation.".format(k))
                        while_condition = False

                # Grow next time step, depending on whether adaptive time-stepping is used
                grow = False
                if args.get("adaptive_time", False) == False:
                    if t < (len(data[i]['dt'])-1):
                        # when no adaptive time stepping is used, use the time-step provided in the input "data"
                        this_simulation['dt'].append(copy.deepcopy(data[i]['dt'][t+1]))
                elif 'growth_factor' in args['adaptive_time'].keys() and shrink == 0:
                    # if no shrink occurred, it is safe to grow the next increment
                    growth_factor = args['adaptive_time']['growth_factor']
                    this_simulation['dt'].append(this_simulation['dt'][-1] * growth_factor)
                    grow = True
                    print("\nGrow next time increment by {}".format(this_simulation['dt'][-1] * growth_factor))
                    sys.stdout.flush()
                elif 'growth_factor' in args['adaptive_time'] and shrink > 0:
                    # if shrink occurred, reset to use the initial time-step for the next increment
                    this_simulation['dt'].append(this_simulation['dt'][0])               
            
                # Update the while condition to stop or continue with a next iteration
                while_condition = while_condition and (this_simulation['t'][-1] < data[i]['t'][-1])  # data[i]['t'][-1] is one time-step ahead of this_simulation['t'][-1]
                if args.get("adaptive_time", False) == False:
                    while_condition = while_condition and t < (len(data[i]['t'])-1)
                elif 'max_inc' in args['adaptive_time'].keys():
                    while_condition = while_condition and t < args['adaptive_time']['max_inc']

                # Store the accumulated time (after dt has been shrunk, and has been appended already, hence: [-2])
                try:
                    # When data is shorter than 3 elements, this fails
                    time_accum = this_simulation['t'][-1] + this_simulation['dt'][t]
                except IndexError:
                    # Break out of the wile loop
                    print("Simulation completed.\n\tReached final time increment.")
                    break
                # raise IOError("debug")
                # Store updated values
                for key in new_values.keys():
                    this_simulation[key].append(new_values[key])
                for key in output_keys:
                    this_simulation['U'][key].append(outputs[key][0])
                    this_simulation['dUdvmJ2'][key].append(doutputs_dvmj2[key][0][0])
                if while_condition:
                    for key in input_keys:
                        if key not in new_values:
                            this_simulation[key].append(data[i][key][t+1])
                this_simulation['t'].append(time_accum)
                this_simulation['shrink'] = shrink
                this_simulation['grow'] = grow
                this_simulation['num_incs'] = t
                this_simulation['simulation_time'] = time.time() - starttime
                this_simulation['clipped_rhoc'] = clipped_rhoc
                t += 1

            # Store result in the larger dictionary creep_output
            creep_output[sim_id] = copy.deepcopy(this_simulation)
            print("Done simulating.")
            sys.stdout.flush()
            
    return creep_output
