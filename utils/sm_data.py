#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================
@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Functions to load, store, structure data for the surrogate model.

Loading physics based data-sets (e.g., pickle files)
Storing surrogate model output
Structuring and organizing data-sets in specific formats 

"""
import numpy as np
import pandas as pd
import os, sys, time, glob, copy, pickle, re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple
import numpy as np

def augment_data_columns(data, add_variable_name="defect_rate", inject_value=0.0):
    """
    This method augments per-sim data with entire columns with constant values.

    Specify the DATA, the variable key to add, and the constant value to add.
    """
    print("Augmenting data.")
    for sim in data.keys():
        if add_variable_name not in data[sim].keys():
            sim_length = len(data[sim]['t'])
            data[sim][add_variable_name] = inject_value * np.ones(sim_length)
        else:
            print("Variable to be added is already present. Do nothing. Return original data")

    return data
            

def augment_piecewise_data(
    DATA: Dict[str, Any],
    nodes: List[float],
    min_points: int = 3,
    max_points_per_sim: int = 500,
    remove_initial_below_max_key: str = "vmJ2",
) -> Dict[str, Any]:
    """
    Augments all simulations in `DATA` along the 'evm' axis using piecewise linear
    interpolation, guaranteeing at least `min_points` per node interval **and**
    that every augmented series reaches the final node.

    Parameters
    ----------
    DATA : dict
        { "data": {sim_id: {...}}, "meta_data": {...} }
    nodes : array-like
        Node positions (must be sorted, include first and last).
    min_points : int
        Minimum number of points inside each node interval.
    max_points_per_sim : int
        Hard cap on points after augmentation (uniform re-sample if exceeded).
    remove_initial_below_max_key : str, default "vmJ2"
        Key used to detect first occurrence of maximum value → remove all points before.
        If set to None, step is skipped.

    Returns
    -------
    new_DATA : dict
        Same structure as input, with augmented data and updated metadata.
    """
    data = DATA["data"]
    meta_data = DATA["meta_data"].copy()
    meta_data["augmented"] = True

    new_data: Dict[int, Dict[str, Any]] = {}
    new_sim_lengths: List[int] = []
    augmentation_counts: Dict[int, int] = {}

    nodes_arr = np.asarray(nodes, dtype=float)
    if nodes_arr.size < 2:
        raise ValueError("`nodes` must contain at least two values.")

    total_sims = len(data)
    for idx, (sim_id, sim) in enumerate(data.items(), 1):
        print(f"Processing simulation {sim_id} ({idx}/{total_sims})...")

        # ------------------------------------------------------------------
        # 1. Trim initial increments below first maximum of selected key
        #    (with small relative tolerance)
        # ------------------------------------------------------------------
        TOLERANCE_DEFAULT = 1e-4   # 0.01% of max value

        if remove_initial_below_max_key in sim:
            arr = np.asarray(sim[remove_initial_below_max_key], dtype=float)
            if arr.size > 0:
                max_val = arr.max()
                tolerance = max_val * TOLERANCE_DEFAULT if max_val != 0 else 1e-12

                # Consider value as "maximum" if within tolerance
                idx_near_max = np.where(arr >= max_val - tolerance)[0]

                if idx_near_max.size > 0:
                    cut_idx = idx_near_max[0]  # earliest point that is close enough to max
                    if cut_idx > 0:
                        print(f"  Dropping first {cut_idx} increments "
                              f"(before first point ≥ max({remove_initial_below_max_key}) - {TOLERANCE_DEFAULT*100:.3g}%)")
                        for key, val in sim.items():
                            if isinstance(val, (list, np.ndarray)):
                                sim[key] = val[cut_idx:]
                            elif isinstance(val, dict):
                                for sk, sv in val.items():
                                    if isinstance(sv, (list, np.ndarray)):
                                        val[sk] = sv[cut_idx:]
                else:
                    print(f"  Warning: Could not find any point near max in {remove_initial_below_max_key}")
            else:
                print(f"  Warning: {remove_initial_below_max_key} array is empty. No data removed before the max point.")
        else:
            print(f"  Warning: Key '{remove_initial_below_max_key}' not found – keeping all")

        evm_orig = np.asarray(sim.get("evm", []), dtype=float)
        if evm_orig.size == 0:
            print(f"  Warning: sim {sim_id} has no points – skipping.")
            continue

        # ------------------------------------------------------------------
        # 2. Build the augmented evm grid
        # ------------------------------------------------------------------
        evm_set: set[float] = set(evm_orig)          # original points (fast lookup)

        for i in range(len(nodes_arr) - 1):
            left, right = nodes_arr[i], nodes_arr[i + 1]
            in_interval = evm_orig[(evm_orig >= left) & (evm_orig <= right)]

            if len(in_interval) < min_points:
                # add interior points only – endpoints are added later
                new_pts = np.linspace(left, right, min_points)
                evm_set.update(new_pts)

        # Force **all** node values into the grid (critical for final node)
        evm_set.update(nodes_arr)

        evm_aug = np.sort(np.array(list(evm_set), dtype=float))

        # ------------------------------------------------------------------
        # 3. Optional hard cap on total points
        # ------------------------------------------------------------------
        if len(evm_aug) > max_points_per_sim:
            print(
                f"\033[33m  Warning: sim {sim_id} has {len(evm_aug)} points → "
                f"downsampling to {max_points_per_sim}\033[0m"
            )
            
            # Calculate stride needed to get close to max_points_per_sim
            stride = max(1, len(evm_aug) // max_points_per_sim)
            
            # Use stride-based sampling to preserve original data distribution
            # Also ensure we include the first and last points which are critical
            indices = np.concatenate([
                [0],  # First point
                np.arange(stride, len(evm_aug) - 1, stride),  # Middle points with stride
                [len(evm_aug) - 1]  # Last point
            ])
            
            # If we still have too many points, further reduce while keeping first/last
            if len(indices) > max_points_per_sim:
                # Keep first and last point, evenly select from the middle
                middle_indices = np.linspace(
                    1, len(evm_aug) - 2, max_points_per_sim - 2, dtype=int
                )
                indices = np.concatenate([[0], middle_indices, [len(evm_aug) - 1]])
            
            # Ensure we keep the node values in the subsampled data
            node_indices = [np.argmin(np.abs(evm_aug - node)) for node in nodes_arr]
            indices = np.unique(np.concatenate([indices, node_indices]))
            
            # If we still exceed max_points_per_sim, do a final trim but keep endpoints and nodes
            if len(indices) > max_points_per_sim:
                # Sort indices and identify which ones are not endpoints or nodes
                sorted_indices = np.sort(indices)
                critical_mask = np.zeros_like(sorted_indices, dtype=bool)
                critical_mask[0] = True  # First point
                critical_mask[-1] = True  # Last point
                for node_idx in node_indices:
                    idx_pos = np.searchsorted(sorted_indices, node_idx)
                    if idx_pos < len(sorted_indices) and sorted_indices[idx_pos] == node_idx:
                        critical_mask[idx_pos] = True
                
                # From non-critical points, select enough to reach max_points_per_sim
                non_critical = sorted_indices[~critical_mask]
                if len(non_critical) > 0:
                    num_to_keep = max(0, max_points_per_sim - np.sum(critical_mask))
                    if num_to_keep > 0:
                        keep_indices = np.linspace(0, len(non_critical) - 1, num_to_keep, dtype=int)
                        keep_non_critical = non_critical[keep_indices]
                        indices = np.sort(np.concatenate([sorted_indices[critical_mask], keep_non_critical]))
                    else:
                        indices = sorted_indices[critical_mask]
                else:
                    indices = sorted_indices[critical_mask]
                    
                # Final safety check - if we still have too many, just take the first max_points
                if len(indices) > max_points_per_sim:
                    indices = indices[:max_points_per_sim-1]
                    # Always keep the last point
                    indices = np.append(indices, len(evm_aug) - 1)
            
            evm_aug = evm_aug[indices]

        # ------------------------------------------------------------------
        # 4. Interpolate every field onto the new grid
        # ------------------------------------------------------------------
        new_sim: Dict[str, Any] = {"evm": evm_aug}

        for key, val in sim.items():
            if key == "evm":
                continue

            # Special handling for displacement dictionary "U"
            if key == "U":
                if not isinstance(val, dict):
                    print(f"  Warning: 'U' is not a dict in sim {sim_id} → skipping")
                    continue

                new_U: Dict[str, np.ndarray] = {}
                for u_key, u_val in val.items():
                    u_arr = np.asarray(u_val)

                    if u_arr.size == 0:
                        new_U[u_key] = np.full_like(evm_aug, np.nan)
                        continue

                    if u_arr.size == 1:
                        new_U[u_key] = np.full_like(evm_aug, u_arr.flat[0])
                        continue

                    if u_arr.size == evm_orig.size:
                        new_U[u_key] = np.interp(evm_aug, evm_orig, u_arr)
                    else:
                        new_U[u_key] = np.full_like(evm_aug, u_arr.flat[0])

                new_sim["U"] = new_U
                continue

            # ── Normal fields ────────────────────────────────────────────────
            arr = np.asarray(val)

            if arr.size == 0:
                new_sim[key] = np.full_like(evm_aug, np.nan)
                continue

            if arr.size == 1:
                new_sim[key] = np.full_like(evm_aug, arr.flat[0])
                continue

            if arr.size == evm_orig.size:
                new_sim[key] = np.interp(evm_aug, evm_orig, arr)
            else:
                new_sim[key] = np.full_like(evm_aug, arr.flat[0])

        new_data[sim_id] = new_sim

        # ------------------------------------------------------------------
        # 5. Metadata
        # ------------------------------------------------------------------
        orig_len = len(evm_orig)
        aug_len = len(evm_aug)
        new_sim_lengths.append(aug_len)
        augmentation_counts[sim_id] = aug_len - orig_len
        print(
            f"  Original: {orig_len} → Augmented: {aug_len} "
            f"(+{augmentation_counts[sim_id]})  "
            f"span: [{evm_aug[0]:.3f}, {evm_aug[-1]:.3f}]"
        )

    # ----------------------------------------------------------------------
    # Final assembly
    # ----------------------------------------------------------------------
    meta_data["sim_lengths"] = new_sim_lengths
    meta_data["augmentation_counts"] = augmentation_counts

    return {"data": new_data, "meta_data": meta_data}

def trim_creep_dataset(DATA, trim_variable=None, trim_max_threshold=np.inf):
    """
    Keep only simulations where ALL values of trim_variable <= trim_max_threshold.
    If ANY value violates the condition, the entire simulation is removed.
    """

    data = DATA['data']
    meta_data = DATA.get('meta_data', {})

    # Determine which variable to base trimming on
    if trim_variable is not None:
        trim_var = trim_variable
    else:
        first_sim = next(iter(data.values()))
        trim_var = next(iter(first_sim.keys()))

    removed_sims = []

    # Evaluate each simulation
    for sim in list(data.keys()):
        sim_data = data[sim]

        values = np.asarray(sim_data[trim_var])
        mask = values <= trim_max_threshold

        # If ANY False appears → remove the whole sim
        if not mask.all():
            removed_sims.append(sim)
            data.pop(sim)

    # ---- Update metadata ----
    DATA['data'] = data
    meta_data['job_number'] = list(range(len(data)))

    # Still assumes each sim has 't'
    meta_data['sim_lengths'] = [len(s['t']) for s in data.values()]

    DATA['meta_data'] = meta_data

    print(f"Removed {len(removed_sims)} simulations (partial violation): {removed_sims}")

    return DATA


def curate_dataset(DATA, evm_max=None, U_evm_max=None, downsample_max_length=250, downsample_factor=None):
    """
    Curates the dataset in place, optionally trimming by max values for 'evm' and rejecting
    whole simulations by 'U.evm'.

    Useful for Legacy (HT9) VPSC datasets from A. Tallman.

    Parameters
    ----------
    DATA : dict
        Dictionary containing 'data' and 'meta_data'.
    evm_max : float or None
        Maximum allowed value for sim_data['evm']; None means no trimming.
    U_evm_max : float or None
        Maximum allowed value for sim_data['U']['evm']; None means no trimming.
    downsample_max_length : int
        Maximum number of timesteps after downsampling.
    downsample_factor : int
        Factor to downsample by
    Returns
    -------
    dict
        Curated DATA dictionary.
    """
    data = DATA['data']
    meta_data = DATA.get('meta_data', {})
    keys_to_pop = ["dt", "rhoc_accum", "rhoc_init", "rhoc_update",
                   "rhow_accum", "rhow_init", "rhow_update"]

    removed_short_sims = []
    removed_large_strain_sims = []

    for sim in list(data.keys()):
        sim_data = data[sim]

        # Remove short simulations
        if sim_data['t'].shape[0] < 5:
            removed_short_sims.append(sim)
            meta_data.setdefault('empty_sim', []).append(sim)
            data.pop(sim)
            continue

        # Keep rhoc/rhow copies
        if 'rhoc_accum' in sim_data.keys():
            sim_data['rhoc'] = sim_data['rhoc_accum']
        if 'rhow_accum' in sim_data.keys():
            sim_data['rhow'] = sim_data['rhow_accum']

        # Drop unused keys
        for k in keys_to_pop:
            sim_data.pop(k, None)

        # --- Trim by evm ---
        if evm_max is not None:
            mask = np.asarray(sim_data['evm']) <= evm_max
            for key, value in sim_data.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if isinstance(subval, (list, np.ndarray)):
                            sim_data[key][subkey] = np.asarray(subval)[mask]
                elif isinstance(value, (list, np.ndarray)):
                    sim_data[key] = np.asarray(value)[mask]

        # --- Reject whole sim by U['evm'] ---
        if U_evm_max is not None:
            if np.any(np.asarray(sim_data['U']['evm']) > U_evm_max):
                removed_large_strain_sims.append(sim)
                meta_data.setdefault('large_strain_rate_sims', []).append(sim)
                data.pop(sim)
                continue

        # Remove short simulations again (after trimming)
        if sim_data['t'].shape[0] < 5:
            removed_short_sims.append(sim)
            meta_data.setdefault('empty_sim', []).append(sim)
            data.pop(sim)
            continue

        # Thinning / downsampling
        if downsample_factor is not None and downsample_factor > 1:
            # Enhanced downsampling: works for integer and fractional factors > 1.
            # Goal: keep approximately total_len / downsample_factor points,
            # while always retaining the first and last points.
            total_len = len(sim_data['evm'])
            # Compute desired number of points after downsampling
            target_len = int(np.ceil(total_len / downsample_factor))
            # Ensure at least two points (first and last)
            target_len = max(target_len, 2)
            # Generate evenly spaced indices between start and end
            indices = np.linspace(0, total_len - 1, num=target_len, dtype=int)
            # Ensure uniqueness and sorted order
            indices = np.unique(indices)
            thinned_indices = indices
        else:
            downsampler = DownSampler(
                max_length=downsample_max_length,
                downsampling_threshold=-20,
                scale_factor=None,
                scale_data=True,
                remove_initial_incs=0
            )
            thinned_indices = downsampler.downsample(sim_data['evm'])
        for key, value in sim_data.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    if isinstance(subval, (list, np.ndarray)):
                        sim_data[key][subkey] = np.asarray(subval)[thinned_indices]
            elif isinstance(value, (list, np.ndarray)):
                sim_data[key] = np.asarray(value)[thinned_indices]

        # Recompute dt
        sim_data['dt'] = np.diff(sim_data['t'])

    # Update meta_data
    DATA['data'] = data
    meta_data['job_number'] = list(range(len(data)))
    meta_data['sim_lengths'] = [len(sim_data['t']) for sim_data in data.values()]
    DATA['meta_data'] = meta_data

    # Print summary
    print(f"Removed {len(removed_short_sims)} short simulations: {removed_short_sims}")
    print(f"Removed {len(removed_large_strain_sims)} large strain rate simulations: {removed_large_strain_sims}")

    return DATA


def check_nans_recursive(obj, parent_key=""):
    nan_keys = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            nan_keys.extend(check_nans_recursive(v, full_key))

    elif isinstance(obj, (np.ndarray, list, tuple)):
        try:
            if np.any(np.isnan(obj)):
                nan_keys.append(parent_key)
        except TypeError:
            pass  # Non-numeric contents (like strings), skip

    else:
        try:
            if np.isnan(obj):
                nan_keys.append(parent_key)
        except TypeError:
            pass  # Not a float-like scalar, skip

    return nan_keys


def filter_data(data : dict, meta_data : dict = None, filter_type : str = "min_length", filter_threshold : int = 2):
    """
    Post process a data dictionary to filter data based on a specified filter type 
    and corresponding threshold value.

    inputs:
        - data : dict
            dict containing data in simulation format, e.g., data[0]['t']
        - meta_data : dict (optional)
            dict containing previously generated meta_data and information. 
            If provided, it will be adapted based on the filtering done here.
        - filter_type : str
            type of filter currently available:
                - min_length : minimum lentgh. Requires integer to use as the min length threshold.
    
    """
    index_to_pop= []
    for i, sim in enumerate(data):
        if filter_type=="min_length" and isinstance(filter_threshold, int) and len(data[sim]['t']) < filter_threshold:
                index_to_pop.append(sim)
                if meta_data is not None and "empty_sims" in meta_data:
                    meta_data['empty_sims'].append(sim)
    index_to_pop=np.array(index_to_pop)

    # Apply index_to_pop to remove simulations from the data
    for idx in sorted(index_to_pop, reverse=True):
        data.pop(idx)

    return data, meta_data

def backtransform_nodes(nodes, scalers):
    """

    Backtransform nodal coordinates to original, physical units, over all input domains. 
    
    Input: 
        nodes : np.ndarray
            contains nodal coordinates
        scalers : dict
            dict with scaler object created during input data transformations
    Output:
        nodes : dict
            transformed nodal coordinates
    """

    for i, key in enumerate(scalers):
        if scalers[key] != None: 
            if isinstance(scalers[key], (StandardScaler, MinMaxScaler)):
                # sklearn scaler input: shape correction 
                nodes[:,i] = scalers[key].inverse_transform(nodes[:,i].reshape(-1,1)).ravel(order="F")
            else:
                nodes[:,i] = scalers[key].inverse_transform(nodes[:,i])

    return nodes

def transform_output(data, scalers):
    """
    Apply transformation mapping to output data: the same ones as have 
    been fit previously, using existing scalers. 

    Input: 
        data : dict
            contains output data
        scalers : dict
            dict containing mapping scaler objects for each input variable (or None)

    Output:
        data : dict
            transformed nodal coordinates
    """
    # copy original values in a separate dict key
    data['unmapped']['U'] = {}
    for key in scalers:
        if key in data['U']:
            data['unmapped']['U'][key] = copy.deepcopy(data['U'][key])

    for key in scalers:
        if scalers[key] != None and key in data['U']:
            if isinstance(scalers[key], (StandardScaler, MinMaxScaler)):
                # sklearn scaler input: shape correction 
                data['U'][key] = scalers[key].transform(data['U'][key].reshape(-1,1)).ravel(order="F")
            else:
                data['U'][key] = scalers[key].transform(data['U'][key])
    
    return data

def transform_input(data, scalers):
    """
    Apply transformation mapping to input data: the same ones as have 
    been fit previously, using existing scalers. 

    Requires order in nodes:
        nodes[:,0] : vmJ2
        nodes[:,1] : temperature
        nodes[:,2] : evm (strain)
        nodes[:,3] : rhoc (cell dislocations)
        nodes[:,4] : rhow (wall dislocations)
        nodes[:,5] : environmental (e.g., neutron flux)

    Input: 
        data : dict
            contains input data
        scalers : dict
            dict containing mapping scaler objects for each input variable (or None)

    Output:
        nodes : dict
            transformed nodal coordinates
    """
    # copy original values in a separate dict key
    data['unmapped'] = {}
    for key in scalers:
        if key in data:
            data['unmapped'][key] = copy.deepcopy(data[key])

    for key in scalers:
        if scalers[key] != None and key in data:
            if isinstance(scalers[key], (StandardScaler, MinMaxScaler)):
                # sklearn scaler input: shape correction 
                data[key] = scalers[key].transform(data[key].reshape(-1,1)).ravel(order="F")
            else:
                # custom scaler takes 1D array as input withouth need for reshape
                data[key] = scalers[key].transform(data[key])
    
    return data

def transform_nodes(nodes, scalers):
    """
    Apply transformation mapping to nodal coordinates: the same ones as have 
    been used to scale the input data. 

    Order in nodes must align with the order of keys in "scalers", e.g.
        nodes[:,0] : vmJ2
        nodes[:,1] : temperature
        nodes[:,2] : evm (strain)
        nodes[:,3] : rhoc (cell dislocations)
        nodes[:,4] : rhow (wall dislocations)
    Input: 
        nodes : np.ndarray
            contains nodal coordinates
        scalers : dict
            dict containing mapping scaler objects for each input variable (or None)

    Output:
        nodes : dict
            transformed nodal coordinates
    """

    for i, key in enumerate(scalers):
        if scalers[key] != None:
            if isinstance(scalers[key], (StandardScaler, MinMaxScaler)):
                # sklearn scaler input: shape correction 
                nodes[:,i] = scalers[key].transform(nodes[:,i].reshape(-1,1)).ravel(order="F")
            else:
                nodes[:,i] = scalers[key].transform(nodes[:,i])
    
    return nodes

def fit_and_transform_input(data : dict, maps : dict):
    """
    Apply transformation mapping to the input data.

    Input: 
        data : dict
            contains data
        maps : dict
            contains mapping strings (or None)

    Output:
        data : dict
            contains data with transformed inputs
        input_map_meta : dict
            contains the used mapping coefficients 
    """
    # copy original values in a separate dict key
    if 'unmapped' not in data:
        data['unmapped'] = {}
    for key in maps:
        if key in data:
            data['unmapped'][key] = copy.deepcopy(data[key])

    # Define different data scalers and transform data.
    # Each if/elif is a new scaler call. Add more here if needed.
    scalers = {}
    for key in maps:
        if key in data and maps[key] is not None:
            if isinstance(maps[key], dict):
                if "log10" in maps[key].keys():
                    scalers[key] = LogScaler(factor=maps[key]['log10']['factor'], lowerbound=maps[key]['log10']['lowerbound'], upperbound=maps[key]['log10']['upperbound'])
                    # transform the data
                    data[key] = scalers[key].fit_transform(data[key])
                elif "symlog" in maps[key].keys():
                    scalers[key] = SymLogScaler(lowerbound=maps[key]['symlog']['lowerbound'], upperbound=maps[key]['symlog']['upperbound'])
                    # transform the data
                    data[key] = scalers[key].fit_transform(data[key])
                elif "compress" in maps[key].keys():
                    # raise IOError("debug")
                    scalers[key] = CompressScaler(factor=maps[key]['compress']['factor'], compressor=maps[key]['compress']['compressor'])
                    # transform the data
                    data[key] = scalers[key].fit_transform(data[key])
                elif "minmax" in maps[key]:
                    if maps[key]['minmax']['range'] == None:
                        # Default scales between (0, 1)
                        scalers[key] = MinMaxScaler()
                    elif isinstance(maps[key]['minmax']['range'], tuple):
                        # Speficy range
                        scalers[key] = MinMaxScaler(feature_range=maps[key]['minmax']['range'])
                    elif isinstance(maps[key]['minmax']['range'], list):
                        print("Converting minmax scaler range 'list' to 'tuple'. ")
                        range_val = tuple(maps[key]['minmax'].get("range", (-1.0, 1.0)))
                        scalers[key] = MinMaxScaler(feature_range=range_val)
                    else:
                        raise IOError("I do not know how to create a minmax scaler for input data. Is 'range' required tuple? {}".format(isinstance(maps[key]['minmax']['range'], tuple)))
                    # Transform the data
                    data[key] = scalers[key].fit_transform(data[key].reshape(-1,1)).ravel(order="F")

                else:
                    raise IOError("I do not know how to create a scaler for input data. Scaler type is: {}".format(maps[key]))
            elif isinstance(maps[key], str):
                if "log1p" == maps[key]:
                    scalers[key] = Log1pScaler()
                    # transform the data
                    data[key] = scalers[key].fit_transform(data[key])
                elif "standard" == maps[key]:
                    scalers[key] = StandardScaler()
                    # transform the data
                    data[key] = scalers[key].fit_transform(data[key].reshape(-1,1)).ravel(order="F")
                
                else:
                    raise IOError("I do not know how to create a scaler for input data. Scaler is given as: {}".format(maps[key]))
            else:
                raise IOError("I do not know how to create a scaler for input data. Scaler is given as: {}".format(maps[key]))
            
        elif key in data and maps[key] is None:
            scalers[key] = None
        else:
            raise IOError("I do not know how to create a scaler for input data. {} not in data".format(key))
    
    return data, scalers

def backtransform_input(data : list, scaler, derivative=False):
    """
    Apply back-transformation mapping to the data set

    Input: 
        data : list
            contains data for one input response (not the entire dictionary)
        scaler : scaling object (either scipy.optimize scalers or custom scalers defined in sm_data.py)
            class containing the scaling functions and inverse scaling functions
    Output:
        data : dict
            contains data with transformed outputs
    """

    if scaler is None:
        return data
    elif scaler is not None:
        if isinstance(scaler, (StandardScaler, MinMaxScaler)):
            if derivative:
                return (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
            else:
                return scaler.inverse_transform(np.array(data).reshape(-1,1)).ravel(order="F")
        else:
            # check if derivative computaion is accurate
            return scaler.inverse_transform(np.array(data))
    else:
        raise IOError("I do not know how to back-transform the input: input map not known or unspecified.")
    
def fit_and_transform_output(data : dict, maps : dict):
    """
    Apply transformation mapping to the data set

    Input: 
        data : dict
            contains data
        maps : dict
            contains mapping strings (or None)

    Output:
        data : dict
            contains data with transformed outputs
        scalers : dict
            contains the scaler object per output variable
    """
    # copy original values in a separate dict key
    if 'unmapped' in data.keys():
        data['unmapped']['U'] = {}
    else:
        data['unmapped'] = {}
        data['unmapped']['U'] = {}
    for key in maps:
        if key in data['U']:
            data['unmapped']['U'][key] = copy.deepcopy(data['U'][key])

    # Define different data scalers and transform data, based "maps" dictionary.
    # Each if / elif is a different scalar call. Add more here if necessary.
    scalers = {}
    for key in maps:
        if key in data['U'] and maps[key] != None:
            if isinstance(maps[key], dict): 
                if "log10" in maps[key].keys():
                    scalers[key] = LogScaler(factor=maps[key]['log10']['factor'], lowerbound=maps[key]['log10']['lowerbound'], upperbound=maps[key]['log10']['upperbound'])
                    # transform the data
                    data['U'][key] = scalers[key].fit_transform(data['U'][key])
                elif "symlog" in maps[key].keys():
                    scalers[key] = SymLogScaler(lowerbound=maps[key]['symlog']['lowerbound'], upperbound=maps[key]['symlog']['upperbound'])
                    # transform the data
                    data['U'][key] = scalers[key].fit_transform(data['U'][key])
                elif "compress" in maps[key].keys():
                    scalers[key] = CompressScaler(factor=maps[key]['compress']['factor'], compressor=maps[key]['compress']['compressor'])
                    # transform the data
                    data['U'][key] = scalers[key].fit_transform(data['U'][key])
                elif "minmax" in maps[key]:
                    if maps[key]['minmax']['range'] == None:
                        # Default scales between (0, 1)
                        scalers[key] = MinMaxScaler()
                    elif isinstance(maps[key]['minmax']['range'], tuple):
                        # Speficy range
                        scalers[key] = MinMaxScaler(feature_range=maps[key]['minmax']['range'])
                    elif isinstance(maps[key]['minmax']['range'], list):
                        print("Converting minmax scaler range 'list' to 'tuple'. ")
                        range_val = tuple(maps[key]['minmax'].get("range", (-1.0, 1.0)))
                        scalers[key] = MinMaxScaler(feature_range=range_val)
                    else:
                        raise IOError("I do not know how to create a scaler for output data.")
                    # Transform the data
                    data['U'][key] = scalers[key].fit_transform(data['U'][key].reshape(-1,1)).ravel(order="F")
                else:
                    raise IOError("I do not know how to create a scaler for output data.")
            elif isinstance(maps[key], str):
                if "log1p" == maps[key]:
                    scalers[key] = Log1pScaler()
                    # transform the data
                    data['U'][key] = scalers[key].fit_transform(data['U'][key])
                
                elif "standard" == maps[key]:
                    scalers[key] = StandardScaler()
                    # transform the data
                    data['U'][key] = scalers[key].fit_transform(data['U'][key].reshape(-1,1)).ravel(order="F")
                else:
                    raise IOError("I do not know how to create a scaler for output data.")
            else:
                raise IOError("I do not know how to create a scaler for output data.")
            
        elif key in data and maps[key] == None:
            scalers[key] = None
        else:
            raise IOError("I do not know how to create a scaler for output data.")
        
    # raise IOError("debug")
    return data, scalers
    # return data, output_map_meta

def backtransform_output(data : list, scaler, derivative=False):
    """
    Apply back-transformation mapping to the data set

    Input: 
        data : list
            Data for one output response (not the entire dictionary)
        scaler : scaler object for one output (not dict of scalers)
            Contains scaler object by which the data was transformed
    Output:
        data_backtransformed : dict
            Backtransformed outputs

    """
    if scaler is None:
        return data
    elif scaler is not None: 
        if isinstance(scaler, LogScaler):
            return scaler.inverse_transform(np.array(data), derivative)
        elif isinstance(scaler, (StandardScaler, MinMaxScaler)):
            if derivative:
                return scaler.data_range_ / (scaler.feature_range[1] - scaler.feature_range[0])
            else:
                return scaler.inverse_transform(np.array(data).reshape(-1,1)).ravel(order="F")
        else:
            return scaler.inverse_transform(np.array(data))
    else:
        raise IOError("I do not know how to back-transform the output: scaler not known or unspecified.")
    
        # return data_backtransformed

def load_data(data_jobdir_common_name, roi, args : dict):
    """
    Load data from directories with .csv files containing physics-based simulations.

    Concatenate all the simulations into long np.arrays. 
        ( Retain the original structure in meta_data['sim_lengths'] )

    Input:
        - data_jobdir_common_name : str
            common name part of data files
        - data_directory : str
            directory containing data files
        - roi : dict
            region of interest for each input variable
        - max_length : int
            maximum length of simulation: thin data with more increments
        - args : dict
            Dictionary with additionaly arguments and flags, e.g., concatenate data in long array (concat=True) or not (concat=False)
    
    Output:
        - data : dict 
            data loaded into np.arrays
        - meta_data : dict
            information on the data set
                - sim_lengths : simulation lenghts to reconstruct the original data structure
                - nan_sims : simulation numbers containing nans
                - empty_sims : simulation numbers containing no data
                - output_mapped : has the output been mapped? True/False: filled later outside this function call
    """
    # Extract some variables from "args"
    concat = args['concat']
    max_length = args['data_thinning']['max_length']
    data_directory = args['data_directory']
    input_key_dict = args['input_key_dict']
    output_key_dict = args['output_key_dict']

    # Use glob to get a list of files that match the common name part
    def extract_number(s):
        # Use regular expression to find the numeric part in the string
        match = re.search(r'_(\d+)$', s)
        return int(match.group(1)) if match else float('inf')

    jobdir = sorted(glob.glob(os.path.join(data_directory, f"{data_jobdir_common_name}*")), key=extract_number)

    # Initialize data dict with empty lists to fill with data
    meta_data = {'original_data_directory' : data_directory, 'data_files_common_name' : data_jobdir_common_name, 'job_number' : [], 'sim_lengths' : [], 'nan_sims' : [], 'negative_strainrate_sims' : [], 'negative_rhoc_sims' : [], 'negative_rhow_sims' : [],  'empty_sims' : [], 'out_of_range_sims' : []}
    
    # raise IOError("dbueg")
    if concat is True:
        # For training and testing, where concatenation is necessary
        data = {value: [] for value in input_key_dict.values()}
        # Add the output container 'U'
        data['U'] = {value: [] for value in output_key_dict.values()}
    elif concat is False:
        # for validation set, where concatenation is not appropriate
        data = {}
        cnt = 0

    # Print thinning of data
    if max_length is not None:
        print("\nThin data using maximum length {:4.2f}.".format(max_length))
        sys.stdout.flush()
    elif max_length is None:
        print("\nNo data thinning.")
        sys.stdout.flush()

    # collect data from the pickles
    print("\nCollect data from: '{}':".format(data_directory))
    sys.stdout.flush()
    # raise IOError("debug")
    for j, job in enumerate(jobdir):
        jobnum = int(job.split('_')[-1])
        j+=1
        print('\textracting {} in {} from: {}'.format(j, len(jobdir), job))
        sys.stdout.flush()
        # Load the .csv files from each jobdir. If unsuccessful, flag later as empty simulation base on empty_data_files boolean.
        try:
            data_tmp, files = readcsv(job)
            empty_data_files = False
        except:
            empty_data_files = True

        # Store which sims are empty of data files or data increments, skip storing anything
        if 'simulation_macro_averages.csv' not in data_tmp or empty_data_files==True or len(files) == 0 or len(data_tmp['simulation_macro_averages.csv']['time']) < 1:             
            meta_data['empty_sims'].append(jobnum)
        # If not empty, continue: concatenate inputs and outputs in long lists, keeping track of the length of each simulation. Use lists, and not numpy vstack or concatenate (much slower)   
        else:   
            # Dictionary to store selected columns and new names for each file
            this_input = {}
            # Extract inputs. Process each input file, and store the relevant inputs if they are present in the files
            for file in files:
                for key in input_key_dict:
                    if key in data_tmp[file]:
                        this_input[input_key_dict[key]] = np.array(data_tmp[file][key], dtype=np.float64)

            # Check if all keys were processed into this_input (if not, some variables are missing from the datafiles)
            if len(this_input) < len(input_key_dict):
                raise IOError("Not all requested input variables were found in the output files: \nCheck args['input_key_dict and make sure all variables are present in the files.")
            elif len(this_input) == len(input_key_dict):
                # Restructure to input_key_dict order
                this_input = {key: this_input[key] for key in input_key_dict.values() if key in this_input}

            # Rectify the lengths of the different inputs: make all array lengths equal to the shortest
            # input_to_check = this_input.copy()
            inputs_to_pop_for_check = ['temperature', 'vmJ2', 'prec_dens', 'defect_rate']
            # for key in inputs_to_pop_for_check:
            #     input_to_check.pop(key, None)
            # check_in, this_input_lengths = check_data_consistency(input_to_check)
            check_in, this_input_lengths = check_data_consistency(this_input)
            if check_in == False and this_input_lengths != None:
                # Only check specific entries (strain rate, rhoc, rhow)
                max_increment = min([this_input_lengths[2], this_input_lengths[3], this_input_lengths[4]])
                print("Rectify evolving input variable array lengths to {:d}".format(max_increment))
                for key in this_input:
                    this_input[key] = this_input[key][0:max_increment]
                print("Rectify static input variable arrays lengths to {:d}".format(len(this_input['evm'])))
                for key in inputs_to_pop_for_check:
                    if key not in this_input:
                        print(f"\tWarning: Key '{key}' not found in this_input, skipping.")
                        continue
                    
                    if 'evm' not in this_input:
                        print(f"\tWarning: 'evm' key not found in this_input, cannot perform length correction.")
                        break
                    
                    if len(this_input[key]) < len(this_input['evm']):
                        print(f"\t{key} length correction.")
                        this_input[key] = this_input[key][-1] * np.ones(len(this_input['evm']))

            # Extract outputs from the output files
            this_output = {}
            # Process each file
            for file in files:
                for key in output_key_dict:
                    if key in data_tmp[file]:
                        if key == 'rho_m_total':        
                            dot_rhoc = np.diff(np.array(this_input['rhoc']))/np.diff(this_input['t'])
                            if len(dot_rhoc) == 0:
                                raise IOError("Rhoc rate computation results in empty array with length 0.\nIn file: {}\njob: {}\n len dot_rhoc: {} \nlen_this_input: {}".format(file, job, len(dot_rhoc), len(this_input['rhoc'])))
                            dot_rhoc = np.insert(dot_rhoc, 0, dot_rhoc[0])
                            this_output[output_key_dict[key]] = dot_rhoc
                        elif key == 'rho_cw_total':
                            dot_rhow = np.diff(np.array(this_input['rhow']))/np.diff(this_input['t'])
                            if len(dot_rhoc) == 0:
                                raise IOError("Rhow rate computation results in empty array with length 0.\nIn file: {}\njob: {}\n len dot_rhow: {} \nlen_this_input: {}".format(file, job, len(dot_rhoc), len(this_input['rhoc'])))
                            dot_rhow = np.insert(dot_rhow, 0, dot_rhow[0])
                            this_output[output_key_dict[key]] = dot_rhow
                        else:
                            this_output[output_key_dict[key]] = np.array(data_tmp[file][key])
            
            # Check if all keys were processed into this_input (if not, some variables are missing from the datafiles)
            if len(this_output) < len(output_key_dict):
                raise IOError("Not all requested output variables were found in the outut files: \nCheck args['output_key_dict and make sure all variables are present in the output files.")
            elif len(this_output) == len(output_key_dict):
                # Restructure to input_key_dict order
                this_output = {key: this_output[key] for key in input_key_dict.values() if key in this_output}

            # Rectify the lengths of the different outputs: make all array lengths equal to the shortest
            check_out, this_output_lengths = check_data_consistency(this_output)
            if check_out == False and this_output_lengths != None:   
                max_increment = min(this_output_lengths)
                print("Rectify output array lengths to {:d}".format(max_increment))
                for i, key in enumerate(this_output):
                    this_output[key] = this_output[key][0:max_increment]

            # Check once more if all input and ouput array lengths are equal
            # input_to_check = this_input.copy()
            # inputs_to_pop_for_check = ['temperature', 'vmJ2', 'prec_dens']
            # for key in inputs_to_pop_for_check:
            #     input_to_check.pop(key, None)
            if check_in == False or check_out == False:
                # check_in, this_input_lengths = check_data_consistency(input_to_check)
                check_in, this_input_lengths = check_data_consistency(this_input)
                check_out, this_output_lengths = check_data_consistency(this_output)
                if check_in == False or check_out ==False:
                    # if checks don't pass after rectifying
                    raise IOError("Unsuccessful array length rectification in smd.load_data()")
                else:
                    print("Array length rectification successful. Continue...")
            # filter data within a certain ranges of stress and temperature
            if this_input['vmJ2'][-1] >= roi['vmJ2'][0] and this_input['vmJ2'][-1] <= roi['vmJ2'][1] and this_input['temperature'][-1] >= roi['temperature'][0] and this_input['temperature'][-1] <= roi['temperature'][1]:                    
                
                # store which sims contain nan's in output
                if np.any(np.isnan(this_output['evm'])) or np.any(np.isnan(this_output['rhoc'])):
                    meta_data['nan_sims'].append(jobnum)  
                # store which sims contain negative strain (evm) rates
                if np.any(this_output['evm'] < 0):
                    meta_data['negative_strainrate_sims'].append(jobnum) 
                # store which sims contain negative dislocation content
                if np.any(this_input['rhoc'] < 0):
                    meta_data['negative_rhoc_sims'].append(jobnum)
                    raise IOError("debug rhoc < 0. Filter out this data: add to nonNegative_index (not yet coded)")
                
                # Select some indices based on ROI for evm and checks for NaN and negative values
                # index without NaNs
                nonNaN_list = ~np.isnan(this_output['evm']) & ~np.isnan(this_output['rhoc']) & ~np.isnan(this_output['rhow']) 
                # index without negative strain rate and negative strain
                nonNegative_index = (~(this_output['evm'] < 0) & ~(this_input['evm'] < 0))
                combined_bool_array = nonNaN_list & nonNegative_index
                # index within roi bounds for evm
                index_within_roi = np.where(this_input['evm'] < roi['evm'][1])[0]
                bool_array_roi = np.full_like(nonNaN_list, False) # initialize with all False
                bool_array_roi[index_within_roi] = True # fill with True at locations where within roi
                final_combined_bool_array = combined_bool_array & bool_array_roi
                selected_indices = np.where(final_combined_bool_array)[0]
                
                # Downsampling prep, depending on which domain to use for 
                if max_length is not None:
                    downsample_variable = args['data_thinning']['downsample_variable']
                    if downsample_variable == "evm":
                        data_to_downsample = copy.deepcopy(this_input['evm'][selected_indices])
                    elif downsample_variable in ["time", "t"]:
                        # convert dt incs to time before selecting indices and downsampling
                        data_to_downsample = this_input['time'][selected_indices]                         
                    else:
                        raise IOError("ERROR: Downsampling method unknown in smd.load_data. Proceed without downsampling.")
                    # Downsampling: provide thinned_indices                         
                    try:        
                        downsampler = DownSampler(max_length=max_length, 
                                                downsampling_threshold=args['data_thinning']['downsampling_threshold'], 
                                                scale_factor=None,
                                                scale_data=args['data_thinning']['scale_data'],
                                                remove_initial_incs=args['data_thinning']['remove_initial_incs'])
                        thinned_indices = downsampler.downsample(data_to_downsample)
                    except:    
                        # raise IOError("debug")
                        print("WARNING: Downsampling of data failed in smd.load_data. Proceed without downsampling.")
                        thinned_indices = slice(None)
                else:
                    thinned_indices = slice(None)

                # Collect data and store in new dict "data"               
                if concat is True:                                           

                    # # For training and testing, where concatenation is necessary
                    # data = {key: [] for key in new_input_keys}
                    # # Add the output container 'U'
                    # data['U'] = {key: [] for key in new_output_keys}

                    # Extract inputs and store in data:
                    # input_to_check = this_input.copy()
                    # inputs_to_pop_for_check = ['temperature', 'vmJ2', 'prec_dens']
                    # for key in inputs_to_pop_for_check:
                    #     input_to_check.pop(key, None)
                    # check_in, this_input_lengths = check_data_consistency(input_to_check)
                    check_in, this_input_lengths = check_data_consistency(this_input)
                    for key in this_input.keys():
                        if check_in==True:
                            data[key].extend(this_input[key][selected_indices][thinned_indices])
                        elif check_in==False:
                            raise IOError("Debug: input lengths inconsistent.")
                    
                    # Extract outputs and store in data["U"]
                    for key in this_output.keys():
                        data['U'][key].extend(this_output[key][selected_indices][thinned_indices])              

                # When concat == False simulations are not concatenated and the original data structure is maintained
                elif concat is False:
                    # Valid: fill data, do not concatenate
                    data[jobnum] = {}

                    # Extract inputs and store in data:
                    # input_to_check = this_input.copy()
                    # inputs_to_pop_for_check = ['temperature', 'vmJ2', 'prec_dens']
                    # for key in inputs_to_pop_for_check:
                    #     input_to_check.pop(key, None)
                    # check_in, this_input_lengths = check_data_consistency(input_to_check)
                    check_in, this_input_lengths = check_data_consistency(this_input)
                    for i, key in enumerate(this_input.keys()):
                        if check_in==True:
                            data[jobnum][key] = this_input[key][selected_indices][thinned_indices]
                        elif check_in==False:
                            raise IOError("Debug: input lengths inconsistent.")
                        
                    # Extract time
                    data[jobnum]['dt'] = np.diff(this_input['t'][selected_indices][thinned_indices])

                    # Extract the output
                    data[jobnum]['U'] = {}
                    for i, key in enumerate(this_output.keys()):
                        data[jobnum]['U'][key] = this_output[key][selected_indices][thinned_indices]
                    cnt += 1

                # Store the length of the simulation and the original job number (from the directory name).
                meta_data['sim_lengths'].append(len(this_input['t'][selected_indices][thinned_indices]))
                meta_data['job_number'].append(jobnum)
            else:
                # Not within specified ranges:
                # raise IOError("debug")
                meta_data['out_of_range_sims'].append(j)

    # raise IOError("debug")

    print("DONE.")
    sys.stdout.flush()
    
    # convert lists with data to np.arrays with data type float64
    if concat is True:
        data = convert_lists_to_arrays(data)

    # raise IOError("Debug")
    
    plt.close("all")
    return data, meta_data

def align_to_t(arr, t_len):
        """Pad, truncate, or broadcast scalars to match t_len."""
        if isinstance(arr, (float, int)):  # scalar → broadcast
            return np.full(t_len, arr)
        arr = np.asarray(arr)
        if arr.shape[0] < t_len:  # pad with last value
            pad_val = arr[-1] if arr.size > 0 else 0.0
            arr = np.concatenate([arr, np.full(t_len - arr.shape[0], pad_val)])
        elif arr.shape[0] > t_len:  # truncate
            arr = arr[:t_len]
        return arr

def concatenate_data(DATA: dict, remove_initial_incs=False):
    data = DATA['data']
    meta_data = copy.deepcopy(DATA['meta_data'])  # avoid side-effects

    result = {}
    
    for sim_data in data.values():
        # Reference length from "t"
        t = sim_data["t"]
        if isinstance(remove_initial_incs, int):
            t = t[remove_initial_incs:]
        t_len = len(t)

        for key, value in sim_data.items():
            if isinstance(value, dict):
                # Nested dictionary
                if key not in result:
                    result[key] = {subkey: [] for subkey in value}
                for subkey, subvalue in value.items():
                    if isinstance(remove_initial_incs, int):
                        subvalue = subvalue[remove_initial_incs:]
                    subvalue = align_to_t(subvalue, t_len)
                    result[key][subkey].append(subvalue)
            else:
                # Top-level array, list, or scalar
                if isinstance(remove_initial_incs, int) and not isinstance(value, (float, int)):
                    value = value[remove_initial_incs:]
                # raise IOError("removine {} incs at the start.".format(remove_initial_incs))
                value = align_to_t(value, t_len)
                
                if key not in result:
                    result[key] = []
                result[key].append(value)

    # Convert lists to concatenated arrays
    for key, value in result.items():
        if isinstance(value, dict):
            for subkey in value:
                result[key][subkey] = np.concatenate(value[subkey], axis=0)
        else:
            result[key] = np.concatenate(value, axis=0)

    # Adjust sim_lengths after removing increments
    if remove_initial_incs is not False and isinstance(remove_initial_incs, int):
        meta_data['sim_lengths'] = [
            max(0, l - remove_initial_incs) for l in meta_data['sim_lengths']
        ]

    NEW_DATA = {'data': result, 'meta_data': meta_data}

    return NEW_DATA

def deconcatenate_data(data : dict, meta_data : dict, mesh_specs : dict, output_maps : dict, uh):
    """
    Restructure concatenated data arrays into original format: stored per simulation.

    Input:
        - data : dict
            physics-based data set
        - meta_data : dict
            info on data, incl. simulation lengths
        - args : dict
            Configuration dictionary
        - output_maps : dict
            Dictionary of scaler objects
        - uh : np.array
            approximated output by surrogate model

    Output:
        output_data : dict
            output in the original format of the data-set (deconcatenated)
    """
    print("\nStructure surrogate model into dictionary...")
    sys.stdout.flush()

    # Get get the keys relevant for the surrogate. Output aligns with output maps and inputs with elements.
    input_keys = list(mesh_specs['element_numbers'].keys())
    output_keys = list(output_maps.keys())

    # Initialize empty dictionary to store results
    output_data = empty_structure(data)
    # Add 'Usm' with the same structure as 'U'
    if 'U' in data:
        output_data['Usm'] = empty_structure(data['U'])

    # Add 'unmapped'['Usm'] if 'unmapped' and 'U' are present
    if 'unmapped' in data and 'U' in data['unmapped']:
        if 'unmapped' not in output_data:
            output_data['unmapped'] = {}
        output_data['unmapped']['Usm'] = empty_structure(data['unmapped']['U'])
     
    # Fill the data per simulation, as in the original data-base format
    for i, sim in enumerate(meta_data['job_number']):
        # Inputs:
        for key in input_keys:
            output_data[key][sim] = list(data[key][int(np.sum(meta_data['sim_lengths'][0:i])):np.sum(meta_data['sim_lengths'][0:i+1])])
            if 'unmapped' in data and key in data['unmapped']:
                output_data['unmapped'][key][sim] = list(data['unmapped'][key][int(np.sum(meta_data['sim_lengths'][0:i])):np.sum(meta_data['sim_lengths'][0:i+1])])
            
        # Outputs:
        for key in output_keys:
            output_data['U'][key][sim] = list(data['U'][key][int(np.sum(meta_data['sim_lengths'][0:i])):np.sum(meta_data['sim_lengths'][0:i+1])])
            output_data['Usm'][key][sim] = list(uh[key][int(np.sum(meta_data['sim_lengths'][0:i])):np.sum(meta_data['sim_lengths'][0:i+1])])
            if key in output_maps:
                output_data['unmapped']['U'][key][sim] = list(data['unmapped']['U'][key][int(np.sum(meta_data['sim_lengths'][0:i])):np.sum(meta_data['sim_lengths'][0:i+1])])
                output_data['unmapped']['Usm'][key][sim] = backtransform_output(list(uh[key][int(np.sum(meta_data['sim_lengths'][0:i])):np.sum(meta_data['sim_lengths'][0:i+1])]), output_maps[key])

    print("DONE.")
    sys.stdout.flush()

    return output_data

def deconcatenate_by_dict(data: dict,
                          meta_data: dict,
                          output_maps: dict = None):
    """
    Deconcatenate concatenated arrays in `data` into per-simulation structure.

    Parameters
    ----------
    data : dict
        Dictionary with:
          - input arrays at the top level
          - outputs inside data["U"]
        Each array is concatenated across simulations.
    meta_data : dict
        Must contain:
          - 'job_number': list of simulation IDs
          - 'sim_lengths': list of lengths per simulation
    output_maps : dict, optional
        Optional dict of scaler/map objects indexed by output key.
        If provided, `unmapped` outputs will be produced using each scaler's
        inverse_transform method.

    Returns
    -------
    dict
        Dictionary indexed by simulation ID, each containing inputs and outputs.
    """
    print("\nDeconcatenating data into original simulation structure...")
    sys.stdout.flush()

    total_len = sum(meta_data['sim_lengths'])
    job_numbers = meta_data['job_number']

    # Separate inputs and outputs
    input_keys = [k for k in data.keys() if k != "U"]
    outputs = data["U"]

    # Validation helper — skip "dt" from full-length requirement
    def check_lengths(d, name):
        for k, arr in d.items():
            if k == "dt":
                expected_len = total_len - len(meta_data['sim_lengths'])  # one less per simulation
            else:
                expected_len = total_len
            if np.asarray(arr).shape[0] != expected_len:
                raise ValueError(
                    f"Length mismatch for {name}['{k}']: {len(arr)} != {expected_len}"
                )

    # Validate
    check_lengths({k: data[k] for k in input_keys}, "inputs")
    check_lengths(outputs, "outputs")

    # Prepare result: top level = simulation
    result = {}

    # Slice per simulation
    for i, sim in enumerate(job_numbers):
        start = int(np.sum(meta_data['sim_lengths'][:i]))
        end = int(np.sum(meta_data['sim_lengths'][:i+1]))

        sim_data = {}

        # Fill inputs
        for k in input_keys:
            if k == "dt":
                # dt has one fewer point per simulation
                dt_start = start - i  # adjust because each previous sim had 1 less
                dt_end = dt_start + (meta_data['sim_lengths'][i] - 1)
                sim_data[k] = np.asarray(data[k])[dt_start:dt_end].tolist()
            else:
                sim_data[k] = np.asarray(data[k])[start:end].tolist()

        # Fill outputs
        sim_data["U"] = {}
        if output_maps:
            sim_data.setdefault("unmapped", {})
            sim_data["unmapped"]["U"] = {}

        for k, arr in outputs.items():
            vals = np.asarray(arr)[start:end].tolist()
            sim_data["U"][k] = vals
            if output_maps and k in output_maps:
                scaler = output_maps[k]
                sim_data["unmapped"]["U"][k] = scaler.inverse_transform(
                    np.array(vals).reshape(-1, 1)
                ).ravel().tolist()

        result[sim] = sim_data

    print("DONE.")
    sys.stdout.flush()
    return result

def convert_lists_to_arrays(data):
    """
    Convert lists to np.arrays with float64 values. 

    input "data" may be: 
        - dict containing lists and nested dicts containing lists
        - list
        - np.array in which case this function just returns the original "data" input
    """
    if isinstance(data, dict):
        return {key: convert_lists_to_arrays(value) if isinstance(value, dict) else np.float64(np.array(value)) for key, value in data.items()}
    elif isinstance(data, list):
        # return np.float32(np.array(data))
        return np.float64(np.array(data))
    else:
        return data 

def readcsv(path = './references'):
    """        
    Read data from multiple csv-files and store all data in dictionary
    
        
    Input:
    - path:         The path of a directory specified by user to look for .csv-files.
                    all .csv-files in the specified directory will be read and stored in 
    
    Output:
    - dict_csv:     Dictionary of stored data from all .csv-files
    - files:        the filenames (incl extension) 
    """
    
    # fetch all files in path
    all_files = os.listdir(path)
    all_files.sort()
    
    # filter file name list for files ending with .csv
    csv_files = [file for file in all_files if '.csv' in file]
    
    # pre-allocate list for files
    files = []
    
    # preallocate data arrays
    dict_csv = {}
    
    ## gather and plot the simulated data
    for file in csv_files:
        # read data from csv and store in "data" 
        data = pd.read_csv(os.path.join(path, file), skiprows = 0)
        
        # append files to a list
        files.append(file)
        
        # store each dataframe into a dictionary entry by filename
        dict_csv[file] = data
    
    return dict_csv, files


def check_data_location(data_directory, data_files_common_name):
    """
    Check if directory and data exists in specified directory and common name of data files.
    """
    
    print("\nCheck if data directory exists...")
    sys.stdout.flush()
    if os.path.exists(data_directory) and os.path.isdir(data_directory):
        check = 1
        print("\tOK: the directory '{}' exists.".format(data_directory))
        sys.stdout.flush()
    else:
        check = 0
        raise IOError("\nThe data directory '{}' does not exist.\nMake sure to create the directory: '{}'.".format(data_directory, data_directory))
        sys.stdout.flush()

    # Check if anyb file in the data directory contains the common name part
    matching_files = [file for file in os.listdir(data_directory) if data_files_common_name in file]

    if matching_files:
        check = 1
        print("\tOK: data files found.")
        sys.stdout.flush()
    else:
        check = 0
        raise IOError("\nNo data files exist with the specified common name '{}'.\nCheck specified common file name and make sure to place data-files in '{}'.".format(data_files_common_name, data_directory))
    
    return check

def check_data_consistency(data):
    """
    Check if all data entries have the same length.
    """
    print("\nCheck data consistency...")
    sys.stdout.flush()

    # define array length check for nested dict 
    def check_arrays_length(data):
        if isinstance(data, dict):
            array_lengths = [len(value) for value in data.values() if isinstance(value, np.ndarray)]
            nested_lengths = [check_arrays_length(value) for value in data.values() if isinstance(value, dict)]      
            condition_check = len(set(array_lengths)) == 1 and all(nested_lengths)
            # raise IOError("debug: {}".format(array_lengths))
            return condition_check, array_lengths
        elif isinstance(data, np.ndarray):
            return True, None
        else:
            return True, None

    # check if length of all data entries are equal
    check, array_lengths = check_arrays_length(data)
        
    # raise error if inconsistent
    if int(check) == True:
            print("\tOK: array lengths are consistent.")
            sys.stdout.flush()
    elif int(check) == False:
        raise IOError("\nData is inconsistent: not all data entries have the same length.")
    else:
        raise IOError("smd.check_array_lengths: array length check invalid (not true or false)")

    return int(check), array_lengths

def store_output(output, output_directory, filename="output.pickle"):
    """
    Store output in output_directory, using filename

    """
    if output_directory is not None:
        sys.stdout.flush()

        # Create the directories if they don't exist
        os.makedirs(output_directory, exist_ok=True)

        # Create the file path and surrogate model path
        file_path = os.path.join(output_directory, filename)

        # Open the file in binary write mode
        with open(file_path, "wb") as file:
            # Dump the dictionary into the file using pickle
            pickle.dump(output, file)
        print("\n{} stored in: {}".format(filename, file_path))
        sys.stdout.flush()
    else:
        print("WARNING: storage location unknown. Output not stored.")


def empty_structure(d):
        """
        Copy a dictionary structure into a new empty dictionary.
        """
        if isinstance(d, dict):
            return {k: empty_structure(v) for k, v in d.items()}
        else:
            return {}  # Replace all leaves with empty dicts or lists
        
def initialize_this_simulation(SM, data_i):
    this_simulation = {}

    # Get input and output variable names from SM
    input_keys = list(SM['mesh_specs']['element_numbers'].keys())
    output_keys = list(SM['nodal_values'].keys())
    time_keys = ['t', 'dt']

    # Initialize time stepping values
    for key in time_keys:
        try:
            this_simulation[key] = [data_i[key][0]]
        except KeyError:
            raise IOError(f"Key '{key}' not found in the current simulation data.")
        except IndexError:
            raise IOError(f"Key '{key}' found, but the list is empty for this simulation.")

    # Initialize input values with those from data_i
    for key in input_keys:
        this_simulation[key] = [data_i[key][0]]

    # Initialize outputs under 'U' with np.zeros(1)
    this_simulation['U'] = {}
    this_simulation['dUdvmJ2'] = {}
    for key in output_keys:
        this_simulation['U'][key] = [0.0]
        this_simulation['dUdvmJ2'][key] = [0.0]

    # Simulation meta fields
    this_simulation['shrink'] = None
    this_simulation['grow'] = False
    this_simulation['num_incs'] = None
    this_simulation['simulation_time'] = 0.0

    return this_simulation

## ================================================   
# Scalers:
## ------------------------------------------------

class LogScaler:
    """
    Scaler for shifted base-10 logarithmic transformation,
    followed by linear scaling into [lowerbound, upperbound].

    Ensures that transform(0.0) == 0.0 ⇔ inverse_transform(0.0) == 0.0.
    """

    def __init__(self, factor=None, lowerbound=None, upperbound=None):
        """
        Parameters:
        - factor (float): Small shift added to avoid log10(0); defaults to min positive X on fit.
        - lowerbound (float): Optional minimum of the scaled range.
        - upperbound (float): Optional maximum of the scaled range.
        """
        self.factor = factor
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.logmin = None
        self.logmax = None
        self.a = None
        self.b = None

    def _validate_input(self, X):
        """Ensure input is NumPy array, even if scalar."""
        if np.isscalar(X):
            return np.array([X])
        return np.asarray(X)

    def fit(self, X):
        """Fit scaler parameters based on input data X."""
        X = self._validate_input(X)

        # NaN-safe min if needed: np.nanmin, np.nanmax
        if self.factor is None:
            nonzero_X = X[X > 0.0]
            if nonzero_X.size == 0:
                raise ValueError("Cannot infer factor: no positive entries in X.")
            self.factor = np.min(nonzero_X)

        self.logmin = np.log10(np.min(X) + self.factor)
        self.logmax = np.log10(np.max(X) + self.factor)

        if self.lowerbound is None:
            self.lowerbound = self.logmin
        if self.upperbound is None:
            self.upperbound = self.logmax

        self.a = self.upperbound - self.lowerbound
        log0 = np.log10(self.factor)

        # Ensure transform(0.0) = 0.0. Used to be self.b = self.lowerbound
        self.b = -self.a * (log0 - self.logmin) / (self.logmax - self.logmin)

        return self
    
    def transform(self, X):
        """Apply logarithmic transform followed by linear scaling."""
        X = self._validate_input(X)
        transformed = self.a * (np.log10(X + self.factor) - self.logmin) / (self.logmax - self.logmin) + self.b
        return transformed
        # return transformed if transformed.size > 1 else transformed.item()

    def inverse_transform(self, X, derivative=False):
        """Invert the transform (optionally return the derivative of the inverse)."""
        X = self._validate_input(X)
        if derivative:
            u = 10 ** ((self.logmax - self.logmin) * (X - self.b) / self.a + self.logmin)
            dX = u * np.log(10) * (self.logmax - self.logmin) / self.a
            # return dX if dX.size > 1 else dX.item()
            return dX 
        inv = 10 ** ((X - self.b) * (self.logmax - self.logmin) / self.a + self.logmin) - self.factor
        
        # Correct for zero: zero is zero. Handle scalar vs. array inputs safely
        if np.isscalar(X) or inv.shape == ():
            return 0.0 if np.isclose(X, 0.0) else inv
        else:
            inv[np.isclose(X, 0.0)] = 0.0
            return inv

    def fit_transform(self, X):
        """Convenience method: fit and transform in one step."""
        return self.fit(X).transform(X)

class Log1pScaler:
    """
    Scaler for shifted NATURAL logarithmic log10(x+1) data transform, and its inverse.
    """
    def __init__(self):
        pass

    def fit(self, X):
        return self
    
    def transform(self, X):
        return np.log1p(X)

    def inverse_transform(self, X):
        return np.expm1(X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class CompressScaler:
    """
    Custom scaler for data tranforming dislocation density rates.
    
    """
    def __init__(self, factor=None, compressor=None):
        self.factor = factor
        self.compressor = compressor
        self.original_min = None  # Placeholder for original minimum value

    def fit(self, X):
        return self
    
    def transform(self, X):
        # Compress and scale the data
        X_compressed = np.sign(X) * np.abs(np.array(X) * self.factor)**self.compressor
        # Compute the original minimum value if it's not already set
        if self.original_min is None:
            self.original_min = X_compressed.min()
        # Scale the data 
        X_scaled = np.log10(1 + X_compressed - self.original_min)
        return X_scaled

    def inverse_transform(self, X):
        # Inverse scale the data
        X_unscaled = 10**np.array(X) + self.original_min - 1

        # Undo compression
        X_uncompressed = np.sign(X_unscaled) * np.abs(X_unscaled) ** (1/self.compressor)

        # Undo sign and factor scaling
        X_backtransformed = X_uncompressed / self.factor

        return X_backtransformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class SymLogScaler:
    '''
    Apply signed log transform to 1d array:
    
    log(1+x) if x >= 0
    -log(1-x) if x<0
    
    (this is equivalent to np.sign(x)*np.log(1+ np.abs(x)) )
    
    followed by linear scaling into [-a,a]
    '''
    def __init__(self, lowerbound = -1.0, upperbound = 1.0):
        self.zmin = None
        self.zmax = None
        self.zbar = None
        self.lowerbound = lowerbound
        self.upperbound = upperbound
            
    def fit(self, X : np.array):
        assert(len(X.shape)==1)
        Z = np.sign(X) * np.log10(1.0 + np.abs(X))
        self.zmin = Z.min()
        self.zmax = Z.max()
        self.zbar = 0.5*(self.zmin + self.zmax)
        return self
            
    def transform(self, X: np.array):
        # assert(len(X.shape)==1)
        Z = np.sign(X) * np.log10(1.0 + np.abs(X))
        return (self.upperbound - self.lowerbound) * (Z - self.zbar)/(self.zmax - self.zmin)

    def forward_derivative(self, X: np.array):
        dZ = 1 / ((1.0+np.abs(X))*np.log(10))
        return (self.upperbound - self.lowerbound) / (self.zmax - self.zmin) * dZ
            
    def inverse_transform(self, Z: np.array, derivative=False):
        assert(len(Z.shape)==1) # raises AssertionError if not True
        if derivative:
            X = (self.zmax - self.zmin) * Z/(self.upperbound - self.lowerbound) + self.zbar
            U = 10**( np.abs(X) )         
            return U * np.log(10) * (self.zmax - self.zmin)/(self.upperbound - self.lowerbound)
        X = (self.zmax - self.zmin) * Z / (self.upperbound - self.lowerbound) + self.zbar
        return np.sign(X) * (10**( np.abs(X) ) - 1.0) 

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
## --------------------------------  
## Downsamplers ==================
# -------------
class DownSampler:
    def __init__(self, max_length=250, downsampling_threshold=None, scale_factor=None, scale_data=False, remove_initial_incs=0):
        self.max_length = max_length
        self.downsampling_threshold = downsampling_threshold
        self.scale_factor = scale_factor
        self.scale_data = scale_data
        self.remove_initial_incs = remove_initial_incs

    def downsample(self, data):
        thinned_indices = slice(None)
        if self.max_length is not None:
            # First remove initial increments when considering the (transformed/scaled) data
            if self.scale_data==True:
                scaled_data = self.data_scaler(data[self.remove_initial_incs:])
            else:
                scaled_data = data[self.remove_initial_incs:]

            # Downsample equivalently (balanced) on both sides of the threshold, after initial incs are removed
            threshold_index = np.argmin(np.abs(scaled_data - self.downsampling_threshold))         
            if len(scaled_data[:threshold_index]) >= self.max_length:
                step_below_threshold = np.linspace(scaled_data[0], scaled_data[threshold_index-1], self.max_length)
            else:
                step_below_threshold = scaled_data[0:threshold_index]
            if len(scaled_data[threshold_index:-1]) >= self.max_length:
                step_above_threshold = np.linspace(scaled_data[threshold_index], scaled_data[-1], self.max_length)
            else:
                step_above_threshold = scaled_data[threshold_index:]
            step = np.append(step_below_threshold, step_above_threshold)
            if len(step) > 0:
                thinned_indices = np.unique([np.argmin(np.abs(scaled_data - value)) for value in step])
                if thinned_indices[-1] >= len(scaled_data):
                    print("Rectify length of indices")
                    sys.stdout.flush()
                    thinned_indices[-1] = len(scaled_data) - 1 
            else:
                 thinned_indices = slice(None)
        
            # Map indices back to original array (add back number of removed initial incs to indices)
            if thinned_indices is not slice(None):
                thinned_indices = thinned_indices + self.remove_initial_incs

        elif len(data) < self.max_length or self.max_length is None:
            thinned_indices = slice(None)
        return thinned_indices

    def data_scaler(self, data):
        """
        Local scaler (log) within the downsampler.
        """
        scaler = LogScaler(factor=self.scale_factor)
        scaled_data = scaler.fit_transform(data)
        return scaled_data