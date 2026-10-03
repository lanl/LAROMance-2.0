#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Utility library for evaluating surrogate model accuracy and testing results.

"""
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import interp1d
from scipy.spatial import distance
import os, sys, copy
import pandas as pd

def maxcurve_index(x, y):
    """
    Find the index corresponding to the maximum curvature in data y(x).

    """
    x_normalized = (x - x.min())/(x.max() - x.min())
    y_normalized = (y - y.min())/(y.max() - y.min())
    tangent_data = abs(np.gradient(y_normalized, x_normalized))
    thresh = np.argmin(np.abs(tangent_data-1.0))
    return thresh

def datamap(data : dict, x = 'vmJ2', y = 'temperature', z='evm', inc=-1):
    """
    Map specified z from multiple validation simulations across x and y.

    Input:
        - data : dict
            Output response (surrogate model simulation), coming from validation simulations.
            Originally, this could be, e.g., valid_output['data'] for physics-based or 
            valid_output['valid'] for SM output.
        - x : str, default = 'vmJ2' 
            x variable of 2D domain
        - y : str, default = 'temperature'
            y variable of 2D domain
        - z : str, default = 'evm'
            z variable to map across 2D domain
        - inc : int or str 
            at which increment to analyze the data. 
            Strings: "max", "min", "mean" are also viable.
    
    Output:
        - masked_z : dict
            
    """

    print("Extracting map from data...")
    sys.stdout.flush()

    # initialize empty numpy arrays to collect data in
    X = np.empty((0, 1))    
    Y = np.empty((0, 1))    
    Z = np.empty((0, 1))  

    # # set key for data extraction
    # if z == 'rhoc' and data[sim].keys():
    #     z_key_data = 'rhoc_accum'
    # elif z == 'rhow':
    #     z_key_data = 'rhow_accum' 

    # loop over all the simulations in the dictionaries
    for sim in data:
        # stack to the the numpy arrays    
        if isinstance(inc, int):
            if np.array(data[sim][x]).size > 1:
                X = np.vstack([X, np.array(data[sim][x])[inc]]) 
            elif np.array(data[sim][x]).size == 1:
                X = np.vstack([X, np.array(data[sim][x])]) 
            else:
                raise IOError("No x data at specified inc")
            if np.array(data[sim][y]).size > 1:
                Y = np.vstack([Y, np.array(data[sim][y])[inc]])
            elif np.array(data[sim][y]).size == 1:
                Y = np.vstack([Y, np.array(data[sim][y])])
            else:
                raise IOError("No y data at specified inc")
            if np.array(data[sim][z]).size > 1:
                Z = np.vstack([Z, np.array(data[sim][z])[inc]])
            elif np.array(data[sim][z]).size == 1:
                Z = np.vstack([Z, np.array(data[sim][z])])
            else:
                raise IOError("No z data at specified inc")
        elif isinstance(inc, str):
            if inc == "max":
                X = np.vstack([X, np.array(data[sim][x]).max()]) 
                Y = np.vstack([Y, np.array(data[sim][y]).max()])
                Z = np.vstack([Z, np.array(data[sim][z]).max()])
            elif inc == "min":
                X = np.vstack([X, np.array(data[sim][x]).min()]) 
                Y = np.vstack([Y, np.array(data[sim][y]).min()])
                Z = np.vstack([Z, np.array(data[sim][z]).min()]) 
            elif inc == "mean":
                X = np.vstack([X, np.array(data[sim][x]).mean()]) 
                Y = np.vstack([Y, np.array(data[sim][y]).mean()])
                Z = np.vstack([Z, np.array(data[sim][z]).mean()])
            else:
                raise IOError("I do not understand 'inc'. Use debug mode to analyze.")
        else:
            raise IOError("I do not understand 'inc'. Use debug mode to analyze.")

    # flatten the arrays to become 1D arrays (instead of nested arrays)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()  

    # mask NaN and negative values in the RATIO array
    mask_nan=np.isnan(Z)
    Z_masked = np.ma.array(Z, mask=mask_nan) 

    # raise IOError("debug")
    # collect the field info in a dictionary
    datamap = {'X' : X,
                  'Y' : Y,
                  'Z' : Z_masked}           
                 
    print("Data map extracted.")
    sys.stdout.flush()

    return datamap


def moose_verification(filepath, x = 'vmJ2', y = 'temperature', z = 'creep_rate_diff'):
    """
    Read a MOOSE verification .csv file that contains the relative differences between MOOSE and
    a standalone surrogate model.

    Andre Ruybalid 2023 (c)
    """
    # read the .csv data and store in a dictionary
    verification_data = pd.read_csv(filepath, skiprows = 0)
    x_data = verification_data[x][1:]
    y_data = verification_data[y][1:]
    z_data = verification_data[z][1:]

    # mask the z array
    mask_nan=np.isnan(z_data)
    z_masked = np.ma.array(z_data, mask=mask_nan)

    return x_data, y_data, z_masked

def errorinc(valid_output, x = 'vmJ2', y = 'temperature', z = 'evm', min_relative_time=0.3, max_time=None, eps=1e-13):
    """
    Compare validation output to reference data at each increment.
    Assumes reference data and surrogate model time increments are equal. 
        !Does not work for adaptively time-stepped validation runs.
        !Does not interpolate between points.
    
    """
    print("Constructing error map...")
    sys.stdout.flush()
    
    # initialize empty numpy arrays to collect data in
    X = np.empty((0, 1))    
    Y = np.empty((0, 1))   
    MSRE = np.empty((0, 1))
    REL_RMSE = np.empty((0, 1))
    MARE = np.empty((0, 1))  
    RMSE = np.empty((0, 1)) 
    RISETIME = np.empty((0, 1))
    simnum = []    

    # set key for data extraction
    z_key_data = z
    z_key_valid = z

    # loop over the simulations in the dictionaries
    for sim in valid_output['valid']:
        # if sim == 541:
        #     raise IOError("debug")
        # gather the time arrays from data and valid
        if "t" in valid_output['data'][sim]:
            time_data = copy.deepcopy(np.array(valid_output['data'][sim]['t']))
            time_valid = copy.deepcopy(np.array(valid_output['valid'][sim]['t']))
        else:
            time_data = np.insert(np.cumsum(valid_output['data'][sim]['dt']), 0, 0.0)[:-1] # prepend cumsum with 0.0 as initial value and remove final value
            time_valid =np.insert(np.cumsum(valid_output['valid'][sim]['dt']), 0, 0.0)[:-1]

        # Check if time steps of surrogate and reference are equal
        this_msre = np.nan
        this_rel_rmse = np.nan
        this_mare = np.nan
        this_rmse = np.nan
        this_rise_time = np.nan
        # raise IOError("debug")
        if (len(time_data) >= len(time_valid)) and (time_valid[-1] >= min_relative_time*time_data[-1]):    
            # get the max inc based on max_time   
            if max_time == None:
                    max_time = time_data[-1]
            # max_inc = min([len(time_valid), np.argmin(np.abs(time_data-max_time))])
            max_inc = min(len(time_valid), np.searchsorted(time_data, max_time, side='right'))
            
            # Check if time elements between surrogate and reference data are equal within a margin of 1e-6 (s)
            if np.abs(time_data[:max_inc] - time_valid[:max_inc]).max() < 1e-6:
                # MSRE
                try:    
                    this_msre = compute_msre(valid_output['valid'][sim][z_key_valid][:max_inc], valid_output['data'][sim][z_key_data][:max_inc], eps=1e-13)
                except:
                    this_msre = np.nan
                    # REL_RMSE
                try:    
                    this_rel_rmse = compute_rel_rmse(valid_output['valid'][sim][z_key_valid][:max_inc], valid_output['data'][sim][z_key_data][:max_inc])
                except:
                    this_rel_rmse = np.nan
                    # MARE
                try:    
                    this_mare = compute_mare(valid_output['valid'][sim][z_key_valid][:max_inc], valid_output['data'][sim][z_key_data][:max_inc], eps=1e-13)
                except:
                    this_mare = np.nan
                    # RMSE
                try:    
                    this_rmse = compute_rmse(valid_output['valid'][sim][z_key_valid][:max_inc], valid_output['data'][sim][z_key_data][:max_inc])
                except:
                    this_rmse = np.nan
                try:
                    data_time = copy.deepcopy(np.array(valid_output['data'][sim]['t']))[:max_inc]
                    data_z = copy.deepcopy(np.array(valid_output['data'][sim][z_key_valid]))[:max_inc]
                    valid_time = copy.deepcopy(np.array(valid_output['valid'][sim]['t']))[:max_inc]
                    valid_z = copy.deepcopy(np.array(valid_output['valid'][sim][z_key_valid]))[:max_inc]
                    data_tc = rise_time(data_time, data_z)
                    valid_tc = rise_time(valid_time, valid_z)
                    this_rise_time = (valid_tc - data_tc) / data_tc 
                except:
                    this_rise_time = np.nan
                
        # stack to the the numpy arrays   
        simnum.append(sim) 
        if isinstance(valid_output['data'][sim][x], float):
            X = np.vstack([X, np.array(valid_output['data'][sim][x])])
            Y = np.vstack([Y, np.array(valid_output['data'][sim][y])])
        else:
            X = np.vstack([X, np.array(valid_output['data'][sim][x])[-1]])
            Y = np.vstack([Y, np.array(valid_output['data'][sim][y])[-1]])
        MSRE = np.vstack([MSRE, this_msre])
        REL_RMSE = np.vstack([REL_RMSE, this_rel_rmse])
        MARE = np.vstack([MARE, this_mare])
        RMSE = np.vstack([RMSE, this_rmse])
        RISETIME = np.vstack([RISETIME, this_rise_time])

        # if np.array(valid_output['data'][sim][x])[-1] > 50.0 and np.array(valid_output['data'][sim][y])[-1] > 900.0:
        #     raise IOError("debug")

    # flatten the arrays to become 1D arrays (instead of nested arrays)
    X = X.flatten()
    Y = Y.flatten()
    # strain = strain.flatten()
    MSRE = MSRE.flatten()
    REL_RMSE = REL_RMSE.flatten()
    MARE = MARE.flatten()
    RMSE = RMSE.flatten()
    RISETIME = RISETIME.flatten()

    # mask NaN in the MSRE array
    MSRE = np.ma.array(MSRE, mask=np.isnan(MSRE)) 
    REL_RMSE = np.ma.array(REL_RMSE, mask=np.isnan(REL_RMSE)) 
    MARE = np.ma.array(MARE, mask=np.isnan(MARE)) 
    RMSE = np.ma.array(RMSE, mask=np.isnan(RMSE))
    RISETIME = np.ma.array(RISETIME, mask=np.isnan(RISETIME)) 

    # collect the field info in a dictionary
    errorinc = {'X' : X,
                  'Y' : Y,
                  'MSRE' : MSRE,
                  'REL_RMSE' : REL_RMSE,
                  'MARE' : MARE,
                  'RMSE' : RMSE,
                  'RISETIME' : RISETIME,
                  'simnum' : simnum,
                  'z_key' : z}
    
    print("Data fields constructed.")
    sys.stdout.flush()

    return errorinc
    

def coord2index(x, y, coord, simnums=None):
    """
    coor2index finds the index related to 2D coordinates.

    Parameters
    ----------
    x : numpy array 1D
        x coordinates in data set.
    y : numpy array 1D
        y coordinates in data set.
    coord : list of 2D tuple
        each tuple contains x, y coordinates for which to find the index in the data
    simnums: list of int
        optional index for each simulation (default = None). 
        Needed when the simulation number is not equal to the list index in in x and y,
        e.g., when simulations are missing.

    Returns
    -------
    index : int
        simulation number.

    """
    index = []

    # Combine x and y values into a single numpy array
    xy = np.column_stack((x, y))

    print(coord)
    for i in range(len(coord)):
        # Calculate Euclidean distances between given coordinate and all data points
        distances = distance.cdist([coord[i]], xy, 'euclidean')
        
        # Find the index of the nearest point
        index_temp = np.argmin(distances) 
        print("\ncoord2index --> Coordinates: \n vmJ2[MPa] = {:5.2f}, T[K] = {:5.2f}. Index: {:d}\n".format(x[index_temp], y[index_temp], index_temp))
        sys.stdout.flush()

        # Correct index to simnumber according to optionally provided list of simnumbers
        if simnums == None:            
            index.append(index_temp)
        elif isinstance(simnums, list):
            index.append(simnums[index_temp])
        
    # Obtain only unique idices to prevent duplicates. 
    # np.unique also sorts, so return_index to sort back to original order of "index".
    unique_values, unique_indices = np.unique(index, return_index=True)

    # Re-sort back to original order.
    original_order_indices = np.argsort(unique_indices)
    sorted_index_to_original = unique_values[original_order_indices]

    return sorted_index_to_original

def getsize(N):
    """
    Print the size of an object by first saving it to disk and the computing its size in GB.
    When using sys.getsizeof(), additional factors may give unreasonable outputs. 
    Therefore, this method of first saving to disk is more accurate.
    """
    # Save the array to a file
    np.save('N.npy', N)
    
    # Get the size of the file
    file_size_bytes = os.path.getsize('N.npy')
    file_size_gigabytes = file_size_bytes / (1024 * 1024 * 1024)
    
    print("size of N: {:5.2f} GB.".format(file_size_gigabytes))
    
    # Remove the file
    os.remove('N.npy')

def get_matrix_size(matrix):
    total_size_gb = matrix.size * matrix.itemsize / 1024**3 
    return total_size_gb

def analyze_prism_quality(nodes, conn, mesh_specs, show_plot=True, quality_threshold=0.01):
    """
    Analyze and visualize quality of extruded simplex-based prisms.
    
    Args:
        nodes: (N, 3) array of node coordinates
        conn: (E, 6) array of prism connectivity (2 triangles stacked = 6 nodes per prism)
        mesh_specs: dict, used to identify extruded dimension
        show_plot: bool, if True shows plot of bad elements
        quality_threshold: float, aspect ratio threshold to flag bad elements

    Returns:
        bad_elem_indices: list of element indices with poor aspect ratio
    """

    def prism_height_and_base_area(prism_nodes):
        """
        Computes height and base area of a prism (assuming stacked triangle prisms).
        """
        base_tri = prism_nodes[:3, :]  # First triangle
        top_tri = prism_nodes[3:, :]   # Second triangle

        # Prism height = distance between centroids of top and bottom triangles
        base_centroid = base_tri.mean(axis=0)
        top_centroid = top_tri.mean(axis=0)
        height = np.linalg.norm(top_centroid - base_centroid)

        # Base area using Heron’s formula
        a, b, c = np.linalg.norm(base_tri[0] - base_tri[1]), \
                  np.linalg.norm(base_tri[1] - base_tri[2]), \
                  np.linalg.norm(base_tri[2] - base_tri[0])
        s = (a + b + c) / 2
        area = max(s * (s - a) * (s - b) * (s - c), 0.0)
        area = np.sqrt(area)
        return height, area

    aspect_ratios = []
    bad_elem_indices = []

    for i, elem in enumerate(conn):
        prism_nodes = nodes[elem]
        height, area = prism_height_and_base_area(prism_nodes)
        aspect_ratio = height / (np.sqrt(area) + 1e-12)
        aspect_ratios.append(aspect_ratio)
        if aspect_ratio < quality_threshold:
            bad_elem_indices.append(i)

    aspect_ratios = np.array(aspect_ratios)

    print(f"\n🔍 Analyzed {len(conn)} elements.")
    print(f"⚠️  Found {len(bad_elem_indices)} prism(s) with low quality (aspect ratio < {quality_threshold})")

    if show_plot and len(bad_elem_indices) > 0:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in bad_elem_indices:
            prism_nodes = nodes[conn[i]]
            faces = [
                [0, 1, 2],
                [3, 4, 5],
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 0, 3, 5]
            ]
            face_pts = [[prism_nodes[idx] for idx in face] for face in faces]
            poly = Poly3DCollection(face_pts, facecolor='red', alpha=0.4)
            ax.add_collection3d(poly)
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=2, c='k', alpha=0.3)
        ax.set_title("Bad Prism Elements (Low Aspect Ratio)")
        plt.tight_layout()
        plt.show()

    return bad_elem_indices

def get_sparse_matrix_size(sparse_matrix):
    data_size_gb = sparse_matrix.data.itemsize * len(sparse_matrix.data) / 1024**3
    indptr_size_gb = sparse_matrix.indptr.itemsize * len(sparse_matrix.indptr) / 1024**3
    indices_size_gb = sparse_matrix.indices.itemsize * len(sparse_matrix.indices) / 1024**3

    total_size_gb = data_size_gb + indptr_size_gb + indices_size_gb

    # print
    print("\nTotal size of the global sparse interpolation matrix: {:2.4f} GB. \nMatrix Shape: {}. \nNumber of nonzero elements: {}".format(total_size_gb, sparse_matrix.shape, sparse_matrix.nnz))
    sys.stdout.flush()

    #return
    return total_size_gb

def r2(y, yh):
    """
    Compute the R squared values between observations y and approximation yh.

    Parameters
    ----------
    y : numpy array (1D)
        observations.
    yh : numpy array (1D)
        approximated solution.

    Returns
    -------
    R_squared : scalar
        r squared value.

    """
    mean_y = np.mean(y)
    SSR = np.sum((y - yh) ** 2)
    SST = np.sum((y - mean_y) ** 2)
    R_squared = 1 - (SSR / SST)
    return R_squared

# Error metrics
def compute_rmse(pred : np.array, target : np.array):
    '''
    root mean square error
    '''
    return np.sqrt(np.mean((pred-target)**2))
    
def compute_rel_rmse(pred : np.array, target : np.array):
    '''
    relative root mean square error
    '''
    return np.sqrt(np.mean( (pred-target)**2 )) / np.sqrt(np.mean( target**2 ))

def compute_msre(pred: np.array, target: np.array, eps=1e-20):
    '''
    mean square relative error
    '''
    return np.mean( (np.abs(pred-target)/((np.abs(target) + eps)))**2 )

def compute_mare(pred: np.array, target: np.array, eps=1e-20):
    '''
    mean absolute relative error
    '''
    return np.mean( np.abs(pred-target)/((np.abs(target) + eps)) )

def time_to_target(times: np.array, vals: np.array, target_val: float):
    time_to_target = np.nan
    for i in range(1,len(vals)):
        if vals[i] >= target_val: #first passage
            s1 = vals[i-1]
            s2 = vals[i]
            t1 = times[i-1]
            t2 = times[i]
            time_to_target = t1 + (target_val - s1)/(s2-s1) * (t2-t1)
            break
            
    return time_to_target

def rise_time(times: np.array, vals: np.array):
    '''
    https://en.wikipedia.org/wiki/Rise_time
    time from 10% to 90% of the final values (based on first passage)
    '''
    time_to_ten_perc_final = time_to_target(times, vals, 0.10*vals[-1])
    time_to_ninety_perc_final = time_to_target(times, vals, 0.90*vals[-1])
    return time_to_ninety_perc_final - time_to_ten_perc_final
