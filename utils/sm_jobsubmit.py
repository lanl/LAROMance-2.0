#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================
@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Library for analyzing surrogate model accuracy, validation results.

Latin Hypercube Sampling is supported. Others can be added here.

Andre Ruybalid 2026 
"""
import numpy as np
import os, sys, copy
import pandas as pd
from pyDOE2 import lhs
import matplotlib.pyplot as plt

def generate_LHS_set_modular(parameters, num_samples=1000, output_file=None, visualize=True, 
                           visualization_dims=3, format_specs=None, path_to_template="../sm_workdir/LApx_templates"):
    """
    Generate Latin Hypercube Sample set with modular parameter configuration.
    
    Parameters
    ----------
    parameters : dict
        Dictionary of parameter configurations. Keys are parameter names (used for CSV headers),
        values are dictionaries with 'min' and 'max' keys defining the parameter ranges.
        Example: {'stress': {'min': 0.0, 'max': 250.0}, 'temperature': {'min': 600.0, 'max': 900.0}}
    num_samples : int, optional
        Number of LHS samples to generate. Default is 1000.
    output_file : str, optional
        Name of the output file. If None, a default name will be constructed.
    visualize : bool, optional
        Whether to create visualization plots. Default is True.
    visualization_dims : int, optional
        Number of dimensions to visualize (2 or 3). Default is 3.
    format_specs : dict, optional
        Dictionary mapping parameter names to format strings.
        Example: {'stress': '{:.2f}', 'temperature': '{:.1f}', 'rhoc': '{:2.2e}'}
        If None, defaults to "{:.2f}" for all parameters.
    path_to_template : str, optional
        Base directory for output file. Default is "../sm_workdir/LApx_templates".
    
    Returns
    -------
    str
        Path to the generated CSV file.
    """
    # Validate parameters
    if not parameters:
        raise ValueError("Parameters dictionary cannot be empty")
    
    # Extract parameter names and dimensions
    param_names = list(parameters.keys())
    num_dims = len(param_names)
    
    if num_dims == 0:
        raise ValueError("No parameters provided")
    
    # Generate LHS samples
    lhs_samples = lhs(num_dims, samples=num_samples)
    
    # Scale samples to the desired ranges and store in a dictionary
    scaled_values = {}
    for i, param_name in enumerate(param_names):
        param_config = parameters[param_name]
        min_val = param_config['min']
        max_val = param_config['max']
        scaled_values[param_name] = min_val + (max_val - min_val) * lhs_samples[:, i]
    
    # Visualization (if enabled)
    if visualize:
        if visualization_dims not in [2, 3]:
            raise ValueError("visualization_dims must be either 2 or 3")
        
        if num_dims >= visualization_dims:
            fig = plt.figure()
            
            if visualization_dims == 3 and num_dims >= 3:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                ax.scatter(
                    scaled_values[param_names[0]], 
                    scaled_values[param_names[1]], 
                    scaled_values[param_names[2]], 
                    c='b'
                )
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
                ax.set_zlabel(param_names[2])
            elif visualization_dims == 2 and num_dims >= 2:
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(
                    scaled_values[param_names[0]], 
                    scaled_values[param_names[1]], 
                    c='b'
                )
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
            
            plt.title(f"LHS Samples ({num_samples} points)")
            plt.show()
            fig.tight_layout()
    
    # Format values according to specifications
    formatted_values = {}
    for param_name in param_names:
        # Get format spec for this parameter or use default
        if format_specs and param_name in format_specs:
            fmt_spec = format_specs[param_name]
        else:
            fmt_spec = "{:.2f}"  # Default format
        
        # Format the values
        formatted_values[param_name] = [fmt_spec.format(value) for value in scaled_values[param_name]]
    
    # Prepare for saving
    if output_file is None:
        # Construct default filename based on dimensions and sample count
        output_file = f"{num_dims}D_lhs_{num_samples}.csv"
    
    # Combine all formatted values for saving
    columns = [formatted_values[param_name] for param_name in param_names]
    data_to_save = np.column_stack(tuple(columns))
    
    # Save to CSV file
    output_path = os.path.join(path_to_template, output_file)
    np.savetxt(
        output_path,
        data_to_save,
        delimiter=',',
        header=','.join(param_names),
        comments='',
        fmt='%s'
    )
    
    return output_path
