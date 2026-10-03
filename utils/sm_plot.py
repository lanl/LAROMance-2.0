#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================
@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Plotting funcions

@author: Andre Ruybalid 

"""
import numpy as np, pickle, os, sys, copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LogLocator
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib import scale as mscale
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.scale import FuncScale
import matplotlib.tri as tri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from itertools import cycle
import itertools

from utils import sm_analyze as sma
from utils import sm_data as smd
from scipy.spatial import ConvexHull

def plotmechanisms(path=None, smooth=True, grid_size=400, interp_method='linear'):
    """
    Plot mechanism contributions from mechanisms.csv.

    Parameters:
    - path: str, path to directory containing 'mechanisms.csv'
    - smooth: bool, whether to use smooth grid interpolation
    - grid_size: int, number of points along each axis if smooth=True
    - interp_method: str, 'linear' or 'cubic' for griddata interpolation
    """
    if path is None:
        raise IOError("Provide a path to the directory containing 'mechanisms.csv'.")
    
    d, f = smd.readcsv(path)
    file = 'mechanisms_end.csv'
    stress = d[file]['stress'].values
    temperature = d[file]['temperature'].values
    glide = d[file]['glide'].values
    climb = d[file]['climb'].values
    diffusion = d[file]['diffusion'].values

    # Compute fractions
    total = glide + climb + diffusion
    total[total == 0] = 1.0
    glide_frac = glide / total
    climb_frac = climb / total
    diffusion_frac = diffusion / total

    # RYB primaries
    primaries = np.array([
        [1, 0, 0],   # red
        [1, 1, 0],   # yellow
        [0, 0, 1],   # blue
    ])

    # Setup figure with gridspec
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[3,1], height_ratios=[1,1,1],
                          hspace=0.4, wspace=0.3)

    # Large blend axis
    ax_blend = fig.add_subplot(gs[:,0])
    ax_blend.set_facecolor("black")
    ax_blend.set_xlabel("Von Mises Stress [MPa]", fontsize=14, color='white')
    ax_blend.set_ylabel("Temperature [K]", fontsize=14, color='white')
    ax_blend.set_title("Mechanism contributions (RYB blend)", fontsize=16, color='white', pad=15)
    ax_blend.tick_params(axis='both', labelsize=12)

    # Legend
    legend_elements = [
        Patch(facecolor=[1,0,0], edgecolor='k', label='Glide'),
        Patch(facecolor=[1,1,0], edgecolor='k', label='Climb'),
        Patch(facecolor=[0,0,1], edgecolor='k', label='Diffusion'),
    ]
    ax_blend.legend(handles=legend_elements, title="Mechanisms", loc='lower left',
                    facecolor='black', edgecolor='white', labelcolor='white', title_fontsize=12)

    if smooth:
        # Grid interpolation
        xi = np.linspace(stress.min(), stress.max(), grid_size)
        yi = np.linspace(temperature.min(), temperature.max(), grid_size)
        XI, YI = np.meshgrid(xi, yi)
        R_int = griddata((stress, temperature), glide_frac, (XI, YI), method=interp_method)
        C_int = griddata((stress, temperature), climb_frac, (XI, YI), method=interp_method)
        D_int = griddata((stress, temperature), diffusion_frac, (XI, YI), method=interp_method)

        # Blend RGB
        RGB = np.zeros((XI.shape[0], XI.shape[1], 3))
        for i, channel in enumerate([R_int, C_int, D_int]):
            RGB[:,:,0] += channel * primaries[i,0]
            RGB[:,:,1] += channel * primaries[i,1]
            RGB[:,:,2] += channel * primaries[i,2]
        RGB = np.clip(RGB, 0, 1)
        ax_blend.imshow(RGB, origin='lower',
                        extent=(stress.min(), stress.max(), temperature.min(), temperature.max()),
                        aspect='auto')

        mechanisms = {
            "Glide": (R_int, [1,0,0]),
            "Climb": (C_int, [1,1,0]),
            "Diffusion": (D_int, [0,0,1])
        }

        for i, (name, (grid_frac, base_color)) in enumerate(mechanisms.items()):
            ax = fig.add_subplot(gs[i,1])
            ax.set_facecolor("black")
            mech_RGB = np.zeros((grid_frac.shape[0], grid_frac.shape[1], 3))
            for j in range(3):
                mech_RGB[:,:,j] = grid_frac * base_color[j]
            ax.imshow(mech_RGB, origin='lower',
                      extent=(stress.min(), stress.max(), temperature.min(), temperature.max()),
                      aspect='auto')
            ax.set_title(name, fontsize=12, color='white', pad=8)
            ax.set_xlabel("Stress [MPa]", fontsize=12, color='white')
            ax.set_ylabel("Temp [K]", fontsize=12, color='white')
            ax.tick_params(colors='white', labelsize=10)

    else:
        # Raw scatter points
        fractions = np.vstack([glide_frac, climb_frac, diffusion_frac]).T
        colors = fractions @ primaries
        colors = np.clip(colors, 0, 1)
        ax_blend.scatter(stress, temperature, c=colors, s=100, alpha=0.9, edgecolor=None)

        mechanisms = {
            "Glide": (glide_frac, [1,0,0]),
            "Climb": (climb_frac, [1,1,0]),
            "Diffusion": (diffusion_frac, [0,0,1])
        }

        for i, (name, (frac, base_color)) in enumerate(mechanisms.items()):
            mech_colors = np.outer(frac, base_color)
            ax = fig.add_subplot(gs[i,1])
            ax.set_facecolor("black")
            ax.scatter(stress, temperature, c=mech_colors, s=50, alpha=0.9, edgecolor=None)
            ax.set_title(name, fontsize=12, color='white', pad=8)
            ax.set_xlabel("Stress [MPa]", fontsize=12, color='white')
            ax.set_ylabel("Temp [K]", fontsize=12, color='white')
            ax.tick_params(colors='white', labelsize=10)

    plt.tight_layout()
    plt.show(block=False)

def extract_edges_from_hull(hull, dim):
    if dim == 2:
        return hull.simplices
    elif dim == 3:
        edges = set()
        for simplex in hull.simplices:
            i, j, k = simplex
            edges.update([
                tuple(sorted((i, j))),
                tuple(sorted((j, k))),
                tuple(sorted((k, i)))
            ])
        return np.array(list(edges))
    else:
        raise NotImplementedError(f"Hull edge extraction not implemented for dim={dim}")


def visualize_mesh(node, conn=None, dim=2, scalers=None, highlight_elements=[], ax=None,
                   show_node_numbers=True, show_elem_numbers=True,
                   edge_color="orange", return_artists=False):
    """
    Visualize ND mesh with nodes and connectivity.

    Parameters:
        node : (Nnode, dim) array of nodal coordinates
        conn : (Nelement, Nvertices) array of connectivity (optional)
        dim : int, dimensionality (auto-inferred from node if None)
        show_node_numbers : bool, whether to show node indices
        show_elem_numbers : bool, whether to show element indices at element centroid
        edge_color : color of edges (default 'orange')
        return_artists : bool, whether to return the matplotlib artist objects created

    Returns:
        List of matplotlib artist objects if return_artists=True, else None
    """

    artists = []

    # backtransform to physical units
    if scalers != None and isinstance(scalers, dict):
        node = smd.backtransform_nodes(node, scalers)

    if dim is None:
        dim = node.shape[1]

    if dim == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 1))

        line_artist, = ax.plot(node[:, 0], np.zeros_like(node[:, 0]), 'o', color='orange')
        artists.append(line_artist)

        if show_node_numbers:
            for i, (x,) in enumerate(node):
                text_artist = ax.text(x, 0.02, str(i), ha='center', fontsize=14, color='blue')
                artists.append(text_artist)

        if conn is not None:
            for i, c in enumerate(conn):
                x = node[c, 0]
                line_artist, = ax.plot(x, [0, 0], color=edge_color, lw=0.7)
                artists.append(line_artist)
                if show_elem_numbers:
                    centroid = np.mean(x)
                    text_artist = ax.text(centroid, -0.02, f"E{i}", ha='center', color='red', fontsize=12)
                    artists.append(text_artist)

        ax.set_xlabel('x')
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title('1D Mesh')
        ax.axis('equal')
        if not return_artists:
            plt.show(block=False)
        return artists if return_artists else None

    if dim == 2:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        node2d = node[:, :2] if node.shape[1] > 2 else node

        scatter_artist = ax.plot(node2d[:, 0], node2d[:, 1], 'd',  color='orange')
        artists.extend(scatter_artist)

        if show_node_numbers:
            for i, (x, y) in enumerate(node2d):
                text_artist = ax.text(x, y, str(i), fontsize=14, ha='center', va='center', color='k')
                artists.append(text_artist)

        if conn is not None:
            highlight_elements_set = set(highlight_elements)
            for i, elem in enumerate(conn):
                coords = node2d[elem]
                try:
                    hull = ConvexHull(coords)
                    edges = extract_edges_from_hull(hull, dim=2)
                except Exception:
                    edges = list(itertools.combinations(range(len(coords)), 2))

                # Draw edges
                for edge in edges:
                    p1, p2 = coords[edge]
                    line_artist, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_color, lw=0.7)
                    artists.append(line_artist)

                # Draw filled face if highlighted
                if i in highlight_elements_set:
                    polygon = mpatches.Polygon(coords, closed=True, facecolor='red', alpha=0.3, edgecolor=None)
                    patch_artist = ax.add_patch(polygon)
                    artists.append(patch_artist)

                if show_elem_numbers:
                    centroid = np.mean(coords, axis=0)
                    text_artist = ax.text(centroid[0], centroid[1], f"E{i}", color='red', fontsize=10, ha='center')
                    artists.append(text_artist)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D Mesh')
        if not return_artists:
            plt.show(block=False)
        return artists if return_artists else None

    elif dim == 3:
        # -------------------------------------------------------------- #
        # 0️⃣  Make sure we have a 3‑D axis                               #
        # -------------------------------------------------------------- #
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # -------------------------------------------------------------- #
        # 1️⃣  Work with only the first three coordinates                  #
        # -------------------------------------------------------------- #
        node3d = node[:, :3]                     # <-- slice to (x, y, z)

        # -------------------------------------------------------------- #
        # 2️⃣  Plot the nodes (scatter)                                   #
        # -------------------------------------------------------------- #
        scatter_artist = ax.scatter(node3d[:, 0],
                                    node3d[:, 1],
                                    node3d[:, 2],
                                    c='k',
                                    marker='o',
                                    s=40,
                                    depthshade=True)
        artists.append(scatter_artist)

        # -------------------------------------------------------------- #
        # 3️⃣  Optional node number annotations                           #
        # -------------------------------------------------------------- #
        if show_node_numbers:
            for i, (x, y, z) in enumerate(node3d):
                txt = ax.text(x, y, z,
                            str(i),
                            color='k',
                            fontsize=12,
                            ha='center',
                            va='center')
                artists.append(txt)

        # -------------------------------------------------------------- #
        # 4️⃣  Plot the elements (edges + optional face fill)            #
        # -------------------------------------------------------------- #
        if conn is not None:
            highlight_set = set(highlight_elements)

            for i_elem, elem in enumerate(conn):
                # vertices of the current element (still use the full node array
                # because the indices are the same; then slice to 3‑D)
                coords = node3d[elem]

                # ---- Edge extraction -------------------------------- #
                try:
                    hull = ConvexHull(coords)
                    # collect all unique unordered pairs from the tetrahedra
                    edge_set = set()
                    for tet in hull.simplices:               # tetrahedra (indices)
                        for a, b in itertools.combinations(tet, 2):
                            edge_set.add(tuple(sorted((a, b))))
                    edges = list(edge_set)
                except Exception:               # fallback: fully connected
                    edges = list(itertools.combinations(range(len(coords)), 2))

                # ---- Draw edges -------------------------------------- #
                for v1, v2 in edges:
                    p1 = coords[v1]
                    p2 = coords[v2]
                    line = ax.plot([p1[0], p2[0]],
                                [p1[1], p2[1]],
                                [p1[2], p2[2]],
                                color=edge_color,
                                lw=0.7)
                    artists.extend(line)

                # ---- Highlighted face (semi‑transparent) ------------ #
                if i_elem in highlight_set:
                    try:
                        # exterior facets of the hull (triangles)
                        exterior = hull.simplices[hull.neighbors == -1]
                        faces = [coords[f] for f in exterior]
                    except Exception:
                        faces = [coords]          # fallback: whole element

                    poly = Poly3DCollection(faces,
                                            facecolor='red',
                                            edgecolor='none',
                                            alpha=0.3)
                    ax.add_collection3d(poly)
                    artists.append(poly)

                # ---- Element number annotation ----------------------- #
                if show_elem_numbers:
                    centroid = np.mean(coords, axis=0)
                    txt = ax.text(centroid[0],
                                centroid[1],
                                centroid[2],
                                f"E{i_elem}",
                                color='red',
                                fontsize=10,
                                ha='center',
                                va='center')
                    artists.append(txt)

        # -------------------------------------------------------------- #
        # 5️⃣  Axes cosmetics                                            #
        # -------------------------------------------------------------- #
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Mesh')


        # -------------------------------------------------------------- #
        # 6️⃣  Show or return                                            #
        # -------------------------------------------------------------- #
        if not return_artists:
            plt.show(block=False)

        return artists if return_artists else None

def plottraces(data: dict, input_keys: list, output_keys: list, xlog=False, ylog=True, max_number_plots=50, storepath=None, filename="", dark=True, color='gray'):
    """
    Generate and store data output traces of individual simulations.
    Inputs:
    data : dict
        Data dictionary structured in individual runs (not concatenated)
    input_keys : list
        List of input field variable keys to plot
    output_keys : list
        List of output field variable keys to plot
    xlog : bool
        Whether to use logarithmic scale for x-axis
    ylog : bool
        Whether to use logarithmic scale for y-axis
    max_number_plots : int
        Number of individual traces to plot
    storepath : str
        Path to store plot
    dark : bool
        Dark background (True) in plots or white background (False)
    color : str
        Color of the trace lines
    """
    
    # Prepare trace figure
    fig_traces, ax_traces = plt.subplots(len(input_keys), len(output_keys), figsize=(11, 8))
    print("len input: {},\nlen_output: {}".format(len(input_keys), len(output_keys)))
    plt_cnt = 0
    for sim in data:
        # Extract simulation traces
        for i, input_key in enumerate(input_keys):
            for j, output_key in enumerate(output_keys):
                if output_key.startswith("U["):
                    # raise IOError("debug")
                    key = output_key[2:-1]
                    plotxy(data[sim][input_key], data[sim]['U'][key], color=color, marker='-', ax = ax_traces[i,j], xlog=xlog, ylog=ylog)
                else:
                    plotxy(data[sim][input_key], data[sim][output_key], color=color, marker='-', ax = ax_traces[i,j], xlog=xlog, ylog=ylog)
                ax_traces[i, j].set_xlabel(input_key)
                ax_traces[i, j].set_ylabel(output_key)

        plt_cnt += 1
        if plt_cnt >= max_number_plots:
            break
    fig_traces.tight_layout()

    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    if storepath is not None:
        os.makedirs(storepath, exist_ok=True)
        fig_traces.savefig(os.path.join(storepath, "Data_simulation_traces" + filename + ".png"))
    elif storepath is None:
        plt.show(block=False)

def plotdata(data: dict, input_keys: list, output_keys: list, simulation_recording_interval: int = 1, inc=-1, cmap= "magma", storepath=None, filename= "", dark=True):
    """
    Generate and store data output scatter plots.
    Inputs:
    data : dict
        Data dictionary structured in individual runs (not concatenated)
    input_keys : list
        List of input field variable keys to plot
    output_keys : list
        List of output field variable keys to plot
    simulation_recording_interval : int
        The interval with which to skip simulations in the dataset to add to the scatter plot
    inc : int
        Increment to plot. inc = -1 plots the final increment.
    storepath : str
        Path to store plots
    dark : bool
        Dark background (True) in plots or white background (False)
    """

    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    def parse_output_access(simdata, key, inc):
        """
            Mirrors plottraces logic:
            - If key is like U[evm] or U['evm'], look in simdata['U'][...]
            - else, look in simdata[key]
        """
        if key.startswith("U["):
            # Extract inner variable name:
            # Handles U[evm], U['evm'], U["evm"]
            inner = key[key.find("[")+1 : key.rfind("]")]
            inner = inner.strip("'\"")   # remove quotes if present
            return simdata["U"][inner][inc]
        else:
            return simdata[key][inc]
        
    # Prepare field variable containers
    fields_inputs = {key: [] for key in input_keys}
    fields_outputs = {key: [] for key in output_keys}

    keys = list(data.keys())  # Convert dict keys to a list
    # Loop over simulations (depending on the provided recording interval)
    for sim in keys[::simulation_recording_interval]:
        # Extract field variables from the final increment of each simulation trace
        for key in fields_inputs.keys():
            try:
                fields_inputs[key].append(data[sim][key][inc])
            except:
                fields_inputs[key].append(data[sim][key])
        # for key in fields_outputs.keys():
        #     fields_outputs[key].append(data[sim]['U'][key][inc])

        for key in fields_outputs.keys():
            try:
                fields_outputs[key].append(parse_output_access(data[sim], key, inc))
            except Exception:
                raise KeyError(f"Output variable '{key}' not found in data structure for simulation {sim}.")


    # Initialize plot axes
    num_output_fields = len(output_keys)
    fig_input_fields, ax_input_fields = plt.subplots(1, len(input_keys), figsize=(6 * len(input_keys), 4))
    fig_output_fields, ax_output_fields = plt.subplots(1, num_output_fields, figsize=(6 * num_output_fields, 4))

    # Plot input fields
    for i, key in enumerate(input_keys):
        scatter = ax_input_fields[i].scatter(fields_inputs[input_keys[0]], fields_inputs[input_keys[1]], c=fields_inputs[key],
                                            s=50, marker=',',
                                            cmap='viridis',
                                            edgecolors='k',
                                            alpha=0.9,
                                            norm=matplotlib.colors.LogNorm(vmin=min(fields_inputs[key]), vmax=max(fields_inputs[key])))
        ax_input_fields[i].set_title(f'Scatter plot for final {key}', fontsize=18)
        ax_input_fields[i].tick_params(labelsize=18)
        ax_input_fields[i].set_xlabel(input_keys[0], fontsize=18)
        ax_input_fields[i].set_ylabel(input_keys[1], fontsize=18)
        cbar1 = fig_input_fields.colorbar(scatter, ax=ax_input_fields[i])
        cbar1.set_label(key, fontsize=18)
        cbar1.ax.tick_params(labelsize=16)

    # Plot output fields
    for i, key in enumerate(output_keys):
        # if key in ['rhoc', 'rhow']:
        # raise IOError("debug")
        print("MAX: {} in {}".format(max(fields_outputs[key]), key))
        if min(fields_outputs[key]) > 0.0:
            scatter = ax_output_fields[i].scatter(fields_inputs[input_keys[0]], fields_inputs[input_keys[1]], 
                                                    c=fields_outputs[key],
                                                    s=50, marker=',',
                                                    cmap=cmap,
                                                    edgecolors='k',
                                                    alpha=0.9,
                                                    norm=matplotlib.colors.LogNorm(vmin=min(fields_outputs[key]), vmax=max(fields_outputs[key]))
                                                    )

        else:
            print("Trying Symlog since Lognorm didn't work for plotting. Likely due to negative values in the data.")
            if max(fields_outputs[key]) == 0:
                raise IOError("MAX: {} in {}".format(max(fields_outputs[key]), key))
            scatter = ax_output_fields[i].scatter(fields_inputs[input_keys[0]], fields_inputs[input_keys[1]], c=fields_outputs[key],
                                                 s=50, marker=',',
                                                 cmap=cmap,
                                                 edgecolors='k',
                                                 alpha=0.9,
                                                 norm=matplotlib.colors.SymLogNorm(vmin=min(fields_outputs[key]), vmax=max(fields_outputs[key]), linthresh=0.01)
                                                 )

        ax_output_fields[i].set_title(f'Output scatter plot for final {key}', fontsize=18)
        ax_output_fields[i].tick_params(labelsize=18)
        ax_output_fields[i].set_xlabel(input_keys[0], fontsize=18)
        ax_output_fields[i].set_ylabel(input_keys[1], fontsize=18)
        cbar2 = fig_output_fields.colorbar(scatter, ax=ax_output_fields[i])
        cbar2.set_label(key, fontsize=18)
        cbar2.ax.tick_params(labelsize=14)

    fig_output_fields.tight_layout()
    fig_input_fields.tight_layout()
    if storepath is not None:
        os.makedirs(storepath, exist_ok=True)
        fig_input_fields.savefig(os.path.join(storepath, "Data_input_fields" + filename + ".png"))
        fig_output_fields.savefig(os.path.join(storepath, "Data_output_fields" + filename + ".png"))
    elif storepath is None:
        plt.show(block=False)

    return ax_input_fields, ax_output_fields

def fieldpicker(valid_output, errorinc, quality_metric="REL_RMSE", minmax=None, numsims : np.array = None, numclicks=9, flagnan=True, dark=True):
    """
    Pick some points in a 2D field. 
    Then plot the individual runs from the nearest neighbor of the picked location.
    
    minmax : tuple of float
        (min, max)
    """
    simnumbers = []

    # Extract z_key from errorinc (key plot in the field)
    z_key = errorinc['z_key'] 
    
    try:
        if minmax is None:
            fig1, ax1 = scatterfield(errorinc['X'], errorinc['Y'], errorinc[quality_metric], 
                                    zlabel= quality_metric + '[-]', 
                                    vmin=errorinc[quality_metric].min(), vmax=errorinc[quality_metric].max(), 
                                    cmap='RdBu_r', edgecolors=None,
                                    logcolors=True, flagnan=flagnan,
                                    histplot=False,
                                    dark=dark)
        else:
            fig1, ax1 = scatterfield(errorinc['X'], errorinc['Y'], errorinc[quality_metric], 
                                    zlabel= quality_metric + '[-]', 
                                    vmin=minmax[0], vmax=minmax[1], 
                                    cmap='RdBu_r', edgecolors=None,
                                    logcolors=True, flagnan=flagnan,
                                    histplot=False,
                                    dark=dark)
    except:
        fig1, ax1 = scatterfield(errorinc['X'], errorinc['Y'], errorinc[quality_metric], 
                                zlabel= quality_metric + '[-]', 
                                cmap='RdBu_r', edgecolors=None,
                                logcolors=True,  flagnan=flagnan,
                                histplot=False,
                                dark=dark)
    
    if not isinstance(numsims, np.ndarray):
        # click on figure
        numclicks
        plt.waitforbuttonpress()
        if True:
            click = plt.ginput(numclicks)
            print(click)
        numsims = sma.coord2index(errorinc['X'], errorinc['Y'], click, simnums=errorinc['simnum'])
    
    # raise IOError("debug")
    # If needed, adjust key for plotting
    z_key_data_plotxy = z_key
    z_key_valid_plotxy = z_key
    
    # plot the individual plots based on the clicks
    fig8, axs = plt.subplots(3, 3, figsize=(12, 8))
    for i in range(len(numsims)):
        row = i // 3
        col = i % 3
        # raise IOError("debug")
        if numsims[i] in errorinc['simnum']:
            sim = numsims[i]
            simnumbers.append(sim)
            plotxy(valid_output['data'][sim]['t'], valid_output['data'][sim][z_key_data_plotxy], marker='.', color='green', ax=axs[row,col], dark=dark)
            plotxy(valid_output['valid'][sim]['t'], valid_output['valid'][sim][z_key_valid_plotxy], marker='.-', color='orange', ax=axs[row,col], dark=dark)
            subtitle_text = "sim#: {} \n$\sigma$: {:4.0f} \n$T$: {:4.0f}".format(sim, valid_output['data'][sim]['vmJ2'][-1], valid_output['data'][sim]['temperature'][-1])
            axs[row,col].text(0.7, 0.2, subtitle_text, ha='center', va='center', transform=axs[row,col].transAxes, fontsize=10)
            axs[row,col].set_xlabel("[s]", fontsize=12)
            axs[row,col].set_ylabel(r"$\varepsilon_\mathrm{vm}$ [-]", fontsize=12)
            axs[row,col].tick_params(labelsize = 12)
        else:
            raise IOError("Not sure which sim to plot. Abort.")

    fig8.tight_layout()

    plt.show(block=False)

    return np.array(simnumbers)

def datahist(data : dict, num_inputs : int, storepath : str = None, filename : str = "", input_label_keys = None, output_label_keys = None, dark = True):
    """
    Plot data histogram projections in one figure and a matrix of axes.
    1D histograms are plot on the diagonal axes of the figure.
    2D histogram maps are plot on the non-diagonal axes of the figure. 
    inputs:
        - data : dict
            Dictionary containing the data (concatenated format)
        - num_inputs : int
            Number of inputs to create histograms from 
        - storepath, filename : str
            File path/name to store figure in (tip: use the name of the data-files)
        - input_label_keys, output_label_keys : [str] (default: None)
            list of strings for each input/output key to use as axis label in histogram plots. 
            Order in the list must correspond to the key ordering in data
        - dark : bool (default = True)
            Dark background theme for the figure: True or False
    
    outputs:
        No return variables: figure will be stored in "storepath", under "filename"

    """
    print("\nPlot histograms. Store in: '{}'.".format(storepath))
    sys.stdout.flush()
    darktheme(dark)

    fs = 9 # fontsize in histogram axes

    # plot histograms of output responses
    # output_label_keys = [r"$\dot{\varepsilon}_p$ [-]" , r"$\dot{\rho}_\mathrm{cell}$ [m$^{-2}$]", r"$\dot{\rho}_\mathrm{wall}$ [m$^{-2}$]"]
    figout, ax = plt.subplots(1, len(data['U'].keys()),figsize=(10,4))
    for i, key in enumerate(data['U'].keys()):
        y = data['U'][key]
        ax[i].hist(y, bins=20)  
        if output_label_keys is not None:
            ax[i].set_xlabel(output_label_keys[i], fontsize=fs)
        elif output_label_keys is None:
            ax[i].set_xlabel(key, fontsize=fs)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
    figout.tight_layout()

    # plot histograms of input responses
    fig, axs = plt.subplots(num_inputs, num_inputs, figsize=(10, 8))  
    keys = list(data.keys())
    # input_label_keys = [r"$\sigma_\mathrm{vm}$ [MPa]", "$T$ [K]", r"$\varepsilon_\mathrm{p}$ [s$^{-1}$]", r"$\rho_\mathrm{cell}$ [m$^{-2}$]", r"$\rho_\mathrm{wall}$ [m$^{-2}$]", "MX [-]", r"$\dot{\phi}$ [dpa]"]
    # Plotting histograms and 2D histograms in each subplot
    for i in range(num_inputs):
        for j in range(num_inputs):
            if i == j:
                x = data[keys[i]]
                # histograms on diagonals
                magma_mid = plt.cm.magma(0.6)          # returns an RGBA tuple
                hex_colour = matplotlib.colors.to_hex(magma_mid)   # → '#ff7e41' (example)
                axs[i, j].hist(x, bins=20, color=hex_colour) 
                if input_label_keys is not None:
                    axs[i, j].set_xlabel(input_label_keys[i], fontsize=fs) 
                    axs[i, j].tick_params(labelsize=fs)
                    axs[i, j].xaxis.get_offset_text().set_fontsize(fs)
                    axs[i, j].yaxis.get_offset_text().set_fontsize(fs)
                elif input_label_keys is None:
                    axs[i, j].set_xlabel(keys[i], fontsize=fs) 
                    axs[i, j].tick_params(labelsize=fs)
                    axs[i, j].xaxis.get_offset_text().set_fontsize(fs)
                    axs[i, j].yaxis.get_offset_text().set_fontsize(fs)
            elif i > j:
                y = data[keys[j]]
                x = data[keys[i]]
                # Plotting 2D histograms
                axs[i, j].hist2d(x, y, bins=20, cmap='magma') 
                if input_label_keys is not None:
                    axs[i, j].set_xlabel(input_label_keys[i], fontsize=fs)
                    axs[i, j].set_ylabel(input_label_keys[j], fontsize=fs)
                    axs[i, j].tick_params(labelsize=fs)
                    axs[i, j].xaxis.get_offset_text().set_fontsize(fs)
                    axs[i, j].yaxis.get_offset_text().set_fontsize(fs)
                elif input_label_keys is None:
                    axs[i, j].set_xlabel(keys[i], fontsize=fs)
                    axs[i, j].set_ylabel(keys[j], fontsize=fs)
                    axs[i, j].tick_params(labelsize=fs)
                    axs[i, j].xaxis.get_offset_text().set_fontsize(fs)
                    axs[i, j].yaxis.get_offset_text().set_fontsize(fs)
            else:
                # Turn off axes for upper non-diagonal subplots
                axs[i, j].axis('off')  
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
    fig.tight_layout()
    # Store plots
    if storepath is not None:
        os.makedirs(storepath, exist_ok=True)
        fig.savefig(os.path.join(storepath, "Data_input_histograms_"+filename+".png"))
        figout.savefig(os.path.join(storepath, "Data_output_histograms_"+filename+".png"))
    else:
        plt.show(block=False)

def plotxy(x,y,label=None, marker='.-', xlabel='x', ylabel='y', ylog=False, xlog=False, dark=True, color="white",
    linewidth=2.5, show=True, ax=None, linthresh=1e-3, eps=1e-12,  max_ticks=3,
):
    """
    Plot one variable against another with robust log‑scale handling.
    The function now guarantees at most ``max_ticks`` major tick marks
    (default 3) on any log‑scaled axis.

    Parameters
    ----------
    x, y : array‑like
        Data to plot.
    label : str, optional
        Legend label.
    marker : str, optional
        Matplotlib line/marker spec.
    xlabel, ylabel : str, optional
        Axis labels.
    ylog, xlog : bool, optional
        If True, use a log‑scale (or symlog for mixed signs) on the
        corresponding axis.
    dark : bool, optional
        Use dark background theme when True.
    color : str, optional
        Colour of the plotted line.
    linewidth : float, optional
        Width of the plotted line.
    show : bool, optional
        Call ``plt.show()`` before returning if True.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None a new figure/axes are created.
    linthresh : float, optional
        Threshold for symlog scaling.
    eps : float, optional
        Small positive value used to replace non‑positive data when a
        pure log scale is requested.
    max_ticks : int, optional
        Desired maximum number of major ticks on a log‑scaled axis
        (default 2).

    Returns
    -------
    ax, line : tuple
        The Matplotlib ``Axes`` object and the ``Line2D`` artist.
    """
    # -------------------------------------------------
    # Convert possible list inputs to numpy arrays
    # -------------------------------------------------
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    # -------------------------------------------------
    # Theme handling
    # -------------------------------------------------
    plt.style.use('dark_background' if dark else 'default')

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # -------------------------------------------------
    # X‑axis handling
    # -------------------------------------------------
    x_plot = x.copy()
    if xlog:
        # Replace non‑positive values with a tiny positive placeholder
        nonpos = x_plot <= 0
        if np.any(nonpos):
            pos_vals = x_plot[x_plot > 0]
            replacement = np.min(pos_vals) * 0.1 if pos_vals.size else eps
            x_plot[nonpos] = replacement
        ax.set_xscale('log')
        # Force at most `max_ticks` major ticks
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=max_ticks, subs=[1]))
        # Pad limits if the data span is too narrow for two ticks
        lo, hi = ax.get_xlim()
        if hi / lo < 10:
            ax.set_xlim(lo / 10, hi * 10)

    # -------------------------------------------------
    # Y‑axis handling
    # -------------------------------------------------
    y_plot = y.copy()
    if ylog:
        if np.all(y >= 0):
            # Positive‑only log scale – sanitise and set locator
            nonpos = y_plot <= 0
            if np.any(nonpos):
                pos_vals = y_plot[y_plot > 0]
                replacement = np.min(pos_vals) * 0.1 if pos_vals.size else eps
                y_plot[nonpos] = replacement
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=max_ticks, subs=[1]))
            lo, hi = ax.get_ylim()
            if hi / lo < 10:
                ax.set_ylim(lo / 10, hi * 10)

        elif np.all(y < 0):
            # All negative – custom “negative‑log” transformation (unchanged)
            mask_large = y_plot < -1           # y <= -1
            mask_small = (y_plot >= -1) & (y_plot <= 0)  # -1 < y < 0
            y_plot[mask_large] = -np.log10(-y_plot[mask_large])
            y_plot[mask_small] = 0.0
            ax.set_ylabel(f'custom negative‑log scale ({ylabel})', fontsize=14)
            # After transformation the axis is linear – limit tick count
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))

        else:
            # Mixed signs – use symlog (unchanged) and cap linear region ticks
            ax.set_yscale('symlog', linthresh=linthresh, linscale=1.0)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))

    # -------------------------------------------------
    # Plot the data (use the possibly modified x_plot/y_plot)
    # -------------------------------------------------
    line, = ax.plot(
        x_plot if xlog else x,
        y_plot,
        marker,
        alpha=0.9,
        color=color,
        linewidth=linewidth,
        label=label,
    )

    # -------------------------------------------------
    # Axis labels and tick styling
    # -------------------------------------------------
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)

    # -------------------------------------------------
    # Show figure (if requested)
    # -------------------------------------------------
    if show:
        plt.show(block=False)
        try:
            ax.figure.tight_layout()
        except Exception:
            pass

    return ax, line

def scatterfield(x,y, masked_z, xlabel='vmJ2 [MPa]', ylabel='temperature [K]',
                 zlabel='MSRE', logcolors=True, vmin=1e-4, vmax=1e0,
                 cmap='viridis', edgecolors=None, dark=1, flagnan=True, histplot=1, msize=30):
    """
    Plots a 2D scatter plot of surrogate model validation data
    
    Parameters
    ----------
    x : list
        x data.
    y : list
        y data.
    masked_z : numpy (masked) array
        z data.
    xlabel : str
        x axis label.
    ylabel : str
        y axis label.
    zlabel : str
        z axis label.
    save : bool/string, optional
        path for saving figure. The default is 0 (do not save).
    cmap : str, optional
        colormap. The default is 'viridis'.
    dark : bool, optional
        dark theme for plot. The default is 1.
    flagnan : bool, optional
        add nans as red data points in plots. The default is 1.
    histplot : bool, optional
        histogram yes or no? The default is 1.
    msize : int
        marker size of data points in plot.

    Returns
    -------
    fig, ax: figure and axis handle

    """
    
    ## -------------------------------------------------------------------------------
    ## initialize field figure
    ## -------------------
    if histplot==1:
        fig = plt.figure(figsize=(6,8))
    elif histplot==0:
        fig = plt.figure(figsize=(8,6))
    # fig.suptitle(zlabel, fontsize=20)
    # apply dark background theme, based on argument "dark"
    darktheme(dark)
    
    ##-------------------------------------------------------------------------------
    ## plot the msre field data in log-scale colors (excludes NaN's automatically)
    ## -------------------
    if histplot == 1:
      ax1 = fig.add_subplot(2, 1, 1)
    elif histplot == 0:
        ax1 = fig.add_subplot(1, 1, 1)
    try:
        if logcolors == 'sym':
            p1 = ax1.scatter(x, y, c=masked_z.data, s=msize, marker=',',
                            cmap=cmap, 
                            edgecolors=edgecolors,
                            alpha=0.95,
                            norm=matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.01))
        elif logcolors == True:
            p1 = ax1.scatter(x, y, c=masked_z.data, s=msize, marker=',',
                            cmap=cmap, 
                            edgecolors=edgecolors,
                            alpha=0.95,
                            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        elif logcolors == False:
            try:
                p1 = ax1.scatter(x, y, c=masked_z.data, s=msize, marker=',',
                                cmap=cmap, 
                                edgecolors=edgecolors,
                                alpha=0.95,
                                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            except:
                print("Masked data plot failed. Try plotting without assuming masked data.")
                p1 = ax1.scatter(x, y, c=masked_z, s=msize, marker=',',
                                cmap=cmap, 
                                edgecolors=edgecolors,
                                alpha=0.95,
                                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        # colorbar creation
        cbar1 = plt.colorbar(p1)
        cbar1.set_label(zlabel, fontsize=20)
        cbar1.ax.tick_params(labelsize=18)
    except:
        print("Not able to plot fig for. Return empty axis")
        return fig, ax1
        
        
    ## -------------------------------------------------------------------------------
    ## plot histogram on logarithmic x-scale with equally sized bins (in log-space)
    ## -------------------
    if histplot==1:
        ax_hist = fig.add_subplot(2,1,2) 
        if logcolors== True:
            
            plt.xscale('log')
            p_hist = ax_hist.hist(masked_z.data, 
                                  bins=np.logspace(np.log10(vmin),np.log10(vmax), 50), 
                                  edgecolor = "black")
        elif logcolors == 'sym':
            plt.xscale('log')
            # raise IOError("debug")
            # correct the vmin for the histogram, which uses absolute values
            
            if vmin < 0:
                vmin_hist = 1e-4
            else:
                vmin_hist = vmin
            p_hist = ax_hist.hist(np.abs(masked_z.data), 
                                  bins=np.logspace(np.log10(vmin_hist),np.log10(vmax), 50), 
                                  edgecolor = "black")

        elif logcolors == False:
            plt.xscale('linear')
            try:
                p_hist = ax_hist.hist(masked_z.data, 
                                    bins=np.linspace(vmin,vmax, 50), 
                                    edgecolor = "black")
            except:
                p_hist = ax_hist.hist(masked_z, 
                                  bins=np.linspace(vmin,vmax, 50), 
                                  edgecolor = "black")
    # plt.xticks(ticks=[vmin, np.median(np.log([vmin, vmax])), vmax])
    # plt.xlim((vmin, vmax))
    
    ## -------------------------------------------------------------------------------
    ## plot the NaN data (false=0, true=1)
    ## -------------------
    if flagnan == True and isinstance(masked_z, np.ma.MaskedArray) and len(masked_z.mask) > 0:
        pnan = ax1.scatter(x[masked_z.mask==True], y[masked_z.mask], 
                          c = masked_z.mask[masked_z.mask], 
                          s=msize-5, marker='o', cmap='bwr', alpha=0.25, edgecolors= 'k', vmin=0, vmax=1)
        
    ## -------------------------------------------------------------------------------
    ## axis and figure settings 
    ## -------------------
    ax1.set_xlabel(xlabel, fontsize=24)    # axis labels
    ax1.set_ylabel(ylabel, fontsize=24)    # axis labels
    ax1.tick_params(labelsize = 24)      # axis properties
    ax1.locator_params(axis='x', nbins=5)   # number of ticks on x axis
    ax1.locator_params(axis='y', nbins=5)   # number of ticks on y axis
    #ax1.set_aspect('auto')
    if histplot==1:
        ax_hist.set_xlabel(zlabel, fontsize=24)    # axis labels
        ax_hist.set_ylabel('Occurrences', fontsize=24)    # axis labels
        ax_hist.tick_params(labelsize = 24)      # axis properties
    
    fig.tight_layout()    
    plt.show(block=False)
    
    ## -------------------------------------------------------------------------------
    ## return figure handle
    ## ----------------------
    return fig, ax1

def map_empty_elements(nodes, conn, empty_elements, output_dir, xy=[0,1], save=False):
    """
    Plot the location where data is empty

    Inputs:
        nodes : array-like, float
            nodes coordinates per dof (column)
        conn : array-like, float
            node numbers per element (element per row)
        empty_elements : list of int
            index of empty elements in conn
        xy = list of int
            x and y variable to consider and plot 
            For HT9 model, the order is: (0,1,2,3,4,5) = (vmJ2, temperature, evm, rhoc, rhow, flux)
        save : bool or str
            False or filename
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    print("Plot data density per element...")
    # Plot the nodes per element
    for i in range(len(conn)):
        # plot the nodes 
        ax.scatter(nodes[conn[i], xy[0]], nodes[conn[i], xy[1]], c='blue', marker='s', s=5)
        # compute rectangle origin (xy) and width (stress) and height (temperature)
        origin = [nodes[conn[i], xy[0]].min(), nodes[conn[i], xy[1]].min()]
        width = nodes[conn[i], xy[0]].max() - nodes[conn[i], xy[0]].min()
        height = nodes[conn[i], xy[1]].max() - nodes[conn[i], xy[1]].min()

        if isinstance(empty_elements, list) and i in empty_elements:
            rect = mpatches.Rectangle(origin, width, height, fill=True, alpha=0.35, facecolor='white',  edgecolor='k', linewidth=1)
        else:
            rect = mpatches.Rectangle(origin, width, height, fill=False, alpha=1, edgecolor='k', linewidth=1)
        ax.add_patch(rect)

        ax.autoscale()
    if output_dir is None or save == False:
        plt.show(block=False)
    elif isinstance(output_dir, str) and save != False:
        # Create the directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, save))

def darktheme(dark):
    """
    sets the background theme of the figure light or dark
    
    input:
        dark : boolean
            dark = True, dark theme
            dark = False, light theme
    output:
        none
    """
    # dark theme
    if dark == True:
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#2f2f2f'
        plt.rcParams['savefig.facecolor'] = '#2f2f2f'
        plt.rcParams['legend.facecolor'] = '#2f2f2f'
        plt.rcParams['font.family']= 'serif'
        plt.rcParams['savefig.dpi']=200
    #light theme
    elif dark == False:
        plt.style.use('default')
        plt.rcParams['font.family']='serif'
        plt.rcParams['savefig.dpi']=200


def test_report_plots(valid_output : dict, z_key = "evm", max_time_in_maps=1e9, flagnan=True, logtraces=False,
                       plot_surf=True, plot_runtime=False, path = None, name_addon = "", dark = True):
    """
    Report plots with multiple error metrics for analyzing surrogate models.
    Calls different analyzers and plotters.


    inputs:
    - valid_output : dict
        Contains "valid" and "data" results from the surrogate and reference data.
    - z_key : str
        String with the output key to plot: typically "evm", "rhoc", or "rhow" for creep surrogates.
    - path : str
        String to path for storing figs. If None, nothing is saved/stored.
    - dark : bool
        True/False dark theme for plots

  
    """

    # Convert some dict keys to keys present in the data dict
    if z_key == 'rhoc':
        z_key_data = 'rhoc'
    elif z_key == 'evm':
        z_key_data = 'evm'
    elif z_key == 'rhow':
        z_key_data = 'rhow'


    ## Extract data maps at particular increments
    end_evm_data = sma.datamap(valid_output['data'], x = 'vmJ2', y = 'temperature', z=z_key_data, inc=-1)
    end_evm_valid = sma.datamap(valid_output['valid'], x = 'vmJ2', y = 'temperature', z=z_key, inc=-1)
    simulation_time_valid = sma.datamap(valid_output['valid'], x = 'vmJ2', y = 'temperature', z='simulation_time', inc=-1)
    errorinc= sma.errorinc(valid_output, x='vmJ2', y='temperature', z=z_key, min_relative_time=0.3, max_time=max_time_in_maps, eps=1e-20)

    # raise IOError("debug")

    # Plot data and surrogate (valid) response at particular increment
    if plot_runtime == True:
        fig0, ax0 = scatterfield(simulation_time_valid['X'], simulation_time_valid['Y'], simulation_time_valid['Z'], 
                                    zlabel=' SM simulation time [s]', logcolors=0, 
                                    vmin=simulation_time_valid['Z'].min(), vmax=simulation_time_valid['Z'].max(),  
                                    cmap='RdYlGn_r', 
                                    # cmap='seismic',
                                    edgecolors='k', flagnan=flagnan,
                                    dark=dark)
    
    fig4, ax4 = scatterfield(errorinc['X'], errorinc['Y'], errorinc['REL_RMSE'], 
                                xlabel="Von Mises Stress [MPa]",
                                ylabel="Temperature [K]",
                                zlabel=' Rel. RMSE', 
                                vmin=1e-2, vmax=1e0, 
                                cmap='RdBu_r', 
                                # cmap='seismic',
                                edgecolors='k',
                                # vmin=1e-2, vmax=1e2, cmap='PiYG_r',                    
                                logcolors=1, flagnan=flagnan,
                                dark=dark)
    # Triangulation for surface plot
    # Extract the data points
    x = errorinc['X']
    y = errorinc['Y']
    z = errorinc['REL_RMSE'].data if isinstance(errorinc['REL_RMSE'], np.ma.MaskedArray) else errorinc['REL_RMSE']

    # Create a mask for finite values
    good_points = np.isfinite(z)
    x_good = x[good_points]
    y_good = y[good_points]
    z_good = z[good_points]

    if plot_surf == True:
        # Now create triangulation with only the valid data points
        triang = tri.Triangulation(x_good, y_good)

        # Create the contour plot
        cf = ax4.tricontourf(triang, z_good,
                            levels=np.logspace(-2, 0, 100),
                            cmap='RdBu_r',
                            # cmap='RdYlGn_r',
                            norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e0),
                            extend='both')

    ## ==================================
    # plot individual validation runs

    # define locations of individual plot runs
    temps = np.linspace(y.max(), y.min(), 6)
    stresses = np.linspace(x.min(), x.max(), 6)
    cnt = 0
    click = []
    for t in temps:
        for s in stresses:
            click.append((s, t))

    # index from coordinates list of tuples (stress[MPa], temperature[K])
    sims = sma.coord2index(errorinc['X'], errorinc['Y'], click)

    fig9, axs = plt.subplots(6, 6, figsize=(12, 8))

    # raise IOError("debug")
    # plot the individual plots based on the click
    numsims = list(valid_output['valid'].keys())
    simnumber = []
    for i in range(len(sims)):
        row = i // 6
        col = i % 6 
        sim = numsims[sims[i]]
        simnumber.append(sim)
        plotxy(valid_output['data'][sim]['t'], valid_output['data'][sim][z_key_data], xlog=logtraces, ylog=logtraces, marker='.', color='green', ax=axs[row,col], dark=dark)
        plotxy(valid_output['valid'][sim]['t'], valid_output['valid'][sim][z_key], xlog=logtraces, ylog=logtraces, marker='.-', color='orange', ax=axs[row,col], dark=dark)
        # If the input is an array
        if isinstance(valid_output['data'][sim]['vmJ2'], np.ndarray) and isinstance(valid_output['data'][sim]['temperature'], np.ndarray):
            subtitle_text = "$\sigma$: {:4.0f} \n $T$: {:4.0f}".format(valid_output['data'][sim]['vmJ2'][-1], valid_output['data'][sim]['temperature'][-1])
        # If the input is a single value
        elif isinstance(valid_output['data'][sim]['vmJ2'], (float, int)) and isinstance(valid_output['data'][sim]['temperature'], (float, int)):
            subtitle_text = "$\sigma$: {:4.0f} \n $T$: {:4.0f}".format(valid_output['data'][sim]['vmJ2'], valid_output['data'][sim]['temperature'])
        else:
            # Handle the case where the input is not an array or a single value
            subtitle_text = "Error: Invalid input format"        
        axs[row,col].text(0.7, 0.2, subtitle_text, ha='center', va='center', transform=axs[row,col].transAxes, fontsize=10)
        axs[row,col].set_xlabel("[s]", fontsize=12)
        axs[row,col].set_ylabel(z_key, fontsize=12)
        axs[row,col].tick_params(labelsize = 12)

    fig9.tight_layout()

    # Pick some individual runs from an error field
    # smp.fieldpicker(valid_output, errorinc, dark=True)

    ## ===================
    # save the figures and store pickle with plotted data
    if path != None:
        if plot_runtime == True:
            fig0.savefig(os.path.join(path, "Testing_SM_simulated-time_" + name_addon+ ".png"))
        fig4.savefig(os.path.join(path, "Testing_REL_RMSE_" + z_key + "_" + name_addon + ".png"))
        fig9.savefig(os.path.join(path, "Testing_runs_6x6_" + z_key + "_" + name_addon + ".png"))

    ## store the plotted data in a dict
    test_plots = {"end_evm_data" : end_evm_data,
                "end_evm_valid" : end_evm_valid,
                "errormaps" : errorinc,
                "trace_sims" : simnumber}

    if path is not None:
        print("\nStore validation plots in .pickle file...")
        sys.stdout.flush()

        # Create the directories if they don't exist
        os.makedirs(path, exist_ok=True)
        # Create the file path
        file_path = os.path.join(path, z_key + "_test_plots.pickle")
        # Open the file in binary write mode
        with open(file_path, "wb") as file:
            # Dump the dictionary into the file using pickle
            pickle.dump(test_plots, file)
    
    return  test_plots