#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

Utlity library for generating shape functions and finite element grids for surrogate modeling.

"""
import numpy as np
from matplotlib.path import Path
import itertools, sys
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
import warnings

# Import R-tree functionality if available
try:
    from utils import sm_rtree
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    print("Warning: R-tree indexing not available. Install rtree package for faster element lookup.")
    sys.stdout.flush()

## OLD FINDER 
def find_element(this_input_point, nodes, conn, shapefunc_degree='linear', i=None):
    """
    Determine which finite element contains the given input point.

    Parameters:
        - this_input_point: np.array, shape (D,)   -- D-dimensional input
        - nodes: np.array, shape (num_nodes, D)
        - conn: np.array, shape (num_elements, N)  -- connectivity (indices of nodes per element)
        - shapefunc_degree: 'linear' or something else
        - i: optional simulation index (for warning messages)

    Returns:
        - node_range: min/max per input dimension for the matching element
        - Iel: index of the element, or False if not found
    """
    num_dims = nodes.shape[1]
    num_elements = conn.shape[0]

    # Gather values at nodes for all elements: shape (num_elements, N_nodes_per_elem, num_dims)
    element_nodes = nodes[conn]  # shape (num_elements, N_nodes_per_elem, D)

    # Compute min/max bounds per element and dimension
    min_vals = np.min(element_nodes, axis=1)  # shape (num_elements, D)
    max_vals = np.max(element_nodes, axis=1)

    # Create mask: True where point is within all element bounds
    mask = np.ones(num_elements, dtype=bool)
    for d in range(num_dims):
        mask &= (this_input_point[d] >= min_vals[:, d]) & (this_input_point[d] <= max_vals[:, d])

    Iel = np.where(mask)[0]

    # Case 1: No matching element
    if len(Iel) == 0:
        print(f"\n\nWARNING: No element contains data point.\nData goes out of bounds.\nSTOP simulation {i if i is not None else ''}.\n")
        sys.stdout.flush()
        return False, False

    # Case 2: Multiple elements
    elif len(Iel) > 1:
        if shapefunc_degree != 'linear':
            print(f"\nWARNING: Local coordinate falls in more than one element in sim {i if i is not None else ''}. Shape functions are not 'linear': I do not know how to handle this case.")
            sys.stdout.flush()
        else:
            # Check for boundary point
            for d in range(num_dims):
                unique_vals = np.unique(element_nodes[Iel, :, d])
                if len(unique_vals) == 3 and np.isclose(this_input_point[d], unique_vals[1]):
                    print("\nData point exactly on element boundary: pick one of two adjacent elements.")
                    Iel = Iel[0]
                    break
            else:
                print(f"\nWARNING: in sim {i if i is not None else ''}, input point lies in multiple elements but not on a clear boundary.")
                sys.stdout.flush()
                return False, False
    else:
        Iel = Iel[0]

    # Compute range (min/max) per dimension for the selected element
    node_coords = element_nodes[Iel]  # shape (N_nodes_per_elem, D)
    node_range = []
    for d in range(num_dims):
        node_range.extend([np.min(node_coords[:, d]), np.max(node_coords[:, d])])

    return node_range, Iel
## END OLD ELEMENT FINDER (not used)

# Only works for 2D triangles
def point_in_simplex(point, simplex_nodes):
    """
    Check if a point lies inside a simplex using barycentric coordinates.
    Optimized for 2D simplices (triangles).
    """
    point = np.array(point)  # Ensure it's an array for shape access
    dim = point.shape[0]
    A = simplex_nodes[1:] - simplex_nodes[0]  # shape: (2, 2) for a triangle
    b = point - simplex_nodes[0]  # shape: (dim,)

    try:
        lambdas = np.linalg.solve(A.T, b)
    except np.linalg.LinAlgError:
        return False

    if np.any(lambdas < -1e-10) or np.sum(lambdas) > 1 + 1e-10:
        return False
    return True

def point_in_hypercube(point, cube_nodes):
    """
    Check if a point lies inside a hypercube.
    """
    point = np.array(point)
    min_vals = np.min(cube_nodes, axis=0)
    max_vals = np.max(cube_nodes, axis=0)
    return np.all((point >= min_vals) & (point <= max_vals))

def eval_shape_mixed_point(point_dict, node_coords, mesh_specs, debug=False):
    """
    Evaluate shape‑function values and their physical‑space gradients for a mixed
    (simplex + hyper‑cube) element.

    Parameters
    ----------
    point_dict : dict
        Mapping from dimension name to coordinate (scalar or 1‑element array).
    node_coords : np.ndarray, shape (n_nodes, ndim)
        Coordinates of the element’s nodes in the parametric space.
    mesh_specs : dict
        Must contain:
            - 'element_numbers' : dict of dimension name → count
            - 'tri_elements'    : list of dimensions that form a simplex
    debug : bool, optional
        If True, prints intermediate data for verification.

    Returns
    -------
    N_row  : np.ndarray, shape (1, n_nodes)
        Shape‑function values ordered exactly like the nodes appear in
        ``node_coords`` (original permutation is retained).
    dN_row : np.ndarray, shape (ndim, 1, n_nodes)
        Physical gradient ∂N/∂x for each parametric direction.
        The first axis indexes the spatial direction, the second is a singleton
        kept for backward compatibility, and the third indexes the node.
    """
    # ------------------------------------------------------------------ #
    # 0.  Unpack mesh information
    # ------------------------------------------------------------------ #
    dim_names   = list(mesh_specs['element_numbers'].keys())
    simplex_dims = mesh_specs.get('tri_elements', [])
    ndim        = len(dim_names)

    simplex_idx   = [dim_names.index(k) for k in simplex_dims]          # simplex axes
    hypercube_idx = [i for i in range(ndim) if i not in simplex_idx]    # hyper‑cube axes

    # ------------------------------------------------------------------ #
    # 1.  Build the parametric coordinate vector ``point``
    # ------------------------------------------------------------------ #
    point = np.array(
        [point_dict[k][0] if isinstance(point_dict[k], np.ndarray) else point_dict[k]
         for k in dim_names],
        dtype=float).reshape(1, -1)   # (1, ndim)

    # ------------------------------------------------------------------ #
    # 2.  SIMPLEX part (linear simplex → barycentric coordinates)
    # ------------------------------------------------------------------ #
    if simplex_idx:
        X_simplex      = point[:, simplex_idx]                         # (1, n_simplex_dim)
        verts_simplex = node_coords[:len(simplex_idx) + 1, simplex_idx]  # (n_vertices, n_simplex_dim)
        lambdas        = map_to_barycentric_nd(X_simplex, verts_simplex)   # (1, n_simplex)
        shape_simplex  = lambdas                                        # (1, n_simplex)
        n_simplex      = shape_simplex.shape[1]

        # ----- derivative of simplex shape functions (reference space) -----
        dshape_simplex = np.zeros((ndim, n_simplex))      # (ndim, n_simplex)
        for s_dim in simplex_idx:
            # MOOSE convention: reference interval [-1, 1] → derivative = ±0.5
            dshape_simplex[s_dim, :] = np.array([-0.5, 0.5])
    else:
        shape_simplex  = np.ones((1, 1))
        n_simplex      = 1
        dshape_simplex = np.zeros((ndim, 1))

    # ------------------------------------------------------------------ #
    # 3.  HYPERCUBE part (linear Lagrange on a tensor‑product cell)
    # ------------------------------------------------------------------ #
    if hypercube_idx:
        X_hyper   = point[:, hypercube_idx]                         # (1, n_hyper_dim)
        mins_hyper = node_coords[:, hypercube_idx].min(axis=0)      # (n_hyper_dim,)
        maxs_hyper = node_coords[:, hypercube_idx].max(axis=0)      # (n_hyper_dim,)
        spans      = np.maximum(maxs_hyper - mins_hyper, 1e-15)     # avoid div‑by‑zero
        local      = (X_hyper - mins_hyper) / spans                # normalized ξ ∈ [0,1]
        # raise IOError("debug: break here to check outputs of local coordinates for hypercube shape functions")
        corners   = list(itertools.product([0, 1], repeat=len(hypercube_idx)))
        n_corners = len(corners)
        shape_hyper = np.zeros((1, n_corners))

        # ----- shape values on [0, 1] -------------------------------- #
        for col_idx, corner in enumerate(corners):
            factors = [
                (1 - local[:, d]) if c == 0 else local[:, d]
                for d, c in enumerate(corner)
            ]
            shape_hyper[:, col_idx] = np.prod(factors, axis=0)

        # ----- reference‑space derivative of hyper‑cube shape functions ---- #
        dshape_hyper = np.zeros((ndim, n_corners))
        for k, dim in enumerate(hypercube_idx):          # loop over parametric directions
            for col_idx, corner in enumerate(corners):
                # derivative magnitude = ±1 (correct for [0, 1] isparametric range)
                sign = -1.0 if corner[k] == 0 else 1.0
                # product of the unchanged factors from the other dimensions
                other = [
                    (1 - local[:, d]) if c == 0 else local[:, d]
                    for d, c in enumerate(corner) if d != k
                ]
                prod_other = np.prod(other, axis=0) if other else np.ones_like(local[:, 0])
                dshape_hyper[dim, col_idx] = sign * prod_other
                
    else:
        shape_hyper   = np.ones((1, 1))
        n_corners     = 1
        dshape_hyper = np.zeros((ndim, 1))

    # ------------------------------------------------------------------ #
    # 4.  Tensor product (reference‑space values and gradient)
    # ------------------------------------------------------------------ #
    N_elem = (shape_simplex[:, :, None] * shape_hyper[:, None, :]).reshape(1, -1)   # (1, n_simplex * n_corners)

    dN_ref = (
        dshape_simplex[:, :, None] * shape_hyper[:, None, :] +
        shape_simplex[:, :, None] * dshape_hyper[:, None, :]
    ).reshape(ndim, -1)   # (ndim, n_simplex * n_corners)

    # ------------------------------------------------------------------ #
    # 5.  Build element Jacobian J (reference → physical)
    # ------------------------------------------------------------------ #
    # Simplex Jacobian (full matrix, size = n_simplex_dims)
    if simplex_idx:
        # verts_simplex shape (n_vertices, n_simplex_dims)
        # columns are vectors from vertex 0 to the other vertices
        J_simplex = verts_simplex[1:] - verts_simplex[0]          # (n_simplex_dims, n_simplex_dims)
    else:
        J_simplex = np.empty((0, 0))

    # Hypercube Jacobian (diagonal matrix with spans)
    if hypercube_idx:
        J_hyper = np.diag(spans)                                 # (n_hyper_dims, n_hyper_dims)
    else:
        J_hyper = np.empty((0, 0))

    # Assemble block‑diagonal Jacobian respecting the order
    if J_simplex.size and J_hyper.size:
        J_full = np.block([
            [J_simplex,               np.zeros((J_simplex.shape[0], J_hyper.shape[1]))],
            [np.zeros((J_hyper.shape[0], J_simplex.shape[1])), J_hyper]
        ])
    elif J_simplex.size:
        J_full = J_simplex
    elif J_hyper.size:
        J_full = J_hyper
    else:
        J_full = np.eye(ndim)   # should never happen, but keep safe

    # Inverse Jacobian (constant for linear elements)
    invJ = np.linalg.inv(J_full)   # (ndim, ndim)

    # ------------------------------------------------------------------ #
    # 6.  Apply chain rule: physical gradient = invJ @ reference gradient
    # ------------------------------------------------------------------ #
    dN_phys = invJ @ dN_ref        # (ndim, n_simplex * n_corners)
    # raise IOError("debug: break here to check outputs of dN_phys for mixed shape functions")
    # ------------------------------------------------------------------ #
    # 7.  Permutation to match node order in node_coords
    # ------------------------------------------------------------------ #
    # Build the same permutation used in the original code
    hypercube_order = (
        list(itertools.product([0, 1], repeat=len(hypercube_idx)))
        if hypercube_idx else [()]
    )
    perm = np.zeros(node_coords.shape[0], dtype=int)

    for local_pos in range(node_coords.shape[0]):
        # simplex index (closest simplex vertex)
        s = 0
        if simplex_idx:
            sc = node_coords[local_pos, simplex_idx]
            diffs = np.linalg.norm(verts_simplex - sc[None, :], axis=1)
            s = int(np.argmin(diffs))

        # hypercube index (closest corner)
        h_index = 0
        if hypercube_idx:
            hc = node_coords[local_pos, hypercube_idx]
            local_coord = (hc - mins_hyper) / spans
            local_coord = np.clip(local_coord, 0, 1)
            bits = (local_coord > 0.5).astype(int)
            corner_tuple = tuple(int(b) for b in bits)
            try:
                h_index = hypercube_order.index(corner_tuple)
            except ValueError:
                # fallback: nearest corner (unlikely)
                dists = [np.linalg.norm(local_coord - np.array(t)) for t in hypercube_order]
                h_index = int(np.argmin(dists))

        perm[local_pos] = s * n_corners + h_index

    # Apply permutation to both values and gradients
    N_row = N_elem[:, perm]                     # (1, n_nodes)
    dN_row = dN_phys[:, perm]                  # (ndim, n_nodes)
    dN_row = dN_row[:, None, :]                # (ndim, 1, n_nodes) – keep dummy axis

    # ------------------------------------------------------------------ #
    # 8.  Optional debugging output
    # ------------------------------------------------------------------ #
    if debug:
        print("point            :", point)
        print("simplex idx      :", simplex_idx)
        print("hypercube idx    :", hypercube_idx)
        print("shape simplex    :", shape_simplex)
        print("shape hypercube  :", shape_hyper)
        print("J (full)         :", J_full)
        print("invJ             :", invJ)
        print("perm             :", perm)
        print("N (shape)       :", N_row)
        print("dN/dξ (reference) :", dN_ref[:, perm])
        print("dN/dx (gradient) :", dN_row)

    return N_row, dN_row

def point_in_element_mixed(point, elem_nodes, mesh_specs):
    """
    Check if a point is inside an element using the appropriate method based on element type.
    
    Parameters:
    -----------
    point : ndarray
        Coordinates of the point to check
    elem_nodes : ndarray
        Coordinates of the element nodes
    mesh_specs : dict
        Mesh specifications containing element type information
    
    Returns:
    --------
    bool
        True if the point is inside the element, False otherwise
    """
    point = np.array(point)
    D = elem_nodes.shape[1]
    num_nodes_in_element = elem_nodes.shape[0]
    
    # Expected node counts for different element types
    expected_simplex_nodes = D + 1
    expected_hypercube_nodes = 2**D
    expected_extruded_nodes = 3 * (2**(D-2))  # 2D triangle extruded in (D-2) dimensions
    
    # Determine element type based on number of nodes
    if num_nodes_in_element == expected_simplex_nodes:
        return point_in_simplex(point, elem_nodes)
    elif num_nodes_in_element == expected_hypercube_nodes:
        return point_in_hypercube(point, elem_nodes)
    elif num_nodes_in_element == expected_extruded_nodes:
        # For extruded-simplex, need to determine which dimensions are simplex vs hypercube
        tri_dims = mesh_specs.get('tri_elements', [])
        element_numbers = mesh_specs.get('element_numbers', {})
        dim_order = list(element_numbers.keys())
        
        # Find indices of simplex dimensions and extrusion dimensions
        simplex_indices = np.array([i for i, dim in enumerate(dim_order) if dim in tri_dims])
        extrusion_indices = np.array([i for i, dim in enumerate(dim_order) if dim not in tri_dims])
        
        # For this single element, get the first three nodes which define the triangle
        triangle_nodes = elem_nodes[:3]
        
        # Check if point is within the simplex projection
        in_simplex = point_in_simplex(point[simplex_indices], triangle_nodes[:, simplex_indices])
        
        # Use point_in_hypercube for the extrusion dimensions
        in_ext = point_in_hypercube(point[extrusion_indices], elem_nodes[:, extrusion_indices])
        
        return in_simplex and in_ext
    else:
        # Unknown element type
        return False

def find_element_mixed_original(point, nodes, conn, mesh_specs, i=None, verbose=False):
    """
    Original brute force implementation of find_element_mixed. Used as a fallback when R-tree is not available.
    This version is kept for compatibility and to ensure correctness of the R-tree implementation.
    """
    point = np.array(point)
    num_elements = conn.shape[0]
    D = nodes.shape[1]

    tri_dims = mesh_specs.get('tri_elements', [])
    element_numbers = mesh_specs.get('element_numbers', {})
    dim_order = list(element_numbers.keys())

    for Iel in range(num_elements):
        node_ids = conn[Iel]
        elem_nodes = nodes[node_ids]
        
        # Check if point is inside this element using the appropriate method
        if point_in_element_mixed(point, elem_nodes, mesh_specs):
            min_vals = np.min(elem_nodes, axis=0)
            max_vals = np.max(elem_nodes, axis=0)
            node_range = [v for pair in zip(min_vals, max_vals) for v in pair]
            return node_range, Iel

    if verbose:
        print(f"WARNING: No matching element found for point {point} in sim {i if i is not None else ''}.")
        sys.stdout.flush()
    return False, False

def find_element_mixed(point, nodes, conn, mesh_specs, i=None, verbose=False, rtree_index=None):
    """
    Determine which element contains the given point in a mixed mesh.
    
    Parameters:
    -----------
    point : ndarray, shape (D,)
        D-dimensional input point to locate in the mesh.
    nodes : ndarray, shape (num_nodes, D)
        Node coordinates in D-dimensional space.
    conn : ndarray, shape (num_elements, N)
        Connectivity matrix (indices of nodes per element).
    mesh_specs : dict
        Mesh specifications including element types, etc.
    i : int, optional
        Simulation index for warning messages.
    verbose : bool
        Whether to print detailed information.
    rtree_index : sm_rtree.MeshRtreeIndex, optional
        Pre-built R-tree spatial index for fast element lookup.
    
    Returns:
    --------
    node_range : list or False
        Min/max range per dimension for the matching element, or False if not found.
    element_idx : int or False
        Index of the matching element, or False if not found.
    """
    # Use R-tree index for fast lookup if available
    if rtree_index is not None and RTREE_AVAILABLE:
        if verbose:
            print("Using R-tree index for fast element lookup")
            sys.stdout.flush()
        return sm_rtree.find_element_rtree(point, nodes, conn, mesh_specs, rtree_index, verbose)
    
    # Fall back to brute force search if R-tree not available or not provided
    return find_element_mixed_original(point, nodes, conn, mesh_specs, i, verbose)

def map_to_barycentric_nd(X, verts):
    """
    Compute barycentric coordinates of points X with respect to a simplex defined by verts.

    Args:
        X: (N_points, ndim) array of points
        verts: (n_nodes=ndim+1, ndim) array of simplex vertex coordinates

    Returns:
        lambdas: (N_points, ndim + 1) barycentric coordinates
    """
    ndim = verts.shape[1]

    if ndim == 1:
        v0, v1 = verts[0, 0], verts[1, 0]
        span = v1 - v0
        if np.isclose(span, 0.0):
            raise ValueError("Degenerate 1D simplex (zero length)")
        t = (X[:, 0] - v0) / span
        lambdas = np.column_stack([1 - t, t])
        return lambdas

    # General ND case
    T = (verts[1:] - verts[0]).T  # (ndim, ndim)
    rhs = (X - verts[0]).T        # (ndim, N_points)

    # Least-squares solve
    sol, residuals, rank, s = np.linalg.lstsq(T, rhs, rcond=None)

    lambdas = np.vstack((1 - sol.sum(axis=0), sol))  # (ndim + 1, N_points)
    return lambdas.T  # (N_points, ndim + 1)

def data_density_per_element_mixed(data_array, nodes, conn, mesh_specs, density_thresh=50, verbose=False):
    """
    Compute density of data points per element in a mixed mesh using provided scalar
    point_in_simplex and point_in_hypercube checks.

    Parameters:
        data_array : (N_points, D) ndarray
            Input points in D-dim space.
        nodes : (N_nodes, D) ndarray
            Node coordinates.
        conn : (N_elements, N_nodes_per_element) ndarray
            Connectivity matrix.
        mesh_specs : dict
            Must contain:
                - 'tri_elements': list of simplex element indices
                - 'element_numbers': dict for element indexing (not directly used here)
                - optionally 'mesh_filtering' dict with 'density_thresh'
        density_thresh : int
            Minimum points count threshold to consider element dense.
        verbose : bool
            Print progress.

    Returns:
        sparse_elements : list of element indices with point counts below threshold.
        sparse_centroids : (M, D) ndarray of centroid coords of sparse elements.
    """

    n_elem = conn.shape[0]
    n_points = data_array.shape[0]
    D = data_array.shape[1]

    if 'mesh_filtering' in mesh_specs and 'density_thresh' in mesh_specs['mesh_filtering']:
        density_thresh = mesh_specs['mesh_filtering']['density_thresh']

    tri_set = set(mesh_specs.get('tri_elements', []))
    density = np.zeros(n_elem, dtype=int)

    if verbose:
        print(f"Counting data density for {n_points} points across {n_elem} elements (dim={D})")

    # For efficiency, precompute min/max bounding boxes of elements for quick rejection
    elem_bboxes = np.zeros((n_elem, 2, D))  # shape: (elem, min/max, dim)
    for i in range(n_elem):
        elem_nodes = nodes[conn[i]]
        elem_bboxes[i, 0, :] = np.min(elem_nodes, axis=0)
        elem_bboxes[i, 1, :] = np.max(elem_nodes, axis=0)

    # Loop over points: for each point find which elements could contain it based on bbox
    for ip, point in enumerate(data_array):
        # Quick bbox filter: elements where all(point >= min & point <= max)
        candidates = []
        for i in range(n_elem):
            if np.all((point >= elem_bboxes[i, 0]) & (point <= elem_bboxes[i, 1])):
                candidates.append(i)
        # For candidates, run precise point-in-element test
        found = False
        for i in candidates:
            elem_nodes = nodes[conn[i]]
            if i in tri_set:
                inside = point_in_simplex(point, elem_nodes)
            else:
                inside = point_in_hypercube(point, elem_nodes)

            if inside:
                density[i] += 1
                found = True
                break  # assume point belongs to one element only

        if verbose and ip % 1000 == 0:
            print(f"Processed point {ip+1}/{n_points}")

    sparse_elements = [i for i, count in enumerate(density) if count < density_thresh]

    if verbose:
        print(f"Sparse elements (count < {density_thresh}): {sparse_elements}")

    # Centroids of sparse elements (mean node coordinates)
    sparse_centroids = np.array([nodes[conn[i]].mean(axis=0) for i in sparse_elements]) if sparse_elements else np.empty((0, D))

    return sparse_elements, sparse_centroids


def remove_elements_by_index(nodes, conn, highlight_elements, enforce_diagonals=True):
    """
    Remove elements listed in highlight_elements from the mesh and update nodes/conn accordingly.

    Parameters
    ----------
    nodes : (N, D) ndarray
        All nodal coordinates.
    conn : (M, K) ndarray
        Connectivity array with indices into nodes (M elements, K nodes per element).
    highlight_elements : list of int
        Indices of elements in conn to remove.
    enforce_diagonals : bool
        Whether to call enforce_checkerboard_diagonals() after cleanup.

    Returns
    -------
    new_nodes : (N', D) ndarray
        Updated list of nodes (some may be removed).
    new_conn : (M', K) ndarray
        Updated connectivity matrix (element and node indices adjusted).
    """
    if not highlight_elements:
        return nodes, conn  # No removal needed


    # Step 1: Remove selected elements from connectivity
    mask = np.ones(len(conn), dtype=bool)
    mask[highlight_elements] = False
    new_conn = conn[mask]

    # Step 2: Get unique nodes that are still used
    used_node_indices = np.unique(new_conn)
    new_nodes = nodes[used_node_indices]

    # Step 3: Create mapping from old to new indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_node_indices)}

    # Step 4: Remap connectivity indices
    new_conn_mapped = np.vectorize(index_map.get)(new_conn)

    # Step 5: Optional reordering / diagonal enforcement. 
    # TOFIX: seems buggy, but this is also not needed when diagonals were intact before removing elements.
    if enforce_diagonals:
        new_nodes, new_conn_mapped = enforce_checkerboard_diagonals(new_nodes, new_conn_mapped)

    return new_nodes, new_conn_mapped


def data_density_in_element_2d_hypercube(data, nodes, conn, mesh_specs, verbose=False, min_points=50):
    """
    Identify sparse elements in a 2D hypercube (quad) mesh.
    """
    X = np.column_stack([data[k] for k in list(mesh_specs["element_numbers"].keys())[:2]])
    highlight_elements = []

    for i, elem_nodes in enumerate(conn):
        quad = nodes[elem_nodes, :2]  # Ensure it's 2D
        path = Path(quad)
        inside = path.contains_points(X)
        if np.sum(inside) < min_points:
            highlight_elements.append(i)

    if verbose:
        print(f"Flagged {len(highlight_elements)} sparse hypercube elements")

    return highlight_elements


def data_density_in_element_2d_simplex(data, nodes, conn, mesh_specs, verbose=False):
    """
    Count number of data points inside each 2D triangle element and
    return list of sparse element indices below density threshold.

    Parameters
    ----------
    data : dict of 1D arrays
        Must contain keys for all coordinate dimensions (e.g., x, y).
    nodes : (n_nodes, 2) ndarray
        Coordinates of all mesh nodes.
    conn : (n_elem, 3) ndarray
        Triangle connectivity matrix (3 node indices per element).
    mesh_specs : dict
        Must contain the key 'element_numbers' (list of coordinate variable names).
        Also must contain 'mesh_filtering' dict with 'density_thresh'.
    verbose : bool
        If True, prints progress info.

    Returns
    -------
    sparse_elements : list
        List of element indices with data points count below threshold.
    """
    coord_keys = list(mesh_specs['element_numbers'].keys())
    assert len(coord_keys) == 2, "Only 2D supported. \nn dim: {}".format(len(coord_keys))

    points = np.vstack([data[coord_keys[0]], data[coord_keys[1]]]).T
    n_elem = conn.shape[0]

    density_thresh = mesh_specs.get('mesh_filtering', {}).get('density_thresh', 50)  # default to 50

    sparse_elements = []

    for i in range(n_elem):
        tri_nodes = nodes[conn[i], :2]  # shape (3, 2). make sure only first 2 dims
        path = Path(tri_nodes)
        inside_mask = path.contains_points(points)
        count = np.sum(inside_mask)

        if verbose and (i % 1000 == 0 or i == n_elem - 1):
            print(f"Element {i+1}/{n_elem} has {count} points.")

        if count < density_thresh:
            sparse_elements.append(i)
        if verbose:
            print("Sparse Elements: \n{}".format(sparse_elements))

    return sparse_elements

def fe_shapefunc_sparse_mixed(data, nodes, conn, mesh_specs, debug=False):
    """
    Mixed simplex-hypercube finite-element shape function builder.

    - Permutes tensor-product columns to match conn ordering.
    - Assigns each data point to the element where its local shape has the maximum value.
    - Returns N_sparse (Npoints x Nnodes) CSR matrix.

    """
    tol = 1e-14
    dim_names = list(mesh_specs['element_numbers'].keys())
    simplex_dims = mesh_specs.get('tri_elements', [])
    ndim = len(dim_names)

    if nodes.shape[1] > ndim:
        raise IOError(
            f"[fe_shapefunc_sparse_mixed] ERROR: nodes.shape[1]={nodes.shape[1]} exceeds data dimensionality {ndim}."
        )

    simplex_idx = [dim_names.index(k) for k in simplex_dims]
    hypercube_idx = [i for i in range(ndim) if i not in simplex_idx]

    data_matrix = np.vstack([data[k] for k in dim_names])  # (ndim, Npoints)
    Npoints = data_matrix.shape[1]

    best_for_point = {}  # pt_idx -> (max_val, elem_id, permuted_node_ids, N_row_permuted)

    # Loop elements
    for iel in range(conn.shape[0]):
        node_ids = np.asarray(conn[iel, :], dtype=int)
        node_coords = nodes[node_ids, :]  # (n_nodes_elem, ndim)

        # split coords
        simplex_coords = node_coords[:, simplex_idx] if simplex_idx else None
        hypercube_coords = node_coords[:, hypercube_idx] if hypercube_idx else None

        # expected counts
        n_simplex = (len(simplex_idx) + 1) if simplex_idx else 1
        n_hypercube = 2 ** len(hypercube_idx) if hypercube_idx else 1
        expected_nodes = n_simplex * n_hypercube
        if expected_nodes != len(node_ids):
            raise RuntimeError(f"Element {iel}: Node count mismatch expected {expected_nodes} got {len(node_ids)}")

        # bounding box filter
        mins = node_coords.min(axis=0)
        maxs = node_coords.max(axis=0)
        mask = np.all((data_matrix >= mins[:, None] - tol) & (data_matrix <= maxs[:, None] + tol), axis=0)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            continue

        # simplex part
        if simplex_idx:
            X_simplex = data_matrix[simplex_idx, :][:, candidates].T  # (Ncand, ndim_simplex)
            verts_simplex = simplex_coords[:n_simplex, :]
            lambdas = map_to_barycentric_nd(X_simplex, verts_simplex)  # user-provided
            inside = np.all(lambdas >= -tol, axis=1)
            if not np.any(inside):
                continue
            valid_cands = np.array(candidates)[inside]
            shape_simplex = lambdas[inside]  # (Nvalid, n_simplex)
        else:
            valid_cands = np.array(candidates)
            shape_simplex = np.ones((len(valid_cands), 1))

        # hypercube part
        if hypercube_idx:
            X_hyper = data_matrix[hypercube_idx, :][:, valid_cands].T  # (Nvalid, ndim_hyper)
            mins_hyper = hypercube_coords.min(axis=0)
            maxs_hyper = hypercube_coords.max(axis=0)
            spans = np.maximum(maxs_hyper - mins_hyper, 1e-15)
            local = (X_hyper - mins_hyper) / spans  # normalized [0,1]

            corners = list(itertools.product([0, 1], repeat=len(hypercube_idx)))
            n_corners = len(corners)
            shape_hyper = np.zeros((len(valid_cands), n_corners))
            for col_idx, corner in enumerate(corners):
                # tensor product formula
                shape_hyper[:, col_idx] = np.prod([
                    (1 - local[:, d]) if c == 0 else local[:, d]
                    for d, c in enumerate(corner)
                ], axis=0)
        else:
            n_corners = 1
            shape_hyper = np.ones((len(valid_cands), 1))

        # Build N_elem (unpermuted, flatten order: simplex fastest, hypercube slowest as used here)
        N_elem = (shape_simplex[:, :, None] * shape_hyper[:, None, :]).reshape(len(valid_cands), -1)
        if N_elem.shape[1] != expected_nodes:
            raise RuntimeError(f"Element {iel}: N_elem columns {N_elem.shape[1]} != expected {expected_nodes}")

        # --- Important: compute flattened index for each local node (matching the order N_elem assumes) ---
        # We assume N_elem order is: for s in [0..n_simplex-1]: for h in corners(order from itertools.product)
        hypercube_order = list(itertools.product([0,1], repeat=len(hypercube_idx))) if hypercube_idx else [()]
        flattened_expected = []
        for s in range(n_simplex):
            for h_idx in range(len(hypercube_order)):
                flattened_expected.append((s, hypercube_order[h_idx]))  # tuple for identification

        # Now map actual node positions (node_coords order = conn[iel]) to flattened indices
        flattened_index_per_conn_pos = np.zeros(len(node_ids), dtype=int)
        for local_pos in range(len(node_ids)):
            coord = node_coords[local_pos, :]
            # simplex index s
            if simplex_idx:
                sc = coord[simplex_idx]
                # find matching simplex vertex (closest row in verts_simplex)
                diffs = np.linalg.norm( (simplex_coords[:n_simplex,:] - sc[None,:]), axis=1 )
                s = int(np.argmin(diffs))
            else:
                s = 0

            # hypercube corner index
            if hypercube_idx:
                hc = coord[hypercube_idx]
                local_coord = (hc - mins_hyper) / spans
                # clamp to 0..1
                local_coord = np.clip(local_coord, 0.0, 1.0)
                # bits: threshold at 0.5
                bits = (local_coord > 0.5).astype(int)
                # convert to tuple to find index in hypercube_order
                corner_tuple = tuple(int(b) for b in bits)
                try:
                    h_index = hypercube_order.index(corner_tuple)
                except ValueError:
                    # fallback by exact match with tolerance
                    found = False
                    for idx_h, t in enumerate(hypercube_order):
                        if np.allclose(np.array(t), local_coord, atol=1e-8):
                            h_index = idx_h
                            found = True
                            break
                    if not found:
                        # fallback: nearest corner by distance in local space
                        dists = [np.linalg.norm(local_coord - np.array(t)) for t in hypercube_order]
                        h_index = int(np.argmin(dists))
            else:
                h_index = 0

            flattened_index = s * n_corners + h_index
            flattened_index_per_conn_pos[local_pos] = flattened_index

        # Now flattened_index_per_conn_pos tells which column in the raw N_elem corresponds to conn[iel][local_pos]
        # Build permutation array perm such that perm[k] is the column index in raw N_elem that should be placed into column k (node order)
        # i.e., we want N_elem_permuted[:, local_pos] = N_elem[:, flattened_index_per_conn_pos[local_pos]]
        perm = flattened_index_per_conn_pos.tolist()
        # Permute columns of N_elem so column j corresponds to node_ids[j]
        N_elem_permuted = N_elem[:, perm]

        # Optional: fix numerical rounding (cap)
        N_elem_permuted = np.clip(N_elem_permuted, 0.0, 1.0)

        # For each candidate point, consider this element's N_row and keep best (max) element per point
        for i_local, pt_idx in enumerate(valid_cands):
            N_row = N_elem_permuted[i_local]
            max_val = float(N_row.max())
            prev = best_for_point.get(pt_idx)
            if (prev is None) or (max_val > prev[0]):
                # store tuple: (max_val, iel, node_ids_copy, N_row_copy)
                best_for_point[pt_idx] = (max_val, iel, node_ids.copy(), N_row.copy())

    # Assemble final sparse matrix from best_for_point entries
    dat, row, col = [], [], []
    for pt_idx, (_, _, node_ids, N_row) in best_for_point.items():
        dat.extend(N_row.tolist())
        row.extend([int(pt_idx)] * len(node_ids))
        col.extend(node_ids.tolist())

    if len(dat) == 0:
        N_sparse = coo_matrix(([], ([], [])), shape=(Npoints, nodes.shape[0])).tocsr()
    else:
        dat_all = np.array(dat)
        row_all = np.array(row, dtype=int)
        col_all = np.array(col, dtype=int)
        N_sparse = coo_matrix((dat_all, (row_all, col_all)), shape=(Npoints, nodes.shape[0])).tocsr()

    row_sums = np.array(N_sparse.sum(axis=1)).ravel()
    tol = 1e-6
    rows_sum_to_one = np.isclose(row_sums, 1, atol=tol)
    if debug == True:
        if np.abs(np.mean(rows_sum_to_one) - 1) > 1e-6:
            print(f"\033[91m\tPercentage of rows that sum to 1: {np.mean(rows_sum_to_one) * 100:.2f}%\033[0m")
        else: 
            print(f"\033[92m\tPercentage of rows that sum to 1: {np.mean(rows_sum_to_one) * 100:.2f}%\033[0m")

    return N_sparse

def ensure_positive_orientation_nd(conn, nodes, tol=1e-12, verbose=True):
    """
    Ensure all simplices have a positive signed volume (i.e., consistent orientation).

    Args:
        conn: (Nelements, ndim+1) array of element connectivity (simplices)
        nodes: (Nnodes, ndim) array of nodal coordinates
        tol: tolerance for zero volume
        verbose: whether to print number of flips

    Returns:
        conn_fixed: (Nelements, ndim+1) with reordered nodes if necessary
    """
    conn_fixed = conn.copy()
    ndim = nodes.shape[1]
    n_flips = 0

    for iel, elem in enumerate(conn):
        verts = nodes[elem]  # shape (ndim+1, ndim)

        # Compute signed volume via determinant
        T = (verts[1:] - verts[0]).T  # shape (ndim, ndim)
        signed_vol = np.linalg.det(T)

        if signed_vol < -tol:
            # Flip first two nodes to reverse orientation
            conn_fixed[iel, [0, 1]] = conn_fixed[iel, [1, 0]]
            n_flips += 1
        elif abs(signed_vol) < tol:
            raise ValueError(f"Degenerate simplex (zero volume) at element {iel}.")

    if verbose:
        print(f"[Orientation Check] {n_flips} simplices flipped to ensure positive orientation out of {len(conn)} total.")

    return conn_fixed

def refine_lasso_region(preview_nodes, verts, spacing=1e-10, decimals=8):
    """
    Perform 2D lasso-based refinement on possibly extruded mesh nodes.

    Args:
        preview_nodes: (N, D) ndarray of existing mesh node coordinates.
        verts: List of (x, y) vertices defining the lasso polygon.
        spacing: Tolerance for duplicate node detection.
        decimals: Rounding for matching & sorting.

    Returns:
        refined_nodes: (N + K, D) ndarray of updated nodes, sorted.
        new_nodes_full: (K, D) ndarray of newly added nodes (optional for diagnostics).
    """

    if preview_nodes is None or preview_nodes.shape[1] < 2:
        raise ValueError("Expected 2D or higher-dimensional node array.")

    # 1. Work on base 2D projection
    base_coords = np.round(preview_nodes[:, :2], decimals=decimals)

    # 2. Select nodes inside lasso polygon
    path = Path(verts)
    selected_mask = path.contains_points(base_coords)
    selected_coords_2d = base_coords[selected_mask]

    if len(selected_coords_2d) == 0:
        warnings.warn("No nodes selected for refinement.")
        return preview_nodes, np.empty((0, preview_nodes.shape[1]))

    unique_selected_2d = np.unique(selected_coords_2d, axis=0)

    # 3. Generate new refinement nodes (in 2D)
    new_nodes_2d = refine_lasso_region_2d(
        existing_nodes=base_coords,
        selected_nodes=unique_selected_2d,
        spacing=spacing
    )

    # 4. Lift new 2D nodes into full dimension
    D = preview_nodes.shape[1]
    n_new = len(new_nodes_2d)
    new_nodes_full = np.zeros((n_new, D))
    new_nodes_full[:, :2] = new_nodes_2d

    if D > 2:
        z_template = np.mean(preview_nodes[:, 2:], axis=0)
        new_nodes_full[:, 2:] = z_template

    # 5. Combine and sort
    all_nodes = np.vstack([preview_nodes, new_nodes_full])
    all_nodes = np.round(all_nodes, decimals)
    sort_idx = np.lexsort((all_nodes[:, 0], all_nodes[:, 1]))
    sorted_all_nodes = all_nodes[sort_idx]

    return sorted_all_nodes, new_nodes_full

def refine_lasso_region_2d(
    existing_nodes,
    selected_nodes,
    spacing=1e-10,
    return_combined=False,
    return_conn=False
):
    if selected_nodes.shape[1] != 2:
        raise ValueError("Only 2D refinement supported.")

    x_sel = np.unique(selected_nodes[:, 0])
    y_sel = np.unique(selected_nodes[:, 1])

    x_mid = (x_sel[:-1] + x_sel[1:]) / 2
    y_mid = (y_sel[:-1] + y_sel[1:]) / 2

    x_fine = np.unique(np.concatenate([x_sel, x_mid]))
    y_fine = np.unique(np.concatenate([y_sel, y_mid]))

    xx, yy = np.meshgrid(x_fine, y_fine, indexing="xy")
    all_new_points = np.column_stack([xx.ravel(), yy.ravel()])

    decimals = int(-np.log10(spacing))
    existing_set = set(map(tuple, np.round(existing_nodes, decimals)))

    refined = np.array([
        pt for pt in np.round(all_new_points, decimals)
        if tuple(pt) not in existing_set
    ])

    if not return_combined and not return_conn:
        return refined

    all_nodes = np.vstack([existing_nodes, refined])
    all_nodes_unique = np.unique(all_nodes, axis=0)

    return all_nodes_unique

# OLD: for box area refinement
def generate_refinement_nodes(existing_nodes, bbox, spacing=0.1, mode="grid", shape="box", tol=1e-10):
    dim_names = list(bbox.keys())
    ndim = len(dim_names)

    if isinstance(spacing, (int, float)):
        spacing = [spacing] * ndim

    vectors = [np.arange(bbox[d][0], bbox[d][1] + spacing[i], spacing[i]) for i, d in enumerate(dim_names)]

    if mode == "grid":
        mesh = np.meshgrid(*vectors, indexing='xy' if ndim == 2 else 'ij')
        points = np.column_stack([m.flatten() for m in mesh])
    elif mode == "random":
        n_rand = int(np.prod([len(v) for v in vectors]) * 0.5)
        rng = np.random.default_rng()
        points = np.column_stack([
            rng.uniform(bbox[d][0], bbox[d][1], n_rand) for d in dim_names
        ])
    else:
        raise ValueError("Mode must be 'grid' or 'random'")

    if shape == "ellipse":
        center = np.array([(bbox[d][0] + bbox[d][1]) / 2 for d in dim_names])
        radii = np.array([(bbox[d][1] - bbox[d][0]) / 2 for d in dim_names])
        deltas = (points - center) / radii
        inside = np.sum(deltas**2, axis=1) <= 1.0 + tol
        points = points[inside]

    # existing_tree = existing_nodes.view([('', existing_nodes.dtype)] * ndim)
    # new_tree = points.view([('', points.dtype)] * ndim)
    # unique_mask = ~np.in1d(new_tree, existing_tree, assume_unique=False)
    # new_simplex_nodes = points[unique_mask]

    # Round points to eliminate floating-point noise
    decimals = int(-np.log10(tol))
    existing_set = set(map(tuple, np.round(existing_nodes, decimals)))
    points_rounded = np.round(points, decimals)

    # Filter only truly new points
    new_simplex_nodes = np.array([
        pt for pt in points_rounded if tuple(pt) not in existing_set
    ])


    return new_simplex_nodes


def build_simplex_extrusion_nd(mesh_specs, roi, nodes=None):
    dim_names = list(mesh_specs['element_numbers'].keys())
    simplex_dims = mesh_specs.get('tri_elements', [])
    stack_dims = [d for d in dim_names if d not in simplex_dims]

    ndim_simplex = len(simplex_dims)
    ndim_stack = len(stack_dims)

    if ndim_simplex not in [0, 2, 3]:
        raise ValueError("Only 2D or 3D simplex bases supported.")

    # === Step 1: Build or re-mesh base simplex mesh ===
    if nodes is None:
        # Generate structured grid points for base simplex mesh
        coord_vectors = {}
        for d in simplex_dims:
            spec = mesh_specs['element_numbers'][d]
            distribution = mesh_specs.get('mesh_distribution', {}).get(
                d, {"type": "power", "param": 1.0}
            )
            if isinstance(spec, int):
                bounds = roi[d]
                if distribution["type"] == "boundratio":
                    vec = boundratiovec(bounds[0], bounds[1], spec + 1, distribution["param"])
                elif distribution["type"] == "power":
                    vec = quadraticvec(bounds[0], bounds[1], spec + 1, distribution["param"])
                else:
                    raise ValueError(f"Unknown mesh distribution type '{distribution['type']}'")
            else:
                vec = np.array(spec)
            coord_vectors[d] = vec

        base_grids = np.meshgrid(*[coord_vectors[d] for d in simplex_dims], indexing='xy')
        simplex_nodes = np.column_stack([g.flatten() for g in base_grids])
    else:
        # User provided nodes for base simplex Delaunay triangulation
        simplex_nodes = nodes[:, [dim_names.index(d) for d in simplex_dims]]

    tri = Delaunay(simplex_nodes)
    conn_simplex = tri.simplices
    nodes_simplex = simplex_nodes

    conn_simplex = ensure_positive_orientation_nd(conn_simplex, nodes_simplex)
    conn_simplex = enforce_checkerboard_diagonals(conn_simplex, nodes_simplex)

    # === Step 2: Extrude into stack dimensions ===
    all_nodes = nodes_simplex
    all_conn = conn_simplex

    for d in stack_dims:
        spec = mesh_specs['element_numbers'][d]
        distribution = mesh_specs.get('mesh_distribution', {}).get(
            d, {"type": "power", "param": 1.0}
        )
        if isinstance(spec, int):
            bounds = roi[d]
            if distribution["type"] == "boundratio":
                vec = boundratiovec(bounds[0], bounds[1], spec + 1, distribution["param"])
            elif distribution["type"] == "power":
                vec = quadraticvec(bounds[0], bounds[1], spec + 1, distribution["param"])
            else:
                raise ValueError(f"Unknown mesh distribution type '{distribution['type']}'")
        else:
            vec = np.array(spec)

        new_all_nodes = []
        new_all_conn = []

        n_layers = len(vec)
        n_nodes_per_layer = all_nodes.shape[0]
        n_base_elems = all_conn.shape[0]

        for layer in range(n_layers):
            coord = vec[layer]
            new_col = np.full((n_nodes_per_layer, 1), coord)
            layer_nodes = np.hstack([all_nodes, new_col])
            new_all_nodes.append(layer_nodes)

        all_nodes = np.vstack(new_all_nodes)

        for layer in range(n_layers - 1):
            offset_lower = layer * n_nodes_per_layer
            offset_upper = (layer + 1) * n_nodes_per_layer
            for e in range(n_base_elems):
                conn_lower = all_conn[e] + offset_lower
                conn_upper = all_conn[e] + offset_upper
                conn_prism = np.concatenate([conn_lower, conn_upper])
                new_all_conn.append(conn_prism)

        all_conn = np.array(new_all_conn, dtype=all_conn.dtype)

    return all_nodes, all_conn


def extrude_premade_simplex_mesh_to_nd(mesh_specs, roi):
    """
    Extrude a 2D premade simplex mesh stored in mesh_specs['premade_mesh'] into higher dimensions.

    Parameters
    ----------
    mesh_specs : dict
        Must contain:
            - 'premade_mesh': dict with 'nodes' and 'conn' from cleaned 2D simplex mesh
            - 'element_numbers': full ND mesh specification
            - 'mesh_distribution': optional distribution info for stacking dimensions
            - 'tri_elements': list of 2D simplex dimension names
    roi : dict
        Bounds for each variable in the full ND domain.

    Returns
    -------
    all_nodes : ndarray
        Extruded nodal coordinates in full dimension space.
    all_conn : ndarray
        Extruded connectivity array with stacked prism topology.
    """
    dim_names = list(mesh_specs['element_numbers'].keys())
    simplex_dims = mesh_specs.get('tri_elements', [])
    stack_dims = [d for d in dim_names if d not in simplex_dims]

    assert len(simplex_dims) == 2, "Only 2D simplex base supported for extrusion."

    # --- Load the premade mesh ---
    premade_nodes = np.array(mesh_specs['premade_mesh']['nodes'])
    premade_conn = np.array(mesh_specs['premade_mesh']['conn'])

    # --- Step 1: Align 2D premade nodes into full ND dimension space ---
    all_nodes = premade_nodes
    all_conn = premade_conn

    # --- Step 2: Extrude along remaining stacking dimensions ---
    for d in stack_dims:
        spec = mesh_specs['element_numbers'][d]
        distribution = mesh_specs.get('mesh_distribution', {}).get(
            d, {"type": "power", "param": 1.0}
        )
        if isinstance(spec, int):
            bounds = roi[d]
            if distribution["type"] == "boundratio":
                vec = boundratiovec(bounds[0], bounds[1], spec + 1, distribution["param"])
            elif distribution["type"] == "power":
                vec = quadraticvec(bounds[0], bounds[1], spec + 1, distribution["param"])
            else:
                raise ValueError(f"Unknown mesh distribution type '{distribution['type']}'")
        else:
            vec = np.array(spec)

        new_all_nodes = []
        new_all_conn = []

        n_layers = len(vec)
        n_nodes_per_layer = all_nodes.shape[0]
        n_base_elems = all_conn.shape[0]

        for layer in range(n_layers):
            coord = vec[layer]
            new_col = np.full((n_nodes_per_layer, 1), coord)
            layer_nodes = np.hstack([all_nodes, new_col])
            new_all_nodes.append(layer_nodes)

        all_nodes = np.vstack(new_all_nodes)

        for layer in range(n_layers - 1):
            offset_lower = layer * n_nodes_per_layer
            offset_upper = (layer + 1) * n_nodes_per_layer
            for e in range(n_base_elems):
                conn_lower = all_conn[e] + offset_lower
                conn_upper = all_conn[e] + offset_upper
                conn_prism = np.concatenate([conn_lower, conn_upper])
                new_all_conn.append(conn_prism)

        all_conn = np.array(new_all_conn, dtype=all_conn.dtype)

    return all_nodes, all_conn

def extrude_premade_hypercube_mesh_to_nd(mesh_specs, roi):
    """
    Extrude a 2D premade hypercube mesh stored in mesh_specs['premade_mesh'] into higher dimensions.

    Parameters
    ----------
    mesh_specs : dict
        Must contain:
            - 'premade_mesh': dict with 'nodes' and 'conn' from 2D hypercube mesh
            - 'element_numbers': full ND mesh specification
        Optional:
            - 'mesh_distribution': optional distribution info per extruded dimension

    roi : dict
        Bounds for each variable in the full ND domain.

    Returns
    -------
    all_nodes : ndarray
        Extruded nodal coordinates in full ND space.
    all_conn : ndarray
        Connectivity array with stacked brick topology.
    """
    dim_names = list(mesh_specs['element_numbers'].keys())

    base_dims = mesh_specs['premade_mesh'].get('base_dims')
    stack_dims = mesh_specs['premade_mesh'].get('stack_dims')

    if base_dims is None:
        # fallback for old files that only ever used the first two dims
        ndim_base = np.array(mesh_specs['premade_mesh']['nodes']).shape[1]
        base_dims = dim_names[:ndim_base]
        stack_dims = [d for d in dim_names if d not in base_dims]

    # --- Load base mesh ---
    base_nodes = np.array(mesh_specs['premade_mesh']['nodes'])  # (Nnodes, ndim_base)
    base_conn = np.array(mesh_specs['premade_mesh']['conn'])    # (Nelements, 2^ndim_base)

    all_nodes = base_nodes
    all_conn = base_conn

    # --- Stack along each higher dimension ---
    for d in stack_dims:
        spec = mesh_specs['element_numbers'][d]
        distribution = mesh_specs.get('mesh_distribution', {}).get(
            d, {"type": "power", "param": 1.0}
        )

        if isinstance(spec, int):
            bounds = roi[d]
            if distribution["type"] == "boundratio":
                vec = boundratiovec(bounds[0], bounds[1], spec + 1, distribution["param"])
            elif distribution["type"] == "power":
                vec = quadraticvec(bounds[0], bounds[1], spec + 1, distribution["param"])
            else:
                raise ValueError(f"Unknown mesh distribution type '{distribution['type']}'")
        else:
            vec = np.array(spec)

        new_all_nodes = []
        new_all_conn = []

        n_layers = len(vec)
        n_nodes_per_layer = all_nodes.shape[0]
        n_base_elems = all_conn.shape[0]

        for layer in range(n_layers):
            coord = vec[layer]
            new_col = np.full((n_nodes_per_layer, 1), coord)
            layer_nodes = np.hstack([all_nodes, new_col])
            new_all_nodes.append(layer_nodes)

        all_nodes = np.vstack(new_all_nodes)

        # OLD
        for layer in range(n_layers - 1):
            offset_lower = layer * n_nodes_per_layer
            offset_upper = (layer + 1) * n_nodes_per_layer
            for e in range(n_base_elems):
                lower = all_conn[e] + offset_lower
                upper = all_conn[e] + offset_upper
                # For 2D quads extruded to 3D bricks, combine 4+4 nodes = 8 node brick
                conn_brick = np.concatenate([lower, upper])
                new_all_conn.append(conn_brick)

        all_conn = np.array(new_all_conn, dtype=all_conn.dtype)

    return all_nodes, all_conn

def enforce_checkerboard_diagonals(conn, nodes, decimals=8, verbose=True):
    """
    Enforces consistent checkerboard diagonals in a structured 2D triangulation.

    Args:
        conn: (Nelements, 3) array of triangle connectivity
        nodes: (Nnodes, 2) array of 2D node coordinates
        decimals: number of decimals to round to when comparing coordinates
        verbose: print summary

    Returns:
        conn_fixed: new connectivity array with consistent diagonals
    """
    if nodes.shape[1] != 2:
        raise ValueError("This checkerboard enforcer only works in 2D.")

    rounded_nodes = np.round(nodes, decimals=decimals)

    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(rounded_nodes)}

    x_vals = np.unique(rounded_nodes[:, 0])
    y_vals = np.unique(rounded_nodes[:, 1])

    conn_fixed = conn.copy()
    total_squares = 0

    for j in range(len(y_vals) - 1):
        for i in range(len(x_vals) - 1):
            x0, x1 = x_vals[i], x_vals[i + 1]
            y0, y1 = y_vals[j], y_vals[j + 1]

            try:
                n0 = coord_to_index[(x0, y0)]
                n1 = coord_to_index[(x1, y0)]
                n2 = coord_to_index[(x0, y1)]
                n3 = coord_to_index[(x1, y1)]
            except KeyError:
                continue

            square_nodes = {n0, n1, n2, n3}

            elems_for_square = [
                iel for iel, tri in enumerate(conn_fixed)
                if len(set(tri) & square_nodes) == 3
            ]

            if len(elems_for_square) != 2:
                continue  # partial square due to refinement

            iel0, iel1 = elems_for_square

            if (i + j) % 2 == 0:
                conn_fixed[iel0] = [n0, n1, n2]
                conn_fixed[iel1] = [n1, n3, n2]
            else:
                conn_fixed[iel0] = [n0, n3, n2]
                conn_fixed[iel1] = [n0, n1, n3]

            total_squares += 1

    if verbose:
        print(f"[Checkerboard] Enforced diagonals in {total_squares} full squares.")

    return conn_fixed

def build_hypercube_mesh(mesh_specs, roi, dim_names=None):
    """
    Build a regular hypercube mesh in arbitrary dimensions, 
    supporting both integer counts and explicit node arrays.

    Args:
        mesh_specs: dict with 'element_numbers' and optional 'mesh_distribution'
        roi: dict {dim_name: [min, max]}
        dim_names: optional list of dimension names. If None, inferred.

    Returns:
        nodes: (Nnode, ndim) array of node coordinates
        conn: (Nelem, 2**ndim) array of connectivity (corners per hypercube)
        dim_names: ordered list of dimension names
    """
    element_numbers = mesh_specs['element_numbers']

    if dim_names is None:
        dim_names = list(element_numbers.keys())

    ndim = len(dim_names)

    # === Create coordinate vectors ===
    coord_vectors = {}
    element_counts = []

    for name in dim_names:
        n = element_numbers[name]
        bounds = roi[name]

        if isinstance(n, int):
            distribution = mesh_specs.get('mesh_distribution', {}).get(
                name, {"type": "power", "param": 1.0}
            )
            if distribution["type"] == "boundratio":
                vec = boundratiovec(bounds[0], bounds[1], n + 1, distribution["param"])
            elif distribution["type"] == "power":
                vec = quadraticvec(bounds[0], bounds[1], n + 1, distribution["param"])
            else:
                raise ValueError(f"Unknown mesh distribution type '{distribution['type']}'")
            element_counts.append(n)
        else:
            vec = np.array(n)
            if vec.ndim != 1 or len(vec) < 2:
                raise ValueError(f"Invalid node vector for dimension {name}: {vec}")
            element_counts.append(len(vec) - 1)
        
        coord_vectors[name] = vec

    # === Create node grid ===
    grids = np.meshgrid(*[coord_vectors[k] for k in dim_names], indexing='ij')
    nodes = np.column_stack([g.ravel(order='F') for g in grids])

    # === Build index grid ===
    index_shapes = [len(coord_vectors[k]) for k in dim_names]
    index_array = np.arange(nodes.shape[0]).reshape(index_shapes, order='F')

    # === Build connectivity ===
    conn_list = []
    elem_ranges = [range(ec) for ec in element_counts]  # element_counts instead of element_numbers directly

    for cell in itertools.product(*elem_ranges):
        # Each cell corner is an offset in [0, 1] in each dimension
        corner_offsets = list(itertools.product([0, 1], repeat=ndim))
        corner_indices = []
        for offset in corner_offsets:
            corner = tuple(c + o for c, o in zip(cell, offset))
            idx = index_array[corner]
            corner_indices.append(idx)
        conn_list.append(corner_indices)

    conn = np.array(conn_list, dtype=int)

    return nodes, conn

def boundratiovec(a, b, N, boundratio):
    """
    create spaced nodes:

    a : roi min
    b : roi max
    N : number of nodes
    boundratio : spatial distribution (1.5 means 1.5 larger elements at the edges)
    """
    N = max([N, 2])
    N = round(N)
    x = np.zeros(N)
    boundratio = abs(boundratio)
    
    if N == 2:
        x = np.array([a, b])
    elif N == 3:
        x = np.array([a, (a+b)/2, b])
    else:
        # total length
        Lt = b - a
        # number of internal nodes
        Ni = N - 2
        # internal element length
        Li = Lt / (2*boundratio + Ni - 1)
        # external element length
        Le = boundratio*Li
        # knot vector
        x[0] = a
        x[1] = a + Le
        for k in range(2, N-1):
            x[k] = x[k-1] + Li
        x[-1] = b
    return x

def quadraticvec(a, b, N, power=2.3):
    N = max([N, 2])
    x = np.zeros(N)

    # Create an array of powers that increase from 0 to (N-1) with the specified power
    powers = np.linspace(0, (N - 1), N) ** power

    # Scale the powers array to fit between a and b
    x = a + (b - a) * (powers - powers[0]) / (powers[-1] - powers[0])

    return x