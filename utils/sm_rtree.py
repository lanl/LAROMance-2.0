#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================
@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

R-tree spatial indexing for finite element mesh lookups.

This module provides efficient spatial indexing for element lookup in finite element meshes
using R-tree data structures, significantly speeding up point-in-element queries in
higher-dimensional spaces.

"""

import numpy as np
import sys

from utils import fe_shapes

# Check if rtree package is available
try:
    from rtree import index
    RTREE_AVAILABLE = True
    print("R-tree available.")
except ImportError:
    RTREE_AVAILABLE = False
    print("\nWarning: R-tree indexing not available. Install rtree package for faster element lookup.")
    sys.stdout.flush()

class MeshRtreeIndex:
    """
    R-tree spatial index for finite element meshes.
    
    This class provides a spatial index for quickly finding which element
    contains a given point in a finite element mesh, which is much faster
    than brute force checking every element, especially in higher dimensions.
    """
    
    def __init__(self, nodes, conn, mesh_specs):
        """
        Initialize the R-tree index for a mesh.
        
        Parameters:
        -----------
        nodes : ndarray, shape (num_nodes, D)
            Nodal coordinates in D-dimensional space.
        conn : ndarray, shape (num_elements, N)
            Element connectivity matrix (indices of nodes per element).
        mesh_specs : dict
            Mesh specifications including element types, etc.
        """
        self.idx = None
        self.element_bbox_cache = None
        self.num_elements = conn.shape[0]
        self.dimension = nodes.shape[1]
        self.mesh_specs = mesh_specs
        
        # Build the R-tree index
        self._build_index(nodes, conn)
        
    def _build_index(self, nodes, conn):
        """
        Build the R-tree spatial index for the mesh.
        
        Parameters:
        -----------
        nodes : ndarray, shape (num_nodes, D)
            Nodal coordinates in D-dimensional space.
        conn : ndarray, shape (num_elements, N)
            Element connectivity matrix (indices of nodes per element).
        """
        # Create a property based index with simplified properties
        p = index.Property()
        p.dimension = self.dimension
        
        # Create the R-tree index with default properties
        # Avoid custom settings that might cause compatibility issues
        self.idx = index.Index(properties=p)
        
        # Create bounding box cache for each element
        self.element_bbox_cache = []
        
        # Insert each element into the R-tree
        for i in range(self.num_elements):
            elem_nodes = nodes[conn[i]]
            
            # Get the bounding box of the element
            bbox = self._element_bbox(elem_nodes)
            
            # Store the bounding box for later use
            self.element_bbox_cache.append(bbox)
            
            # Insert the element into the R-tree with the element index as the ID
            self.idx.insert(i, bbox)
    
    def _element_bbox(self, elem_nodes):
        """
        Calculate the bounding box of an element.
        
        Parameters:
        -----------
        elem_nodes : ndarray, shape (N, D)
            Coordinates of nodes in this element.
        
        Returns:
        --------
        bbox : tuple
            (min_x, min_y, ..., max_x, max_y, ...)
        """
        # Get min/max coordinates in each dimension
        mins = np.min(elem_nodes, axis=0)
        maxs = np.max(elem_nodes, axis=0)
        
        # Format as (min_x, min_y, ..., max_x, max_y, ...)
        bbox = tuple(mins) + tuple(maxs)
        return bbox
    
    def query_point(self, point, nodes, conn):
        """
        Find the element containing a point.
        
        Parameters:
        -----------
        point : ndarray, shape (D,)
            The point to locate.
        nodes : ndarray, shape (num_nodes, D)
            Nodal coordinates.
        conn : ndarray, shape (num_elements, N)
            Element connectivity.
        
        Returns:
        --------
        element_idx : int or False
            Index of the element containing the point, or False if not found.
        """
        # Query the R-tree for elements whose bounding box contains the point
        # We pass the point twice to create a zero-volume bounding box
        candidates = list(self.idx.intersection(tuple(point) + tuple(point)))
        
        # Check each candidate element precisely
        for elem_idx in candidates:
            node_ids = conn[elem_idx]
            elem_nodes = nodes[node_ids]
            
            # Check if the point is inside this element
            if fe_shapes.point_in_element_mixed(point, elem_nodes, self.mesh_specs):
                return elem_idx
        
        # No matching element found
        return False

    def to_dict(self):
        """
        Convert the R-tree index data to a dictionary for serialization.
        
        Returns:
        --------
        dict
            A dictionary containing the serializable data for the R-tree index.
        """
        return {
            'element_bbox_cache': self.element_bbox_cache,
            'num_elements': self.num_elements,
            'dimension': self.dimension
        }
    
    @classmethod
    def from_dict(cls, data, nodes, conn, mesh_specs):
        """
        Reconstruct an R-tree index from serialized data.
        
        Parameters:
        -----------
        data : dict
            Dictionary with serialized R-tree index data.
        nodes : ndarray, shape (num_nodes, D)
            Nodal coordinates.
        conn : ndarray, shape (num_elements, N)
            Element connectivity.
        mesh_specs : dict
            Mesh specifications.
        
        Returns:
        --------
        MeshRtreeIndex
            The reconstructed R-tree index.
        """
        # Create empty instance
        instance = cls.__new__(cls)
        
        # Set attributes from data
        instance.element_bbox_cache = data['element_bbox_cache']
        instance.num_elements = data['num_elements']
        instance.dimension = data['dimension']
        instance.mesh_specs = mesh_specs
        
        # Rebuild the R-tree index
        p = index.Property()
        p.dimension = instance.dimension
        
        # Create the R-tree index with default properties
        # Avoid custom settings that might cause compatibility issues
        instance.idx = index.Index(properties=p)
        
        # Insert elements into the index
        for i, bbox in enumerate(instance.element_bbox_cache):
            instance.idx.insert(i, bbox)
        
        return instance

def create_mesh_rtree_index(nodes, conn, mesh_specs):
    """
    Create an R-tree spatial index for a finite element mesh.
    
    Parameters:
    -----------
    nodes : ndarray, shape (num_nodes, D)
        Nodal coordinates in D-dimensional space.
    conn : ndarray, shape (num_elements, N)
        Element connectivity matrix (indices of nodes per element).
    mesh_specs : dict
        Mesh specifications including element types, etc.
    
    Returns:
    --------
    rtree_index : MeshRtreeIndex
        R-tree spatial index for the mesh.
    """
    return MeshRtreeIndex(nodes, conn, mesh_specs)

def find_element_rtree(point, nodes, conn, mesh_specs, rtree_index, verbose=False):
    """
    Find the element containing a point using R-tree spatial indexing.
    
    Parameters:
    -----------
    point : ndarray, shape (D,)
        D-dimensional input point to locate in the mesh.
    nodes : ndarray, shape (num_nodes, D)
        Nodal coordinates in D-dimensional space.
    conn : ndarray, shape (num_elements, N)
        Element connectivity matrix (indices of nodes per element).
    mesh_specs : dict
        Mesh specifications including element types, etc.
    rtree_index : MeshRtreeIndex
        Pre-built R-tree spatial index for the mesh.
    verbose : bool
        Whether to print detailed information.
    
    Returns:
    --------
    node_range : list or False
        Min/max range per dimension for the matching element, or False if not found.
    element_idx : int or False
        Index of the matching element, or False if not found.
    """
    # Query the R-tree index to find the element containing the point
    element_idx = rtree_index.query_point(point, nodes, conn)
    
    if element_idx is False:
        if verbose:
            print(f"WARNING: No matching element found for point {point}")
            sys.stdout.flush()
        return False, False
    
    # Get the element's node range
    elem_nodes = nodes[conn[element_idx]]
    min_vals = np.min(elem_nodes, axis=0)
    max_vals = np.max(elem_nodes, axis=0)
    node_range = [v for pair in zip(min_vals, max_vals) for v in pair]
    
    return node_range, element_idx