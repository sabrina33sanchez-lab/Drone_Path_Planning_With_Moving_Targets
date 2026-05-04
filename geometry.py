# geometry.py
"""
Path geometry functions.

Provides tools for computing polyline lengths, initializing drone paths
from cluster head positions, and projecting paths to a fixed length.
"""

import numpy as np


def path_length(uVec, vVec):
    """
    Compute the total polyline length of a path given by coordinates (uVec, vVec).

    The function treats consecutive entries as segment endpoints and returns
    the sum of Euclidean lengths of those segments.

    Parameters
    ----------
    uVec, vVec : array_like
        Sequence of x- and y-coordinates defining the path.

    Returns
    -------
    float
        Total path length (sum of segment lengths).
    """
    du = np.diff(uVec)
    dv = np.diff(vVec)
    return np.sum(np.sqrt(du**2 + dv**2))


def initialize_path_from_cluster_heads(X0_j, Y0_j, L):
    """
    Initialize drone path using scaled cluster-head positions.

    Parameters
    ----------
    X0_j, Y0_j : array_like, shape (J,)
        Initial cluster head positions at t = 0
    L : float
        Desired total path length constraint g(u,v) = L

    Returns
    -------
    uVec, vVec : ndarray, shape (J+2,)
        Initialized drone path (with fixed endpoints at the origin).
    """
    X0_j = np.asarray(X0_j, dtype=float)
    Y0_j = np.asarray(Y0_j, dtype=float)
    J = len(X0_j)

    # --- unscaled initial path ---
    u_init = np.zeros(J + 2)
    v_init = np.zeros(J + 2)

    # interior points: cluster heads
    u_init[1:J+1] = X0_j
    v_init[1:J+1] = Y0_j

    # --- compute scaling ---
    g0 = path_length(u_init, v_init)
    if g0 == 0:
        raise ValueError("Initial path length is zero; cannot scale.")

    alpha = L / g0

    # --- apply scaling (interior only) ---
    uVec = u_init.copy()
    vVec = v_init.copy()

    uVec[1:J+1] *= alpha
    vVec[1:J+1] *= alpha

    return uVec, vVec


def project_to_length(u, v, L):
    """
    Project the interior points of (u,v) so that total path length g(u,v) = L.

    Endpoints are fixed. Handles zero-length paths safely to avoid NaNs.

    Parameters
    ----------
    u, v : ndarray
        Full drone path including fixed endpoints.
    L : float
        Target path length.

    Returns
    -------
    u_proj, v_proj : ndarray
        Projected path with g(u_proj, v_proj) = L.
    """
    u_proj = u.copy()
    v_proj = v.copy()

    # Compute segment differences
    du = np.diff(u)
    dv = np.diff(v)
    ell = np.sqrt(du**2 + dv**2)

    # Prevent divide by zero
    ell = np.maximum(ell, 1e-8)

    g = np.sum(ell)
    if g < 1e-12:
        # Path collapsed, just return original
        return u_proj, v_proj

    scale = L / g

    # Scale interior points relative to fixed endpoints
    u_proj[1:-1] = u[0] + (u[1:-1] - u[0]) * scale
    v_proj[1:-1] = v[0] + (v[1:-1] - v[0]) * scale

    return u_proj, v_proj
