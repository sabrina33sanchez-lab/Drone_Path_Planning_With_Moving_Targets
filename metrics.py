# metrics.py
"""
Energy and objective-function metrics.

Provides functions to evaluate how well a drone path serves the cluster heads,
measured by the total communication energy (sum of (distance)^p terms).
"""

import numpy as np


def total_clusterhead_energy(p, u_path, v_path, X0_j, Y0_j, sx_j, sy_j, s_d):
    """
    Compute total communication energy for all cluster heads.

    For each cluster head j, the energy contribution is:
        dist(drone_point_j, cluster_position_at_arrival_j) ** p

    Arrival times are computed from the cumulative drone path distance
    divided by the drone speed s_d.

    Parameters
    ----------
    p : float
        Path-loss exponent.
    u_path, v_path : ndarray
        Full drone path coordinates (including fixed endpoints).
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    sx_j, sy_j : ndarray, shape (J,)
        Cluster head velocities.
    s_d : float
        Drone speed.

    Returns
    -------
    E : float
        Total energy (sum of dist**p over all clusters).
    """
    J = len(X0_j)

    du = np.diff(u_path)
    dv = np.diff(v_path)
    segment_lengths = np.sqrt(du**2 + dv**2)
    cumulative_dist = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Arrival time at each interior node (skip the start endpoint)
    t_arrival = cumulative_dist[1:J+1] / s_d

    E = 0.0
    for j in range(J):
        x_drone = u_path[j+1]
        y_drone = v_path[j+1]

        x_ch = X0_j[j] + sx_j[j] * t_arrival[j]
        y_ch = Y0_j[j] + sy_j[j] * t_arrival[j]

        dist = np.sqrt((x_drone - x_ch)**2 + (y_drone - y_ch)**2)
        E += dist**p

    return E


def gf_values(u, v, u0, v0, X_j, Y_j, p):
    """
    Compute path length g and objective f for a static cluster arrangement.

    Parameters
    ----------
    u, v : ndarray
        Interior drone path coordinates (endpoints excluded).
    u0, v0 : float
        Starting (and ending) drone position.
    X_j, Y_j : ndarray, shape (J,)
        Cluster head positions (fixed, not moving).
    p : float
        Objective exponent.

    Returns
    -------
    g : float
        Total drone path length.
    f : float
        Objective function value (sum of dist**p to cluster heads).
    """
    u_full = np.concatenate([[u0], u, [u0]])
    v_full = np.concatenate([[v0], v, [v0]])
    du = np.diff(u_full)
    dv = np.diff(v_full)
    g = np.sum(np.sqrt(du**2 + dv**2))

    a = u - X_j
    b = v - Y_j
    f = np.sum((a**2 + b**2)**(p/2))

    return g, f


def gf_values_moving(u_path, v_path, X0_j, Y0_j, sx_j, sy_j, s_d, p):
    """
    Compute path length g and objective f for moving cluster heads.

    Like gf_values, but cluster positions are evaluated at the drone's
    arrival time at each interior node.

    Parameters
    ----------
    u_path, v_path : ndarray
        Full drone path coordinates (including fixed endpoints).
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    sx_j, sy_j : ndarray, shape (J,)
        Cluster head velocities.
    s_d : float
        Drone speed.
    p : float
        Objective exponent.

    Returns
    -------
    g : float
        Total drone path length.
    f : float
        Objective function value accounting for cluster head motion.
    """
    du = np.diff(u_path)
    dv = np.diff(v_path)
    segment_lengths = np.sqrt(du**2 + dv**2)
    g = np.sum(segment_lengths)

    cumulative_dist = np.concatenate(([0], np.cumsum(segment_lengths)))
    J = len(X0_j)
    t_arrival = cumulative_dist[1:J+1] / s_d

    f = 0.0
    for j in range(J):
        x_drone = u_path[j+1]
        y_drone = v_path[j+1]

        x_ch = X0_j[j] + sx_j[j] * t_arrival[j]
        y_ch = Y0_j[j] + sy_j[j] * t_arrival[j]

        dist = np.sqrt((x_drone - x_ch)**2 + (y_drone - y_ch)**2)
        f += dist**p

    return g, f
