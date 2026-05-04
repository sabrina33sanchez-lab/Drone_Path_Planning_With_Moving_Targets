# comparison.py
"""
Orchestrates and compares three drone path optimizer variants.

The single function test_optimizers runs all three methods on the same
cluster head configuration and returns a unified results dictionary
that can be passed directly to plot_optimizers_paths for visualization.
"""

import time

import numpy as np

from routing import (
    solve_tsp_exact,
    rolling_optimize_indexed_reorder,
    iterative_tsp_predicted,
    iterative_greedy_predicted,
    iterative_tsp_drone_points,
)
from geometry import initialize_path_from_cluster_heads
from metrics import gf_values_moving


def _natural_path_length(X0, Y0, sx, sy, s_d):
    """
    Compute the length of the direct interception path:
      origin -> cluster_1(T_1) -> ... -> cluster_J(T_J) -> origin
    where each T_j is solved self-consistently: the drone flies in a straight
    line at speed s_d and intercepts cluster j (which moves at constant velocity)
    exactly when it arrives.

    For each step, given drone position (px, py) at time T_prev and cluster j
    with initial position (X0_j, Y0_j) and velocity (sx_j, sy_j), the travel
    time tau satisfies the quadratic:
      (s_d^2 - vx^2 - vy^2) * tau^2 + 2*(A*vx + B*vy) * tau - (A^2 + B^2) = 0
    where A = px - X0_j - vx*T_prev, B = py - Y0_j - vy*T_prev.
    """
    J = len(X0)
    T = 0.0
    px, py = 0.0, 0.0
    total_length = 0.0

    for j in range(J):
        vx, vy = sx[j], sy[j]
        A = px - X0[j] - vx * T
        B = py - Y0[j] - vy * T

        a = s_d**2 - vx**2 - vy**2
        b = 2.0 * (A * vx + B * vy)
        c = -(A**2 + B**2)

        if abs(a) < 1e-12:
            tau = -c / b if abs(b) > 1e-12 else 0.0
        else:
            disc = max(b**2 - 4.0 * a * c, 0.0)
            tau = (-b + np.sqrt(disc)) / (2.0 * a)
            if tau < 0:
                tau = (-b - np.sqrt(disc)) / (2.0 * a)

        tau = max(tau, 0.0)
        T += tau
        total_length += s_d * tau
        px = X0[j] + vx * T
        py = Y0[j] + vy * T

    # Return leg to origin
    total_length += np.sqrt(px**2 + py**2)
    return total_length


def _compute_tsp_path_length(X, Y, order):
    """
    Compute the total path length of a TSP tour.
    
    Parameters
    ----------
    X, Y : ndarray, shape (J,)
        Cluster positions.
    order : ndarray, shape (J,)
        Visitation order (indices).
    
    Returns
    -------
    path_length : float
        Total distance traveled in the tour (origin -> clusters -> origin).
    """
    X_ordered = X[order]
    Y_ordered = Y[order]
    
    # Distance from origin (0,0) to first cluster
    dist = np.sqrt(X_ordered[0]**2 + Y_ordered[0]**2)
    
    # Distances between consecutive clusters
    for i in range(len(order) - 1):
        dist += np.sqrt((X_ordered[i+1] - X_ordered[i])**2 + 
                        (Y_ordered[i+1] - Y_ordered[i])**2)
    
    # Distance from last cluster back to origin
    dist += np.sqrt(X_ordered[-1]**2 + Y_ordered[-1]**2)
    
    return dist


def test_optimizers(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    reOrd=None,
    reorder_vec=None,
    dtau=0.001,
    n_steps=500,
    grad_clip=1e2,
    tsp_time_limit=20,
):
    """
    Run and compare three optimizer variants on a shared test case.

    The three methods are:
      1. Original       : one optimization pass, then a single mid-point reorder,
                          then a second optimization pass.
      2. Predicted Rolling : rolling optimization with automatic reordering based
                          on predicted cluster positions (uses dL or reorder_vec).
      3. TSP Rolling    : rolling optimization with reordering at explicit
                          path-length checkpoints from reorder_vec.

    All three start from the same initial time-aware TSP ordering of clusters.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d : float
        Drone speed.
    p : float
        Path-loss exponent in the objective.
    L : float
        Target drone path length constraint.
    reorder_vec : array_like, optional
        Path-length checkpoints for TSP reordering. Defaults to two evenly
        spaced points within [0, L].
    dL : float
        Reorder spacing for the legacy 'Predicted Rolling' mode.
    dtau : float
        Pseudo-time step size for the optimizer.
    n_steps : int
        Number of pseudo-time steps for the 'Original' method.
    grad_clip : float
        Gradient clipping threshold.

    Returns
    -------
    results : dict
        Dictionary with keys 'Original', 'Predicted Rolling', 'TSP Rolling'.
        Each entry is a dict containing:
          'g'        : float  - path length
          'f'        : float  - objective value
          'E'        : float  - total communication energy
          'reorders' : int    - number of reorder events
          'u', 'v'   : ndarray - final drone path
          'X', 'Y'   : ndarray - final cluster positions (at t=0)
          'sx', 'sy' : ndarray - cluster velocities
    """
    print("\n=== TEST CASE ===")
    N_clusters = len(X0_j)
    cluster_ids = np.arange(N_clusters)

    # ------------------------------------------------------------------
    # Step 0: Initial time-aware TSP ordering
    # Predict where each cluster will be after time L/(2*s_d),
    # then sort by TSP on those predicted positions.
    # ------------------------------------------------------------------
    t_predict = L / (2 * s_d)

    X_pred = X0_j + s_jx * t_predict
    Y_pred = Y0_j + s_jy * t_predict

    order_tsp = solve_tsp_exact(X_pred, Y_pred, tsp_time_limit=tsp_time_limit)

    # Compute initial TSP path length using the actual cluster positions (before reordering)
    initial_tsp_length = _compute_tsp_path_length(X0_j, Y0_j, order_tsp)

    X0_j = X0_j[order_tsp]
    Y0_j = Y0_j[order_tsp]
    s_jx = s_jx[order_tsp]
    s_jy = s_jy[order_tsp]
    cluster_ids = cluster_ids[order_tsp]

    print("\nInitial TSP order:", cluster_ids)

    # Compute objective value for the initial ordering
    u_init, v_init = initialize_path_from_cluster_heads(X0_j, Y0_j, L)
    g_init, f_init = gf_values_moving(u_init, v_init, X0_j, Y0_j, s_jx, s_jy, s_d, p)
    print(f"Initial ordering objective: f = {f_init:.4f}")

    # Default reorder schedule for Method 2/3
    J = len(X0_j)
    if reOrd is None:
        reOrd = list(range(J // 3 - 1, J - 1, max(1, J // 3)))

    if reorder_vec is None:
        reorder_vec = np.linspace(0, L, 4)[1:-1]

    results = {}

    # Decide which orientation (forward vs reversed) should be labeled
    # without the "Rev" suffix.  The requirement is that the method
    # called "FinalXY" corresponds to the ordering whose first path
    # segment (origin → first cluster) is smaller, and "FinalXY Rev" gets
    # the larger-first-distance ordering.  The first clusterhead location
    # passed to the TSP solver is given by the predicted positions
    # `X_pred, Y_pred` sorted by `order_tsp`, so we compute
    # those two distances and sort accordingly.
    #
    # `order_tsp[0]` is the first cluster in the forward order; the reverse
    # starts from `order_tsp[-1]`.
    first_pred_forward = np.hypot(
        X_pred[order_tsp[0]],
        Y_pred[order_tsp[0]]
    )
    first_pred_rev = np.hypot(
        X_pred[order_tsp[-1]],
        Y_pred[order_tsp[-1]]
    )

    # Build a list of (suffix, flip_flag, distance) and sort by distance.
    suffix_list = [('', False, first_pred_forward),
                   (' Rev', True, first_pred_rev)]
    suffix_list.sort(key=lambda tpl: tpl[2])

    # Run all three methods on both the forward and reversed initial ordering,
    # iterating in the order determined above.
    for suffix, flip, _dist in suffix_list:
        if flip:
            X_ord  = X0_j[::-1].copy()
            Y_ord  = Y0_j[::-1].copy()
            sx_ord = s_jx[::-1].copy()
            sy_ord = s_jy[::-1].copy()
        else:
            X_ord  = X0_j.copy()
            Y_ord  = Y0_j.copy()
            sx_ord = s_jx.copy()
            sy_ord = s_jy.copy()

        # --------------------------------------------------------------
        # Method 1: Index-triggered rolling reorder (TSP)
        # --------------------------------------------------------------
        _t0 = time.perf_counter()
        (u_final_pred, v_final_pred, traj_all_pred,
         X_final_pred, Y_final_pred, sx_final_pred, sy_final_pred,
         n_reorders_pred) = rolling_optimize_indexed_reorder(
            X_ord.copy(), Y_ord.copy(),
            sx_ord.copy(), sy_ord.copy(),
            s_d, p, L,
            reOrd=reOrd,
            dtau=dtau,
            n_steps=n_steps,
            grad_clip=grad_clip,
            tsp_time_limit=tsp_time_limit,
        )

        g_val, f_val = gf_values_moving(
            u_final_pred, v_final_pred,
            X_final_pred, Y_final_pred, sx_final_pred, sy_final_pred,
            s_d, p
        )
        if f_val < 1000:
            if _natural_path_length(X_final_pred, Y_final_pred, sx_final_pred, sy_final_pred, s_d) <= 1.001 * L:
                f_val = 0.0

        results[f'Predicted Rolling{suffix}'] = {
            'g': g_val, 'f': f_val, 'E': f_val, 'reorders': n_reorders_pred,
            'time': time.perf_counter() - _t0,
            'u': u_final_pred, 'v': v_final_pred,
            'X': X_final_pred, 'Y': Y_final_pred,
            'sx': sx_final_pred, 'sy': sy_final_pred
        }

        # --------------------------------------------------------------
        # Method 3: Index-triggered rolling reorder (Greedy)
        # --------------------------------------------------------------
        _t0 = time.perf_counter()
        (u_final_greedy, v_final_greedy, traj_all_greedy,
         X_final_greedy, Y_final_greedy, sx_final_greedy, sy_final_greedy,
         n_reorders_greedy) = rolling_optimize_indexed_reorder(
            X_ord.copy(), Y_ord.copy(),
            sx_ord.copy(), sy_ord.copy(),
            s_d, p, L,
            reOrd=reOrd,
            dtau=dtau,
            n_steps=n_steps,
            grad_clip=grad_clip,
            use_greedy=True
        )

        g_val, f_val = gf_values_moving(
            u_final_greedy, v_final_greedy,
            X_final_greedy, Y_final_greedy, sx_final_greedy, sy_final_greedy,
            s_d, p
        )
        if f_val < 1000:
            if _natural_path_length(X_final_greedy, Y_final_greedy, sx_final_greedy, sy_final_greedy, s_d) <= 1.001 * L:
                f_val = 0.0

        results[f'Greedy Rolling{suffix}'] = {
            'g': g_val, 'f': f_val, 'E': f_val, 'reorders': n_reorders_greedy,
            'time': time.perf_counter() - _t0,
            'u': u_final_greedy, 'v': v_final_greedy,
            'X': X_final_greedy, 'Y': Y_final_greedy,
            'sx': sx_final_greedy, 'sy': sy_final_greedy
        }

        # --------------------------------------------------------------
        # Method 3: FinalXY - iterative solve + TSP on predicted positions
        # --------------------------------------------------------------
        _t0 = time.perf_counter()
        (u_final_x1, v_final_x1, traj_x1,
         X_final_x1, Y_final_x1, sx_final_x1, sy_final_x1,
         n_reorders_x1) = iterative_tsp_predicted(
            X_ord.copy(), Y_ord.copy(),
            sx_ord.copy(), sy_ord.copy(),
            s_d, p, L,
            dtau=dtau,
            n_steps=n_steps,
            grad_clip=grad_clip,
            tsp_time_limit=tsp_time_limit,
        )

        g_val, f_val = gf_values_moving(
            u_final_x1, v_final_x1,
            X_final_x1, Y_final_x1, sx_final_x1, sy_final_x1,
            s_d, p
        )
        if f_val < 1000:
            if _natural_path_length(X_final_x1, Y_final_x1, sx_final_x1, sy_final_x1, s_d) <= 1.001 * L:
                f_val = 0.0

        results[f'FinalXY{suffix}'] = {
            'g': g_val, 'f': f_val, 'E': f_val, 'reorders': n_reorders_x1,
            'time': time.perf_counter() - _t0,
            'u': u_final_x1, 'v': v_final_x1,
            'X': X_final_x1, 'Y': Y_final_x1,
            'sx': sx_final_x1, 'sy': sy_final_x1
        }

        # --------------------------------------------------------------
        # Method 4: FinalXY_Greed - iterative solve + greedy on predicted positions
        # --------------------------------------------------------------
        _t0 = time.perf_counter()
        (u_final_gx1, v_final_gx1, traj_gx1,
         X_final_gx1, Y_final_gx1, sx_final_gx1, sy_final_gx1,
         n_reorders_gx1) = iterative_greedy_predicted(
            X_ord.copy(), Y_ord.copy(),
            sx_ord.copy(), sy_ord.copy(),
            s_d, p, L,
            dtau=dtau,
            n_steps=n_steps,
            grad_clip=grad_clip
        )

        g_val, f_val = gf_values_moving(
            u_final_gx1, v_final_gx1,
            X_final_gx1, Y_final_gx1, sx_final_gx1, sy_final_gx1,
            s_d, p
        )
        if f_val < 1000:
            if _natural_path_length(X_final_gx1, Y_final_gx1, sx_final_gx1, sy_final_gx1, s_d) <= 1.001 * L:
                f_val = 0.0

        results[f'FinalXY_Greed{suffix}'] = {
            'g': g_val, 'f': f_val, 'E': f_val, 'reorders': n_reorders_gx1,
            'time': time.perf_counter() - _t0,
            'u': u_final_gx1, 'v': v_final_gx1,
            'X': X_final_gx1, 'Y': Y_final_gx1,
            'sx': sx_final_gx1, 'sy': sy_final_gx1
        }

        # --------------------------------------------------------------
        # Method 5: FinalUV - iterative solve + TSP on drone reception points
        # --------------------------------------------------------------
        _t0 = time.perf_counter()
        (u_final_x2, v_final_x2, traj_x2,
         X_final_x2, Y_final_x2, sx_final_x2, sy_final_x2,
         n_reorders_x2) = iterative_tsp_drone_points(
            X_ord.copy(), Y_ord.copy(),
            sx_ord.copy(), sy_ord.copy(),
            s_d, p, L,
            dtau=dtau,
            n_steps=n_steps,
            grad_clip=grad_clip,
            tsp_time_limit=tsp_time_limit,
        )

        g_val, f_val = gf_values_moving(
            u_final_x2, v_final_x2,
            X_final_x2, Y_final_x2, sx_final_x2, sy_final_x2,
            s_d, p
        )
        if f_val < 1000:
            if _natural_path_length(X_final_x2, Y_final_x2, sx_final_x2, sy_final_x2, s_d) <= 1.001 * L:
                f_val = 0.0

        results[f'FinalUV{suffix}'] = {
            'g': g_val, 'f': f_val, 'E': f_val, 'reorders': n_reorders_x2,
            'time': time.perf_counter() - _t0,
            'u': u_final_x2, 'v': v_final_x2,
            'X': X_final_x2, 'Y': Y_final_x2,
            'sx': sx_final_x2, 'sy': sy_final_x2
        }

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== RESULTS COMPARISON ===")
    print(f"{'Method':25s} {'g':>6s} {'f':>8s} {'% Improv':>10s} {'#Reorders':>10s} {'Time(s)':>9s}")
    for key, val in results.items():
        pct = (f_init - val['f']) / f_init * 100 if f_init != 0 else 0.0
        print(f"{key:25s} {val['g']:6.3f} {val['f']:8.3f} {pct:10.2f} "
              f"{val['reorders']:10d} {val['time']:9.3f}")

    return results, initial_tsp_length, f_init
