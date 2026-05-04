# routing.py
"""
TSP-based routing and rolling optimization algorithms.

This module provides:
  - solve_tsp_exact                 : solve the Traveling Salesman Problem via OR-Tools
  - reorder_by_predicted_positions  : greedy reorder using predicted cluster positions
  - rolling_optimize_with_reorder   : rolling optimizer with flexible reorder schedule
  - rolling_optimize_tsp            : rolling optimizer with explicit TSP reorder schedule
  - rolling_optimize_indexed_reorder: rolling optimizer with reorders at specific
                                      cluster-visit indices (TSP or greedy)
  - iterative_tsp_predicted         : iteratively solve + TSP-reorder predicted cluster
                                      positions until the ordering stabilises (FinalX1)
  - iterative_greedy_predicted      : same as FinalX1 but uses greedy nearest-neighbor
                                      reordering instead of TSP (GreedyX1)
  - iterative_tsp_drone_points      : iteratively solve + TSP-reorder drone reception
                                      points until the ordering stabilises (FinalX2)
"""

import time
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from geometry import initialize_path_from_cluster_heads
from optimization import solve_ivp_fixed_order
from metrics import gf_values_moving


# ---------------------------------------------------------------------------
# TSP Solver
# ---------------------------------------------------------------------------

def solve_tsp_exact(X, Y, tsp_time_limit=20, initial_order=None):
    """
    Solve the Traveling Salesman Problem on points (X, Y) using OR-Tools.

    Returns the visitation order as an index array so that
    X[order], Y[order] gives the TSP tour starting from index 0.

    Parameters
    ----------
    X, Y : array_like, shape (J,)
        Cluster head positions.
    tsp_time_limit : int
        Time limit in seconds for the OR-Tools guided local search.
    initial_order : array_like of int, optional
        Warm-start route as a sequence of node indices. If provided, the
        solver begins local search from this solution instead of building
        one from scratch.

    Returns
    -------
    order : ndarray, shape (J,)
        Indices of cluster heads in tour order.
    """
    J = len(X)

    dist_matrix = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            dist_matrix[i, j] = np.sqrt((X[i]-X[j])**2 + (Y[i]-Y[j])**2)

    manager = pywrapcp.RoutingIndexManager(J, 1, 0)  # 1 vehicle, starts at index 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(i, j):
        return int(dist_matrix[manager.IndexToNode(i), manager.IndexToNode(j)] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    TSPsec = 0.15  # stop if no improvement found within this time window

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    # Run TSP in time slices of TSPsec; stop when a full slice yields no improvement
    t0 = time.perf_counter()
    improvement_log = []
    best_cost = [float('inf')]
    solution = [None]

    def on_solution():
        cost = routing.CostVar().Value()
        if cost < best_cost[0]:
            best_cost[0] = cost
            elapsed = time.perf_counter() - t0
            improvement_log.append((elapsed, cost))

    routing.AddAtSolutionCallback(on_solution)

    elapsed_total = 0.0
    last_improvement_count = 0
    current_solution = None

    # Warm start: seed the search from initial_order if provided
    if initial_order is not None:
        route = [int(v) for v in initial_order if v != 0]
        initial_assignment = routing.ReadAssignmentFromRoutes([route], True)
        if initial_assignment is not None:
            search_parameters.time_limit.FromMilliseconds(int(TSPsec * 1000))
            current_solution = routing.SolveFromAssignmentWithParameters(
                initial_assignment, search_parameters)
            elapsed_total += TSPsec
            last_improvement_count = len(improvement_log)

    while elapsed_total < tsp_time_limit:
        search_parameters.time_limit.FromMilliseconds(int(TSPsec * 1000))
        if current_solution is None:
            current_solution = routing.SolveWithParameters(search_parameters)
            if current_solution is None:
                # print(f"  TSP: no solution in {TSPsec}s, retrying with 1s limit.")
                search_parameters.time_limit.FromMilliseconds(1000)
                current_solution = routing.SolveWithParameters(search_parameters)
                elapsed_total += 1.0
                last_improvement_count = len(improvement_log)
                if current_solution is None:
                    # print(f"  TSP: no solution in 1s, retrying with 5s limit.")
                    search_parameters.time_limit.FromMilliseconds(5000)
                    current_solution = routing.SolveWithParameters(search_parameters)
                    elapsed_total += 5.0
                    last_improvement_count = len(improvement_log)
                    if current_solution is None:
                        # print(f"  TSP: no solution in 5s, falling back to greedy.")
                        greedy_order = _greedy_nearest(0.0, 0.0, X, Y)
                        return greedy_order
                continue
        else:
            current_solution = routing.SolveFromAssignmentWithParameters(
                current_solution, search_parameters)
        elapsed_total += TSPsec

        # Stop if no new improvement in this slice
        if len(improvement_log) == last_improvement_count:
            break
        last_improvement_count = len(improvement_log)

    solution = current_solution

    # if improvement_log:
    #     print(f"  TSP: {len(improvement_log)} improvements, "
    #           f"last at {improvement_log[-1][0]:.3f}s, "
    #           f"total {elapsed_total:.2f}s / {tsp_time_limit}s limit")

    if solution:
        order = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            order.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return np.array(order)
    else:
        raise ValueError("TSP solver failed to find a solution even with 1s retry.")


# ---------------------------------------------------------------------------
# Greedy Reordering
# ---------------------------------------------------------------------------

def reorder_by_predicted_positions(u, v, X0, Y0, sx, sy, g_arrival, s_d, cluster_ids):
    """
    Reorder interior path points and cluster heads using a greedy nearest-neighbor
    heuristic applied to predicted cluster positions at their arrival times.

    Parameters
    ----------
    u, v : ndarray
        Current drone path (including fixed endpoints).
    X0, Y0 : ndarray, shape (J,)
        Initial cluster head positions.
    sx, sy : ndarray, shape (J,)
        Cluster head velocities.
    g_arrival : ndarray, shape (J,)
        Cumulative path distances at each interior node.
    s_d : float
        Drone speed (used to convert distance to time).
    cluster_ids : ndarray, shape (J,)
        Cluster identifiers (reordered consistently with everything else).

    Returns
    -------
    u_new, v_new : ndarray
        Reordered drone path.
    X0_new, Y0_new, sx_new, sy_new : ndarray
        Reordered cluster data.
    cluster_ids_new : ndarray
        Reordered cluster IDs.
    order : ndarray
        Permutation array applied to interior points and cluster arrays.
    """
    J = len(X0)

    t_arrival = g_arrival / s_d
    X_pred = X0 + sx * t_arrival
    Y_pred = Y0 + sy * t_arrival

    # Greedy nearest-neighbor starting from the cluster closest to u[1], v[1]
    visited = np.zeros(J, dtype=bool)
    order = []

    current = np.argmin((X_pred - u[1])**2 + (Y_pred - v[1])**2)
    order.append(current)
    visited[current] = True

    for _ in range(J - 1):
        last = order[-1]
        d2 = (X_pred - X_pred[last])**2 + (Y_pred - Y_pred[last])**2
        d2[visited] = np.inf
        nxt = np.argmin(d2)
        order.append(nxt)
        visited[nxt] = True

    order = np.array(order)

    u_new = u.copy()
    v_new = v.copy()
    u_new[1:-1] = u[1:-1][order]
    v_new[1:-1] = v[1:-1][order]

    return (
        u_new,
        v_new,
        X0[order],
        Y0[order],
        sx[order],
        sy[order],
        cluster_ids[order],
        order
    )


# ---------------------------------------------------------------------------
# Rolling Optimizers
# ---------------------------------------------------------------------------

def rolling_optimize_with_reorder(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    reorder_vec=None,
    dL=None,
    dtau=0.001,
    n_steps_segment=200,
    grad_clip=1e2,
    verbose=True,
    max_iters=500
):
    """
    Rolling optimization with a professor-controlled reordering policy.

    The drone path is optimized in segments. Between segments, remaining
    (not-yet-visited) cluster heads can be re-ordered via TSP to improve
    the plan for the rest of the tour.

    Two reordering modes are supported (use exactly one):
      - reorder_vec : list of path-length checkpoints where reordering occurs.
      - dL          : legacy spacing; reorders occur every dL units of path length.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d, p, L : float
        Drone speed, objective exponent, path length constraint.
    reorder_vec : array_like, optional
        Sorted path-length values at which TSP reordering is triggered.
    dL : float, optional
        Legacy mode: reorder every dL units of path length.
    dtau : float
        Pseudo-time step for the optimizer.
    n_steps_segment : int
        Optimizer steps per segment between reorder events.
    grad_clip : float
        Gradient clipping threshold for numerical stability.
    verbose : bool
        Print progress messages if True.
    max_iters : int
        Maximum number of optimization iterations.

    Returns
    -------
    uVec, vVec : ndarray
        Final optimized drone path.
    traj_all : ndarray
        Full trajectory history across all segments.
    X0_j, Y0_j, s_jx, s_jy : ndarray
        Final cluster data (in the order established by mid-tour reorders).
    n_reorders : int
        Number of reordering events that occurred.
    """
    N = len(X0_j)
    cluster_ids = np.arange(N)

    uVec, vVec = initialize_path_from_cluster_heads(X0_j, Y0_j, L)
    traj_all = [np.concatenate([uVec, vVec])]

    reorder_index = 0
    n_reorders = 0
    iteration = 0

    if reorder_vec is not None:
        reorder_vec = np.sort(np.asarray(reorder_vec))

    while iteration < max_iters:
        iteration += 1

        # Optimize one segment
        uVec, vVec, traj_segment = solve_ivp_fixed_order(
            uVec, vVec,
            s_d, p,
            X0_j, Y0_j,
            s_jx, s_jy,
            L,
            dtau=dtau,
            n_steps=n_steps_segment,
            grad_clip=grad_clip
        )
        traj_all.extend(traj_segment)

        # Compute current path length
        du = np.diff(uVec)
        dv = np.diff(vVec)
        ell = np.sqrt(du**2 + dv**2 + 1e-12)
        g_cum = np.concatenate([[0], np.cumsum(ell)])
        current_path_length = g_cum[-1]

        # --- Schedule-based reordering ---
        if reorder_vec is not None:
            if reorder_index < len(reorder_vec):
                if current_path_length >= reorder_vec[reorder_index]:
                    reorder_index += 1
                    n_reorders += 1

                    locked = min(reorder_index, N - 1)
                    remaining = np.arange(locked, N)

                    if len(remaining) > 1:
                        order_suffix = np.asarray(
                            solve_tsp_exact(X0_j[remaining], Y0_j[remaining]),
                            dtype=int
                        )

                        uVec[remaining]     = np.take(uVec[remaining], order_suffix)
                        vVec[remaining]     = np.take(vVec[remaining], order_suffix)
                        X0_j[remaining]     = np.take(X0_j[remaining], order_suffix)
                        Y0_j[remaining]     = np.take(Y0_j[remaining], order_suffix)
                        s_jx[remaining]     = np.take(s_jx[remaining], order_suffix)
                        s_jy[remaining]     = np.take(s_jy[remaining], order_suffix)
                        cluster_ids[remaining] = np.take(cluster_ids[remaining], order_suffix)

                        if verbose:
                            print(f"=== Reorder event {n_reorders} ===")
                            print(f"  Path length = {current_path_length:.4f}")
                            print(f"  Remaining clusters = {remaining.tolist()}")

        # --- Legacy dL-based reordering (no-op placeholder) ---
        elif dL is not None:
            locked_candidates = np.where(g_cum >= dL * (n_reorders + 1))[0]
            if len(locked_candidates) > 0:
                locked = locked_candidates[0]  # noqa: F841  (legacy; no action taken)

        if verbose:
            print(f"Iteration {iteration} | Path length = {current_path_length:.4f}")

        # Stop once all cluster heads have been locked in
        if reorder_index >= N - 1:
            break

    if verbose and iteration >= max_iters:
        print("Warning: reached max iterations")

    return uVec, vVec, np.vstack(traj_all), X0_j, Y0_j, s_jx, s_jy, n_reorders


def rolling_optimize_tsp(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    reorder_vec,
    dtau=1e-4,
    n_steps_segment=200,
    grad_clip=1e2,
    verbose=True,
    max_iters=500
):
    """
    Rolling optimization with an explicit TSP-based reorder schedule.

    Similar to rolling_optimize_with_reorder but always uses an explicit
    reorder_vec and applies TSP reordering on the remaining (unvisited) clusters.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d, p, L : float
        Drone speed, objective exponent, path length constraint.
    reorder_vec : array_like
        Sorted path-length values at which TSP reordering is triggered.
    dtau : float
        Pseudo-time step for the optimizer.
    n_steps_segment : int
        Optimizer steps per segment between reorder events.
    grad_clip : float
        Gradient clipping threshold.
    verbose : bool
        Print progress messages if True.
    max_iters : int
        Maximum number of optimization iterations.

    Returns
    -------
    uVec, vVec : ndarray
        Final optimized drone path.
    traj_all : ndarray
        Full trajectory history.
    X0_j, Y0_j, s_jx, s_jy : ndarray
        Final (reordered) cluster data.
    n_reorders : int
        Number of reordering events that occurred.
    """
    N = len(X0_j)
    reorder_vec = np.sort(np.asarray(reorder_vec))
    cluster_ids = np.arange(N)

    # Initial TSP ordering
    order_tsp = solve_tsp_exact(X0_j, Y0_j)
    X0_j     = X0_j[order_tsp]
    Y0_j     = Y0_j[order_tsp]
    s_jx     = s_jx[order_tsp]
    s_jy     = s_jy[order_tsp]
    cluster_ids = cluster_ids[order_tsp]

    uVec, vVec = initialize_path_from_cluster_heads(X0_j, Y0_j, L)
    traj_all = [np.concatenate([uVec, vVec])]

    reorder_index = 0
    n_reorders = 0
    iteration = 0

    while iteration < max_iters and reorder_index < len(reorder_vec):
        iteration += 1

        # Optimize one segment
        uVec, vVec, traj_segment = solve_ivp_fixed_order(
            uVec, vVec,
            s_d, p,
            X0_j, Y0_j,
            s_jx, s_jy,
            L,
            dtau=dtau,
            n_steps=n_steps_segment,
            grad_clip=grad_clip
        )
        traj_all.extend(traj_segment)

        # Compute current path length
        du = np.diff(uVec)
        dv = np.diff(vVec)
        ell = np.sqrt(du**2 + dv**2 + 1e-12)
        g_cum = np.concatenate([[0], np.cumsum(ell)])
        current_path_length = g_cum[-1]

        # Check if we've passed the next reorder checkpoint
        if current_path_length >= reorder_vec[reorder_index]:
            reorder_index += 1
            n_reorders += 1

            remaining = np.arange(reorder_index, N)

            if len(remaining) > 1:
                order_suffix = np.asarray(
                    solve_tsp_exact(X0_j[remaining], Y0_j[remaining]),
                    dtype=int
                )

                uVec[remaining]     = np.take(uVec[remaining], order_suffix)
                vVec[remaining]     = np.take(vVec[remaining], order_suffix)
                X0_j[remaining]     = np.take(X0_j[remaining], order_suffix)
                Y0_j[remaining]     = np.take(Y0_j[remaining], order_suffix)
                s_jx[remaining]     = np.take(s_jx[remaining], order_suffix)
                s_jy[remaining]     = np.take(s_jy[remaining], order_suffix)
                cluster_ids[remaining] = np.take(cluster_ids[remaining], order_suffix)

                if verbose:
                    print(f"=== Reorder event {n_reorders} ===")
                    print(f"  Path length = {current_path_length:.4f}")
                    print(f"  Remaining clusters: {remaining.tolist()}")
            else:
                break

    if verbose and iteration >= max_iters:
        print("Warning: reached max iterations")

    return uVec, vVec, np.vstack(traj_all), X0_j, Y0_j, s_jx, s_jy, n_reorders


# ---------------------------------------------------------------------------
# Index-triggered rolling optimizer
# ---------------------------------------------------------------------------

def _greedy_nearest(start_x, start_y, X, Y):
    """Greedy nearest-neighbor tour starting from (start_x, start_y)."""
    n = len(X)
    visited = np.zeros(n, dtype=bool)
    order = []
    cx, cy = start_x, start_y
    for _ in range(n):
        d2 = (X - cx)**2 + (Y - cy)**2
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        order.append(nxt)
        visited[nxt] = True
        cx, cy = X[nxt], Y[nxt]
    return np.array(order)


def rolling_optimize_indexed_reorder(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    reOrd,
    dtau=0.001,
    n_steps=500,
    grad_clip=1e2,
    use_greedy=False,
    tsp_time_limit=20,
):
    """
    Rolling optimization with reordering triggered at specific cluster-visit indices.

    Algorithm
    ---------
    1. Optimize the full path with the given initial cluster ordering.
    2. For each index j in reOrd (sorted):
       a. Compute t_j = (cumulative path distance to cluster j's visit point) / s_d.
          For each cluster k >= j, predict its position xtmp[k], ytmp[k] assuming
          the drone flies directly from uVec[j+1], vVec[j+1] to cluster k starting
          at time t_j.
       b. Reorder clusters k > j via TSP on xtmp[k], ytmp[k] (k > j only).
          The drone starts at u[j], v[j] and ends at u[-1], v[-1].
       c. If the order changed, re-solve the FULL problem (constraint length L)
          keeping clusters k <= j fixed and the new TSP order for k > j.
          The current u[], v[] is used as a warm-start initial solution.
       d. Use the new u[], v[] and updated cluster order for the next iteration.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions (at t = 0).
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d, p, L : float
        Drone speed, objective exponent, full path-length constraint.
    reOrd : array_like of int
        Sorted 0-based cluster indices at which reordering is triggered.
        Entry j means "reorder after visiting cluster j."
        Valid range: 0 <= j <= J-2.
    dtau : float
        Pseudo-time step for the optimizer.
    n_steps : int
        Optimizer steps for the initial pass and for each re-solve.
    grad_clip : float
        Gradient clipping threshold.
    use_greedy : bool
        If True, use greedy nearest-neighbor reordering instead of TSP.

    Returns
    -------
    uVec, vVec : ndarray
        Final optimized drone path (full path including endpoints).
    traj_final : ndarray, shape (1, 2*(J+2))
        Final state snapshot (full trajectory not retained across re-solves).
    X_cur, Y_cur, sx_cur, sy_cur : ndarray
        Original (t=0) cluster positions and velocities in their final visit
        order.  Pass these with the full path to gf_values_moving for correct
        metric calculation.
    n_reorders : int
        Number of reorder events that changed the cluster order.
    """
    J = len(X0_j)
    reOrd = sorted(int(r) for r in reOrd if 0 <= r <= J - 2)

    # Working copies — always store original (t=0) positions in current visit order
    X_cur  = X0_j.copy()
    Y_cur  = Y0_j.copy()
    sx_cur = s_jx.copy()
    sy_cur = s_jy.copy()

    # Step 1: initialize path and run full optimization
    uVec, vVec = initialize_path_from_cluster_heads(X_cur, Y_cur, L)
    uVec, vVec, _ = solve_ivp_fixed_order(
        uVec, vVec, s_d, p, X_cur, Y_cur, sx_cur, sy_cur,
        L, dtau=dtau, n_steps=n_steps, grad_clip=grad_clip
    )

    n_reorders = 0
    last_tsp_order = None   # warm-start cache for consecutive reorder calls
    last_tsp_size  = None

    # Step 2: loop through reorder checkpoints
    for j in reOrd:

        J_remaining = J - j - 1
        if J_remaining <= 0:
            break

        # ---- cumulative distances along current full path ----
        du = np.diff(uVec)
        dv = np.diff(vVec)
        ell = np.sqrt(du**2 + dv**2)
        g_cum = np.concatenate([[0], np.cumsum(ell)])

        g_j     = g_cum[j + 1]    # cumulative distance at cluster j's visit point
        t_j     = g_j / s_d       # arrival time at cluster j
        drone_x = uVec[j + 1]
        drone_y = vVec[j + 1]

        # ---- 2a. predicted positions for k >= j ----
        # For each cluster k >= j: find where it is at t_j, then predict where
        # it will be when the drone arrives flying directly from (drone_x, drone_y).
        xtmp = np.empty(J_remaining + 1)   # indices 0 = cluster j, 1..J_remaining = k > j
        ytmp = np.empty(J_remaining + 1)
        for idx in range(J_remaining + 1):
            k  = j + idx
            xk = X_cur[k] + sx_cur[k] * t_j
            yk = Y_cur[k] + sy_cur[k] * t_j
            t_direct = np.sqrt((drone_x - xk)**2 + (drone_y - yk)**2) / s_d
            xtmp[idx] = xk + sx_cur[k] * t_direct
            ytmp[idx] = yk + sy_cur[k] * t_direct

        # ---- 2b. Reorder on predicted positions for k > j ----
        if use_greedy:
            order_new = _greedy_nearest(drone_x, drone_y, xtmp[1:], ytmp[1:])
        else:
            # warm-start when subset size matches previous call
            warm = last_tsp_order if last_tsp_size == J_remaining else None
            order_new = solve_tsp_exact(xtmp[1:], ytmp[1:], tsp_time_limit=tsp_time_limit,
                                        initial_order=warm)
            last_tsp_order = order_new
            last_tsp_size  = J_remaining

        # skip re-solve if order is unchanged
        if np.array_equal(order_new, np.arange(J_remaining)):
            continue

        # ---- apply reordering ----
        # path interior points at full-path indices j+2 … J
        uVec[j + 2 : J + 1] = uVec[j + 2 : J + 1][order_new]
        vVec[j + 2 : J + 1] = vVec[j + 2 : J + 1][order_new]

        # cluster arrays for k > j (keep original t=0 values, just reorder)
        X_cur [j + 1 : J] = X_cur [j + 1 : J][order_new]
        Y_cur [j + 1 : J] = Y_cur [j + 1 : J][order_new]
        sx_cur[j + 1 : J] = sx_cur[j + 1 : J][order_new]
        sy_cur[j + 1 : J] = sy_cur[j + 1 : J][order_new]

        # ---- 2c. re-solve full problem with warm-started uVec, vVec ----
        # Clusters k <= j keep their positions; clusters k > j use new order.
        # The full path length constraint L is unchanged.
        uVec, vVec, _ = solve_ivp_fixed_order(
            uVec, vVec,
            s_d, p,
            X_cur, Y_cur, sx_cur, sy_cur,
            L,
            dtau=dtau, n_steps=n_steps, grad_clip=grad_clip
        )

        # Step D: uVec, vVec and X_cur/Y_cur are now ready for next iteration
        n_reorders += 1

    traj_final = np.concatenate([uVec, vVec])[np.newaxis, :]

    return uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur, n_reorders


# ---------------------------------------------------------------------------
# Iterative TSP optimizers (FinalX1, FinalX2)
# ---------------------------------------------------------------------------

def iterative_tsp_predicted(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    dtau=0.001,
    n_steps=500,
    grad_clip=1e2,
    max_iters=20,
    tsp_time_limit=20,
):
    """
    FinalX1: iteratively solve and TSP-reorder predicted cluster positions.

    Algorithm
    ---------
    1. Initialize path from current cluster ordering.
    2. Loop (up to max_iters):
       a. Solve the full optimization problem.
       b. For each cluster j, compute its predicted position at the drone's
          arrival time: X_pred[j] = X_cur[j] + sx[j] * t_arrival[j].
       c. Find the TSP ordering of X_pred, Y_pred.
       d. If the ordering is unchanged (identity), exit.
          Otherwise, reorder clusters and drone points and go to a.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)   Initial cluster positions (t=0).
    s_jx, s_jy : ndarray, shape (J,)   Cluster velocities.
    s_d, p, L  : float                 Drone speed, exponent, path-length constraint.
    dtau       : float                 Pseudo-time step.
    n_steps    : int                   Optimizer steps per solve.
    grad_clip  : float                 Gradient clipping threshold.
    max_iters  : int                   Maximum loop iterations (safety guard).

    Returns
    -------
    uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur : ndarray
    n_reorders : int  Number of reordering iterations that changed the order.
    """
    J = len(X0_j)
    X_cur  = X0_j.copy();  Y_cur  = Y0_j.copy()
    sx_cur = s_jx.copy();  sy_cur = s_jy.copy()

    uVec, vVec = initialize_path_from_cluster_heads(X_cur, Y_cur, L)
    n_reorders     = 0
    best_f         = np.inf
    best_state     = None
    current_perm   = np.arange(J)
    seen_orderings = {tuple(current_perm)}

    for _ in range(max_iters):
        # a. Solve
        uVec, vVec, _ = solve_ivp_fixed_order(
            uVec, vVec, s_d, p, X_cur, Y_cur, sx_cur, sy_cur,
            L, dtau=dtau, n_steps=n_steps, grad_clip=grad_clip
        )

        # Track the best result seen so far
        _, f_cur = gf_values_moving(uVec, vVec, X_cur, Y_cur, sx_cur, sy_cur, s_d, p)
        if f_cur < best_f:
            best_f     = f_cur
            best_state = (uVec.copy(), vVec.copy(),
                          X_cur.copy(), Y_cur.copy(),
                          sx_cur.copy(), sy_cur.copy())

        # b. Predicted cluster positions at drone arrival times
        du = np.diff(uVec);  dv = np.diff(vVec)
        ell = np.sqrt(du**2 + dv**2)
        g_cum = np.concatenate([[0], np.cumsum(ell)])
        t_arrival = g_cum[1:J+1] / s_d
        X_pred = X_cur + sx_cur * t_arrival
        Y_pred = Y_cur + sy_cur * t_arrival

        # c. TSP on predicted positions (warm-start from identity after first reorder)
        warm = np.arange(J) if n_reorders > 0 else None
        order_new = solve_tsp_exact(X_pred, Y_pred, tsp_time_limit=tsp_time_limit,
                                    initial_order=warm)

        # d. Exit if order unchanged or if this absolute ordering was seen before
        if np.array_equal(order_new, np.arange(J)):
            break
        new_perm = current_perm[order_new]
        if tuple(new_perm) in seen_orderings:
            break
        seen_orderings.add(tuple(new_perm))
        current_perm = new_perm

        # Apply reordering; use current path as warm start
        uVec[1:-1] = uVec[1:-1][order_new]
        vVec[1:-1] = vVec[1:-1][order_new]
        X_cur  = X_cur [order_new];  Y_cur  = Y_cur [order_new]
        sx_cur = sx_cur[order_new];  sy_cur = sy_cur[order_new]
        n_reorders += 1

    # Return the best result found across all iterations
    uVec, vVec, X_cur, Y_cur, sx_cur, sy_cur = best_state
    traj_final = np.concatenate([uVec, vVec])[np.newaxis, :]
    return uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur, n_reorders


def iterative_greedy_predicted(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    dtau=0.001,
    n_steps=500,
    grad_clip=1e2,
    max_iters=20,
):
    """
    GreedyX1: iteratively solve and greedy-reorder predicted cluster positions.

    Identical to FinalX1 (iterative_tsp_predicted) except that cluster reordering
    uses a greedy nearest-neighbor heuristic starting from the drone's origin
    instead of an exact TSP solve.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)   Initial cluster positions (t=0).
    s_jx, s_jy : ndarray, shape (J,)   Cluster velocities.
    s_d, p, L  : float                 Drone speed, exponent, path-length constraint.
    dtau       : float                 Pseudo-time step.
    n_steps    : int                   Optimizer steps per solve.
    grad_clip  : float                 Gradient clipping threshold.
    max_iters  : int                   Maximum loop iterations (safety guard).

    Returns
    -------
    uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur : ndarray
    n_reorders : int  Number of reordering iterations that changed the order.
    """
    J = len(X0_j)
    X_cur  = X0_j.copy();  Y_cur  = Y0_j.copy()
    sx_cur = s_jx.copy();  sy_cur = s_jy.copy()

    uVec, vVec = initialize_path_from_cluster_heads(X_cur, Y_cur, L)
    n_reorders     = 0
    best_f         = np.inf
    best_state     = None
    current_perm   = np.arange(J)
    seen_orderings = {tuple(current_perm)}

    for _ in range(max_iters):
        # a. Solve
        uVec, vVec, _ = solve_ivp_fixed_order(
            uVec, vVec, s_d, p, X_cur, Y_cur, sx_cur, sy_cur,
            L, dtau=dtau, n_steps=n_steps, grad_clip=grad_clip
        )

        # Track the best result seen so far
        _, f_cur = gf_values_moving(uVec, vVec, X_cur, Y_cur, sx_cur, sy_cur, s_d, p)
        if f_cur < best_f:
            best_f     = f_cur
            best_state = (uVec.copy(), vVec.copy(),
                          X_cur.copy(), Y_cur.copy(),
                          sx_cur.copy(), sy_cur.copy())

        # b. Predicted cluster positions at drone arrival times
        du = np.diff(uVec);  dv = np.diff(vVec)
        ell = np.sqrt(du**2 + dv**2)
        g_cum = np.concatenate([[0], np.cumsum(ell)])
        t_arrival = g_cum[1:J+1] / s_d
        X_pred = X_cur + sx_cur * t_arrival
        Y_pred = Y_cur + sy_cur * t_arrival

        # c. Greedy nearest-neighbor on predicted positions (start from drone origin)
        order_new = _greedy_nearest(uVec[0], vVec[0], X_pred, Y_pred)

        # d. Exit if order unchanged or if this absolute ordering was seen before
        if np.array_equal(order_new, np.arange(J)):
            break
        new_perm = current_perm[order_new]
        if tuple(new_perm) in seen_orderings:
            break
        seen_orderings.add(tuple(new_perm))
        current_perm = new_perm

        # Apply reordering; use current path as warm start
        uVec[1:-1] = uVec[1:-1][order_new]
        vVec[1:-1] = vVec[1:-1][order_new]
        X_cur  = X_cur [order_new];  Y_cur  = Y_cur [order_new]
        sx_cur = sx_cur[order_new];  sy_cur = sy_cur[order_new]
        n_reorders += 1

    # Return the best result found across all iterations
    uVec, vVec, X_cur, Y_cur, sx_cur, sy_cur = best_state
    traj_final = np.concatenate([uVec, vVec])[np.newaxis, :]
    return uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur, n_reorders


def iterative_tsp_drone_points(
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d, p, L,
    dtau=0.001,
    n_steps=500,
    grad_clip=1e2,
    max_iters=20,
    tsp_time_limit=20,
):
    """
    FinalX2: iteratively solve and TSP-reorder the drone's reception points.

    Algorithm
    ---------
    1. Initialize path from current cluster ordering.
    2. Loop (up to max_iters):
       a. Solve the full optimization problem.
       b. Find the TSP ordering of the optimized drone reception points
          u[1:-1], v[1:-1].
       c. If the ordering is unchanged (identity), exit.
          Otherwise, reorder clusters and drone points and go to a.

    Parameters
    ----------
    X0_j, Y0_j : ndarray, shape (J,)   Initial cluster positions (t=0).
    s_jx, s_jy : ndarray, shape (J,)   Cluster velocities.
    s_d, p, L  : float                 Drone speed, exponent, path-length constraint.
    dtau       : float                 Pseudo-time step.
    n_steps    : int                   Optimizer steps per solve.
    grad_clip  : float                 Gradient clipping threshold.
    max_iters  : int                   Maximum loop iterations (safety guard).

    Returns
    -------
    uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur : ndarray
    n_reorders : int  Number of reordering iterations that changed the order.
    """
    J = len(X0_j)
    X_cur  = X0_j.copy();  Y_cur  = Y0_j.copy()
    sx_cur = s_jx.copy();  sy_cur = s_jy.copy()

    uVec, vVec = initialize_path_from_cluster_heads(X_cur, Y_cur, L)
    n_reorders     = 0
    best_f         = np.inf
    best_state     = None
    current_perm   = np.arange(J)
    seen_orderings = {tuple(current_perm)}

    for _ in range(max_iters):
        # a. Solve
        uVec, vVec, _ = solve_ivp_fixed_order(
            uVec, vVec, s_d, p, X_cur, Y_cur, sx_cur, sy_cur,
            L, dtau=dtau, n_steps=n_steps, grad_clip=grad_clip
        )

        # Track the best result seen so far
        _, f_cur = gf_values_moving(uVec, vVec, X_cur, Y_cur, sx_cur, sy_cur, s_d, p)
        if f_cur < best_f:
            best_f     = f_cur
            best_state = (uVec.copy(), vVec.copy(),
                          X_cur.copy(), Y_cur.copy(),
                          sx_cur.copy(), sy_cur.copy())

        # b. TSP on drone reception points (warm-start from identity after first reorder)
        warm = np.arange(J) if n_reorders > 0 else None
        order_new = solve_tsp_exact(uVec[1:-1], vVec[1:-1], tsp_time_limit=tsp_time_limit,
                                    initial_order=warm)

        # c. Exit if order unchanged or if this absolute ordering was seen before
        if np.array_equal(order_new, np.arange(J)):
            break
        new_perm = current_perm[order_new]
        if tuple(new_perm) in seen_orderings:
            break
        seen_orderings.add(tuple(new_perm))
        current_perm = new_perm

        # Apply reordering; use current path as warm start
        uVec[1:-1] = uVec[1:-1][order_new]
        vVec[1:-1] = vVec[1:-1][order_new]
        X_cur  = X_cur [order_new];  Y_cur  = Y_cur [order_new]
        sx_cur = sx_cur[order_new];  sy_cur = sy_cur[order_new]
        n_reorders += 1

    # Return the best result found across all iterations
    uVec, vVec, X_cur, Y_cur, sx_cur, sy_cur = best_state
    traj_final = np.concatenate([uVec, vVec])[np.newaxis, :]
    return uVec, vVec, traj_final, X_cur, Y_cur, sx_cur, sy_cur, n_reorders
