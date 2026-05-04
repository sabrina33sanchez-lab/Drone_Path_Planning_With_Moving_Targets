# optimization.py
"""
Gradient computation and pseudo-time ODE solver.

This module implements the core math for the constrained path optimization:
  - grad_g : gradient of the path-length constraint
  - grad_f : gradient of the communication energy objective
  - rhs_tau : right-hand side of the pseudo-time evolution ODE
  - solve_ivp_fixed_order : Euler integrator that minimizes f subject to g = L
"""

import numpy as np
from geometry import project_to_length
from metrics import gf_values_moving


def grad_g(uVec, vVec):
    """
    Compute the gradient of the path-length constraint g(u,v).

    The result is a vector of length 2*N:
      - First N entries  : partial derivatives d g / d u_i
      - Last  N entries  : partial derivatives d g / d v_i

    Endpoint derivatives are zero (endpoints are fixed).

    Parameters
    ----------
    uVec, vVec : ndarray, shape (N,)
        Full drone path including fixed endpoints.

    Returns
    -------
    ndarray, shape (2*N,)
        Concatenated gradient [dg/du, dg/dv].
    """
    du = uVec[:-1] - uVec[1:]
    dv = vVec[:-1] - vVec[1:]
    ell = np.sqrt(du**2 + dv**2)
    ell = np.maximum(ell, 1e-8)  # avoid divide-by-zero for collapsed segments

    dg_du = np.zeros_like(uVec)
    dg_dv = np.zeros_like(vVec)

    dg_du[1:-1] = (uVec[1:-1] - uVec[2:]) / ell[1:] - (uVec[:-2] - uVec[1:-1]) / ell[:-1]
    dg_dv[1:-1] = (vVec[1:-1] - vVec[2:]) / ell[1:] - (vVec[:-2] - vVec[1:-1]) / ell[:-1]

    dg_du = np.nan_to_num(dg_du)
    dg_dv = np.nan_to_num(dg_dv)

    return np.concatenate([dg_du, dg_dv])


def grad_f(uVec, vVec, s_d, p, X0_j, Y0_j, s_jx, s_jy, eps=1e-12):
    """
    Compute the gradient of the objective function f(u,v).

    f is the total communication energy: sum over clusters of (distance)^p,
    where distance is measured at the drone's arrival time at each cluster.

    The gradient accounts for the time-dependence of cluster positions
    through Jacobian terms Gu and Gv (how arrival times change with path shape).

    Parameters
    ----------
    uVec, vVec : ndarray, shape (N,)
        Drone path including fixed endpoints.
    s_d : float
        Drone speed.
    p : float
        Exponent in the objective (e.g. 4 for path-loss model).
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    ndarray, shape (2*N,)
        Concatenated gradient [df/du, df/dv].
    """
    N = len(uVec)
    J = N - 2  # number of interior points = number of cluster heads

    # Segment lengths
    du = uVec[:-1] - uVec[1:]
    dv = vVec[:-1] - vVec[1:]
    ell = np.sqrt(du**2 + dv**2 + eps)

    # Cumulative path length -> arrival times for each cluster
    g_cum = np.cumsum(ell)
    t_j = g_cum[:J] / s_d

    # Cluster positions at arrival times
    X = X0_j + s_jx * t_j
    Y = Y0_j + s_jy * t_j

    # Squared distances from drone interior points to cluster positions
    a = uVec[1:-1] - X
    b = vVec[1:-1] - Y
    A = a**2 + b**2

    w = (p / 2) * A**(p/2 - 1)  # common weight in gradient

    # Gradient of path-length (used for Jacobian terms)
    dg_du = np.zeros(N)
    dg_dv = np.zeros(N)
    dg_du[1:-1] = (uVec[1:-1] - uVec[2:]) / ell[1:] - (uVec[:-2] - uVec[1:-1]) / ell[:-1]
    dg_dv[1:-1] = (vVec[1:-1] - vVec[2:]) / ell[1:] - (vVec[:-2] - vVec[1:-1]) / ell[:-1]

    # Jacobians: how arrival time of cluster j changes with each path variable
    Gu = np.zeros((J, N))
    Gv = np.zeros((J, N))
    for j in range(J):
        Gu[j, :j+1] += dg_du[:j+1]
        Gu[j, j+1] += -(uVec[j] - uVec[j+1]) / ell[j]
        Gv[j, :j+1] += dg_dv[:j+1]
        Gv[j, j+1] += -(vVec[j] - vVec[j+1]) / ell[j]

    # Identity rows selecting interior path variables
    delta = np.eye(N)[1:-1]  # shape (J, N)

    # Chain rule: dA/du and dA/dv include both direct and time-dependent terms
    dA_du = 2*a[:, None] * (delta - (s_jx[:, None]/s_d) * Gu) \
            - 2*b[:, None] * (s_jy[:, None]/s_d) * Gu

    dA_dv = -2*a[:, None] * (s_jx[:, None]/s_d) * Gv \
            + 2*b[:, None] * (delta - (s_jy[:, None]/s_d) * Gv)

    df_du = np.sum(w[:, None] * dA_du, axis=0)
    df_dv = np.sum(w[:, None] * dA_dv, axis=0)

    return np.concatenate([df_du, df_dv])


def rhs_tau(uVec, vVec, s_d, p, X0_j, Y0_j, s_jx, s_jy):
    """
    Compute the right-hand side of the pseudo-time constrained flow:

        d(u,v)/dτ = -(∇g · ∇g) ∇f  +  (∇f · ∇g) ∇g

    This keeps the path on the constraint surface g = L while
    descending in f. Fixed endpoints are enforced by zeroing
    the corresponding RHS components.

    Parameters
    ----------
    uVec, vVec : ndarray, shape (N,)
        Current drone path.
    s_d, p, X0_j, Y0_j, s_jx, s_jy : various
        Physical parameters passed to grad_f.

    Returns
    -------
    rhs : ndarray, shape (2*N,)
        Concatenated RHS [du/dτ, dv/dτ].
    """
    N = len(uVec)

    gradg = grad_g(uVec, vVec)
    gradf = grad_f(uVec, vVec, s_d, p, X0_j, Y0_j, s_jx, s_jy)

    # Replace any NaNs/Infs before computing inner products
    gradg = np.nan_to_num(gradg, nan=0.0, posinf=0.0, neginf=0.0)
    gradf = np.nan_to_num(gradf, nan=0.0, posinf=0.0, neginf=0.0)

    gg = np.dot(gradg, gradg)
    fg = np.dot(gradf, gradg)

    rhs = -gg * gradf + fg * gradg

    # Enforce fixed endpoints (first and last path points stay put)
    rhs[0] = rhs[N-1] = 0.0      # u-components
    rhs[N] = rhs[2*N - 1] = 0.0  # v-components

    return rhs


def solve_ivp_fixed_order(
    uVec, vVec,
    s_d, p,
    X0_j, Y0_j,
    s_jx, s_jy,
    L,
    dtau=0.1,
    n_steps=500,
    grad_clip=1e2,
    max_backtracks=10,
):
    """
    Minimize f(u,v) subject to g(u,v) = L using pseudo-time Euler integration
    with backtracking line search.

    At each step:
      1. Compute the constrained-flow RHS via rhs_tau.
      2. Clip large gradient values for numerical stability.
      3. Check for convergence every 25 steps.
      4. Backtracking line search: try dtau, halve until f decreases.
      5. Project back to the constraint g = L.

    The visiting order of interior path points (and their corresponding
    cluster heads) is held fixed throughout.

    Parameters
    ----------
    uVec, vVec : ndarray
        Initial drone path (including fixed endpoints).
    s_d, p, X0_j, Y0_j, s_jx, s_jy, L : various
        Physical and problem parameters.
    dtau : float
        Initial step size for the backtracking line search.
    n_steps : int
        Maximum number of steps.
    grad_clip : float
        Maximum allowed magnitude for any gradient component.
    max_backtracks : int
        Maximum number of step-size halvings per iteration.

    Returns
    -------
    u_final, v_final : ndarray
        Optimized path after convergence or n_steps steps.
    trajectory : ndarray, shape (steps_taken, 2*N)
        History of [u; v] at each step (useful for visualization).
    """
    ivpIter = 50      # check convergence every this many steps
    ivpTol  = 0.001  # relative change in f threshold

    ivpIter = 50      # check convergence every this many steps
    ivpTol  = 0.001  # relative change in f threshold

    N = len(uVec)
    w = np.concatenate([uVec, vVec])
    w_prev = w.copy()  # kept only for NaN recovery
    n_steps_taken = 0

    dtau_try = dtau

    _, f_cur = gf_values_moving(w[:N], w[N:], X0_j, Y0_j, s_jx, s_jy, s_d, p)
    f_check = f_cur  # reference f value for convergence check

    for step in range(n_steps):

        rhs = rhs_tau(
            w[:N], w[N:],
            s_d, p,
            X0_j, Y0_j,
            s_jx, s_jy
        )

        rhs = np.clip(rhs, -grad_clip, grad_clip)

        # Backtracking line search
        dtau_try = dtau
        for _ in range(max_backtracks):
            w_new = w + dtau_try * rhs
            u_new, v_new = project_to_length(w_new[:N], w_new[N:], L)
            w_new = np.concatenate([u_new, v_new])
            _, f_new = gf_values_moving(w_new[:N], w_new[N:], X0_j, Y0_j, s_jx, s_jy, s_d, p)
            if f_new < f_cur:
                break
            dtau_try *= 0.5

        # Safety: reset to last good state if NaNs appear
        if np.any(np.isnan(w_new)):
            print(f"Warning: NaN detected at step {step}, resetting to previous state")
            w_new = w_prev.copy()
            f_new = f_cur

        w_prev = w.copy()
        w = w_new
        f_cur = f_new
        n_steps_taken += 1

        # Convergence check: relative change in f over last ivpIter steps
        if (step + 1) % ivpIter == 0:
            if f_check - f_cur < ivpTol * f_cur:
                break
            f_check = f_cur

    # if n_steps_taken > 200:
    #     print(f"  IVP steps: {n_steps_taken}  f={f_cur:.4f}  dtau_last={dtau_try:.2e}")

    # Return a minimal trajectory (final state only) to satisfy the API
    trajectory = w[np.newaxis, :]
    return w[:N], w[N:], trajectory
