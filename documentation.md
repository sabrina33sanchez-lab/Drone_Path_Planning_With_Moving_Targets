# Drone Path Optimization — Code Documentation

## Overview

This codebase optimizes the path of a drone that must visit a set of moving cluster heads (communication nodes) and return to its base. The drone departs from and returns to the origin `(0, 0)`. Each cluster head moves at constant velocity. The objective is to minimize the total communication cost — the sum of `distance^p` between each drone reception point and its corresponding cluster head at the time of arrival — subject to the constraint that the total drone path length equals a fixed value `L`.

A Monte Carlo framework runs multiple random test scenarios across configurable parameter settings and reports comparative statistics for ten optimizer variants.

---

## Problem Formulation

**State:** The drone path is a polyline with `J + 2` points:

```
(u_0, v_0) = (0,0),  (u_1, v_1), ..., (u_J, v_J),  (u_{J+1}, v_{J+1}) = (0,0)
```

The `J` interior points are the drone reception points, one per cluster head.

**Constraint:** Total path length equals `L`:

```
g(u, v) = sum of segment lengths = L
```

**Objective:** Minimize total communication cost:

```
f(u, v) = sum_{j=1}^{J}  dist(drone_point_j,  cluster_j_at_arrival_j)^p
```

where `arrival_j = cumulative_path_length_to_j / s_d` and `cluster_j_at_arrival_j = (X0_j + s_jx * arrival_j,  Y0_j + s_jy * arrival_j)`.

**Key dimensionless ratio:** The ratio `(sMax / s_d) * (L / dInit)` governs problem difficulty — it measures total cluster displacement relative to the initial spread. With `s_d = 1` fixed (choice of units), only `sMax` needs to vary across scenarios.

---

## Module Reference

### `kinematics.py`

Constant-velocity cluster head motion.

| Function | Description |
|---|---|
| `Xj(X0_j, s_jx, t_j)` | x-position at time `t_j` |
| `Yj(Y0_j, s_jy, t_j)` | y-position at time `t_j` |
| `Xj_prime(s_jx)` | x-velocity (returns `s_jx`) |
| `Yj_prime(s_jy)` | y-velocity (returns `s_jy`) |

---

### `geometry.py`

Path geometry utilities. Both endpoints are always fixed at `(0, 0)`, which is a required assumption for `project_to_length` to be correct.

| Function | Description |
|---|---|
| `path_length(uVec, vVec)` | Total polyline length |
| `initialize_path_from_cluster_heads(X0_j, Y0_j, L)` | Build initial path by placing interior points at cluster head positions, then scaling to length `L` |
| `project_to_length(u, v, L)` | Scale interior points radially from `u[0], v[0]` so that total path length equals `L`. Requires start = end = `(0,0)` |

---

### `metrics.py`

Objective function evaluation.

| Function | Description |
|---|---|
| `gf_values_moving(u_path, v_path, X0_j, Y0_j, sx_j, sy_j, s_d, p)` | Returns `(g, f)` — path length and objective value accounting for cluster head motion. Primary evaluation function used throughout. |
| `gf_values(u, v, u0, v0, X_j, Y_j, p)` | Returns `(g, f)` for static (non-moving) cluster heads. |
| `total_clusterhead_energy(...)` | Equivalent to `f` in `gf_values_moving`; kept for reference. |

---

### `optimization.py`

Core gradient-descent optimizer. Minimizes `f` subject to `g = L` using pseudo-time Euler integration with backtracking line search.

#### Pseudo-time flow

The optimizer evolves the path along the constrained gradient flow:

```
d(u,v)/dτ = -(∇g · ∇g) ∇f  +  (∇f · ∇g) ∇g
```

This direction is tangent to the constraint surface `g = L` and descends in `f`. `τ` (pseudo-time) is a purely numerical variable with no physical meaning — it is unrelated to the physical time used by `s_d`.

#### Key functions

| Function | Description |
|---|---|
| `grad_g(uVec, vVec)` | Gradient of path-length constraint, shape `(2N,)` |
| `grad_f(uVec, vVec, s_d, p, X0_j, Y0_j, s_jx, s_jy)` | Gradient of objective including Jacobian terms for cluster motion, shape `(2N,)` |
| `rhs_tau(...)` | Constrained flow RHS; zeros out fixed endpoint components |
| `solve_ivp_fixed_order(...)` | Main optimizer — Euler steps with backtracking line search and convergence check |

#### `solve_ivp_fixed_order` algorithm

1. Evaluate `f_cur` at the initial path.
2. For each step up to `n_steps`:
   - Compute `rhs_tau` and clip to `grad_clip`.
   - **Backtracking line search:** try step size `dtau`, halve up to `max_backtracks` times until `f` decreases after projection.
   - Project back onto `g = L`.
   - **Convergence check** every `ivpIter = 50` steps: if relative decrease in `f` over last 50 steps is less than `ivpTol = 0.001`, stop early.
3. Print step count if steps exceeded 200.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `dtau` | `0.1` | Initial step size (backtracking will reduce as needed) |
| `n_steps` | `500` | Maximum number of steps |
| `grad_clip` | `1e2` | Gradient clipping threshold |
| `max_backtracks` | `10` | Maximum halvings per step |
| `ivpIter` | `50` | Convergence check interval |
| `ivpTol` | `0.001` | Relative `f` change threshold for convergence |

**Scale invariance:** With `p = 2` the optimizer is scale-invariant — `dtau` does not need to change when `L`, `dInit`, or other length scales are changed by a uniform factor.

---

### `routing.py`

TSP solver and rolling/iterative optimization algorithms.

#### `solve_tsp_exact(X, Y, tsp_time_limit, initial_order)`

Solves TSP via OR-Tools Guided Local Search. Runs in time slices of `TSPsec = 0.15` seconds; stops when a full slice produces no improvement (rather than always running to `tsp_time_limit`). Prints improvement count and time of last improvement on each call.

- **Warm start:** pass `initial_order` to start local search from a known route via `SolveFromAssignmentWithParameters`.
- Node 0 is the implicit depot and must be excluded from `initial_order`.

#### `rolling_optimize_indexed_reorder(X0_j, Y0_j, ..., reOrd, use_greedy)`

Rolling optimizer triggered at specific cluster-visit indices.

1. Solve full path with initial ordering.
2. For each index `j` in `reOrd`:
   - Predict cluster positions at time `t_j` (drone arrival at cluster `j`).
   - Reorder remaining clusters `k > j` by TSP (or greedy) on predicted positions.
   - If order changed: warm-start re-solve of the full problem.

#### `iterative_tsp_predicted` (FinalXY)

Iteratively solve + TSP-reorder predicted cluster positions until ordering stabilises. Warm-starts TSP from `np.arange(J)` on iterations after the first (clusters are already in TSP-optimal order). Returns the best result across all iterations.

#### `iterative_greedy_predicted` (FinalXY_Greed)

Same as `iterative_tsp_predicted` but uses greedy nearest-neighbor reordering. No TSP calls.

#### `iterative_tsp_drone_points` (FinalUV)

Same as `iterative_tsp_predicted` but TSP is run on the drone reception points `(u[1:-1], v[1:-1])` rather than predicted cluster positions.

---

### `comparison.py`

Runs all ten optimizer variants on a single test case and returns a unified results dictionary.

#### Initial ordering

Before any method runs, an initial TSP ordering is computed on predicted cluster positions at the midpoint time `t = L / (2 * s_d)`. Both forward and reversed versions of this ordering are run for each method, giving 10 variants total:

| Index | Name |
|---|---|
| 0 | FinalXY |
| 1 | FinalXY_Greed |
| 2 | FinalUV |
| 3 | Predicted Rolling |
| 4 | Greedy Rolling |
| 5 | FinalXY Rev |
| 6 | FinalXY_Greed Rev |
| 7 | FinalUV Rev |
| 8 | Predicted Rolling Rev |
| 9 | Greedy Rolling Rev |

The forward/reverse assignment is based on which starting cluster is closer to the origin in predicted positions.

#### Results dictionary

Each method key maps to:

```python
{
    'g':        float,   # path length
    'f':        float,   # objective value
    'reorders': int,     # number of reorder events
    'time':     float,   # wall-clock time (seconds)
    'u', 'v':   ndarray, # final drone path
    'X', 'Y':   ndarray, # cluster positions at t=0 (in final visit order)
    'sx', 'sy': ndarray, # cluster velocities (in final visit order)
}
```

#### `test_optimizers` summary table

Printed after each test case — columns: `g`, `f`, `% Improv` (relative to initial ordering objective), `#Reorders`, `Time(s)`.

---

### `visualization.py`

| Function | Description |
|---|---|
| `plot_optimizers_paths(results, s_d)` | 2×5 grid of drone trajectories for all 10 methods. Returns `fig`. |
| `plot_monte_carlo_paths(results, s_d)` | Thin wrapper around `plot_optimizers_paths`. Returns `fig`. |
| `plot_best_solution(results, best_method, s_d, test_num)` | Large single-panel plot of the winning method. Returns `fig`. |
| `plot_drone_and_clusterheads(...)` | Grid of 8 snapshots showing path evolution over pseudo-time. |
| `plot_final_path_with_labeled_clusterhead_arrivals(...)` | Final path with numbered cluster arrival labels. |
| `animate_final_path_smooth(...)` | Animation of drone flying with moving cluster heads. |

All plot functions return figures without calling `plt.show()` — the caller is responsible for displaying or saving.

---

### `monte_carlo.py`

Monte Carlo evaluation across multiple scenarios.

#### Scenario definition

Scenarios are defined by three parallel arrays — the `m`-th entry of each defines the `m`-th scenario:

```python
L_values    = [...]   # drone path-length constraints
sMax_values = [...]   # maximum cluster head speeds
nCH_values  = [...]   # number of cluster heads
```

All scenarios share `nTest`, `dInit`, `s_d`, `p`, `seed`, `dtau`, `n_steps`, `tsp_time_limit`.

#### Cluster head placement

Cluster head initial positions are generated using a **Halton low-discrepancy sequence** mapped to a disk of radius `dInit`. This gives more uniform coverage than pure random sampling and is reproducible given `seed`.

#### Winner selection

For each test case, the winning method(s) are those achieving the minimum `f`. Ties between a forward method and its reverse counterpart are resolved in favour of the forward method.

**Combined methods:** For each forward/reverse pair, the element-wise minimum `f` across the pair is taken as the combined score.

#### Output files (all saved to `results/` subfolder)

| File | Content |
|---|---|
| `scenario_params.json` | All scenario and global parameters |
| `results_{tag}.pkl` | Full results dict for scenario `{tag}` |
| `results_{tag}.npz` | Key numeric arrays for scenario `{tag}` |
| `paths_{tag}.png` | `plot_monte_carlo_paths` for last test case |
| `fig1a_{tag}.png` | Histogram of best `f` values |
| `fig1b_{tag}.png` | Win frequency bar chart (individual methods) |
| `fig1c_{tag}.png` | Mean ± std of `(f − best f)` per method |
| `fig1c_pct_{tag}.png` | Mean ± std of % improvement over initial ordering |
| `fig2a_{tag}.png` | Win frequency bar chart (combined methods) |
| `fig2b_{tag}.png` | Mean ± std of `(f − best f)` per combined method |
| `fig2b_pct_{tag}.png` | Mean ± std of % improvement (combined) |
| `fig3_{tag}.png` | Reorder count distribution |
| `fig4_{tag}.png` | Execution time per method |

The scenario tag has the form `L{L:.0f}_sMax{sMax:.2f}_nCH{nCH}`, e.g. `L14000_sMax0.20_nCH20`.

#### Summary tables printed to console

- Initial TSP path lengths and initial ordering objectives
- Individual method wins, mean `f`, std `f`, mean % improvement
- Combined method summary
- Reorder counts (iterative methods only)
- Execution times

---

## Module Dependency Graph

```
monte_carlo.py
├── comparison.py
│   ├── routing.py
│   │   ├── geometry.py
│   │   ├── optimization.py
│   │   │   ├── geometry.py
│   │   │   └── metrics.py
│   │   └── metrics.py
│   ├── geometry.py
│   └── metrics.py
└── visualization.py

main.py
├── comparison.py
└── visualization.py
```

---

## Running the Code

**Full Monte Carlo run:**
```
python monte_carlo.py
```

**Single test case:**
```
python main.py
```

**With line profiling:**
```
kernprof -l -v monte_carlo.py
```
Add `@profile` decorators to target functions before running. Remove them before normal use.

---

## Key Assumptions

- Drone departs from and returns to `(0, 0)`.
- All cluster heads move at **constant velocity** throughout the drone tour.
- The path discretization has exactly `J` interior points — one per cluster head. The visiting order of interior points matches the cluster head visit order.
- `project_to_length` is only correct when both endpoints are the same point (both `(0,0)`).
- `s_d = 1.0` is a choice of units. Only the ratio `sMax / s_d` affects problem difficulty.
