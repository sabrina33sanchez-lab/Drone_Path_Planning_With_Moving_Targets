# main.py
"""
Drone Path Optimization - Main Program
=======================================
Run this file to compare three drone path planning algorithms on a test case
of moving cluster heads.

Usage
-----
    python main.py

This script:
  1. Defines a test scenario (cluster head positions and velocities).
  2. Calls test_optimizers() to run all three methods.
  3. Calls plot_optimizers_paths() to display a side-by-side comparison.

Module overview
---------------
  geometry.py     - Path length, initialization, and projection functions
  kinematics.py   - Cluster head position/velocity equations
  optimization.py - Gradient computation and pseudo-time ODE solver
  routing.py      - TSP solver and rolling optimization algorithms
  metrics.py      - Energy and objective evaluation
  visualization.py- Plotting and animation functions
  comparison.py   - Orchestrates and compares the three methods
"""

import numpy as np
import matplotlib.pyplot as plt
from comparison import test_optimizers
from visualization import plot_optimizers_paths


# =============================================================================
# Test scenario: cluster head initial positions and velocities
# =============================================================================

# --- Circular arrangement (10 cluster heads) ---
X0_j = np.array([-1,  2,  0,  2,  1, -1, -2,  1, -2,  2], dtype=float)
Y0_j = np.array([ 2,  0, -2, -1,  2, -2,  1, -2, -1,  1], dtype=float)
s_jx = np.array([ 0.1, -0.05,  0.0,  0.02, -0.03,  0.1,  0.05, -0.1,  -0.03,  0.09])
s_jy = np.array([ 0.0,  0.1,  -0.05, -0.02,  0.03,  0.1,  0.02, -0.04,  0.01, -0.06])

# --- Uncomment to try other arrangements ---

# # Quadrant arrangement
# X0_j = np.array([-1,  2,  1, -1.5, -1.75,  1, -2,  1.5,  1.25,  1.75], dtype=float)
# Y0_j = np.array([ 1.5, -2,  2,  1,   -2,    1, -1,  1.5,  -1,   -1.5],  dtype=float)
# s_jx = np.array([ 0.1, -0.05,  0.0,  0.02, -0.03,  0.1,  0.05, -0.1, -0.03,  0.09])
# s_jy = np.array([ 0.0,  0.1,  -0.05, -0.02,  0.03,  0.1,  0.02, -0.04,  0.01, -0.06])

# # Random arrangement
# X0_j = np.array([ 0,   2.5,  3,  -2,  -1.5,  2,  0.5,  1,  -1.75,  0.75], dtype=float)
# Y0_j = np.array([ 1.5, 1,    1.5, 2.5,  3,    2,  0.25, 0,   0.5,   1.75], dtype=float)
# s_jx = np.array([ 0.1, -0.05,  0.0,  0.09, -0.03,  0.1,  0.0, -0.01, -0.03,  0.01])
# s_jy = np.array([ 0.0,  0.1,  -0.05, -0.02,  0.03,  0.1,  0.02, -0.05,  0.1,  -0.05])


# =============================================================================
# Optimization parameters
# =============================================================================
s_d = 1.0   # drone speed
p   = 4.0   # path-loss exponent
L   = 15.0  # target path length constraint

# Cluster indices (0-based) at which Method 2 triggers a reorder.
# Reorder after every intermediate node (all except the last cluster).
reOrd = list(range(len(X0_j) - 1))


# =============================================================================
# Run and compare all three methods
# =============================================================================
results, _, _ = test_optimizers(X0_j, Y0_j, s_jx, s_jy, s_d, p, L, reOrd=reOrd)

# Plot side-by-side comparison
fig = plot_optimizers_paths(results, s_d)
plt.show()
plt.close(fig)
