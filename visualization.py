# visualization.py
"""
Visualization functions for drone path optimization.

Provides four plotting/animation functions:
  - plot_drone_and_clusterheads                  : grid of path-evolution snapshots
  - plot_final_path_with_labeled_clusterhead_arrivals : final path with arrival labels
  - animate_final_path_smooth                    : smooth animation with moving clusters
  - plot_optimizers_paths                        : side-by-side comparison of 3 methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_drone_and_clusterheads(
    trajectory, u_init, v_init,
    X0_j, Y0_j, s_jx, s_jy,
    s_d, n_tests=8
):
    """
    Produce a grid of snapshots showing the drone path evolving over pseudo-time.

    Each panel shows the accumulated path history up to a particular pseudo-time
    step, the initial cluster head positions, and where the cluster heads end up
    by the final step.

    Parameters
    ----------
    trajectory : ndarray, shape (n_steps, 2*N)
        Array of concatenated [u; v] states across pseudo-time steps.
    u_init, v_init : ndarray, shape (N,)
        Initial drone path (including fixed endpoints).
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d : float
        Drone speed (used to convert cumulative distance to arrival times).
    n_tests : int
        Number of snapshot panels to draw (at most 8 with the 2x4 layout).
    """
    N = len(u_init)
    J = N - 2

    n_steps_to_plot = min(n_tests, trajectory.shape[0])
    steps_to_plot = np.linspace(0, trajectory.shape[0] - 1, n_steps_to_plot, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Precompute cluster positions at the final pseudo-time step
    w_final = trajectory[-1]
    u_final_arr = w_final[:N]
    v_final_arr = w_final[N:]
    du_final = np.diff(u_final_arr)
    dv_final = np.diff(v_final_arr)
    ell_final = np.sqrt(du_final**2 + dv_final**2 + 1e-12)
    g_cum_final = np.cumsum(ell_final)
    t_j_final = g_cum_final[:J] / s_d
    X_end = X0_j + s_jx * t_j_final
    Y_end = Y0_j + s_jy * t_j_final

    for i, ax in enumerate(axes):
        w = trajectory[steps_to_plot[i]]
        u = w[:N]
        v = w[N:]

        # Plot drone path history up to this snapshot
        for k in range(steps_to_plot[i] + 1):
            wk = trajectory[k]
            ax.plot(wk[:N], wk[N:], color='purple', alpha=0.3)

        # Initial interior drone points
        ax.scatter(u_init[1:-1], v_init[1:-1],
                   marker='x', color='orange', s=80, label='Initial drone pts')

        # Cluster heads: start and end
        ax.scatter(X0_j, Y0_j, marker='x', color='green', s=80, label='Cluster start')
        ax.scatter(X_end, Y_end, marker='x', color='blue',  s=80, label='Cluster end')

        # Drone start and end markers
        ax.scatter(u[0],  v[0],  color='green', s=80, label='Drone start')
        ax.scatter(u[-1], v[-1], color='red',   s=80, label='Drone end')

        ax.set_title(f'Snapshot {i}')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.grid(True)

        if i == 0:
            ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_final_path_with_labeled_clusterhead_arrivals(
    trajectory, u_init, v_init,
    X0_j, Y0_j, s_jx, s_jy,
    s_d
):
    """
    Plot the final optimized drone path with cluster heads labeled at arrival.

    Each cluster head label (number) appears at the cluster's position when
    the drone arrives, and at the corresponding drone reception point.

    Parameters
    ----------
    trajectory : ndarray, shape (n_steps, 2*N)
        Pseudo-time trajectory history.
    u_init, v_init : ndarray, shape (N,)
        Initial drone path (used to determine N).
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d : float
        Drone speed.
    """
    w_final = trajectory[-1]
    N = len(u_init)
    J = N - 2

    u = w_final[:N]
    v = w_final[N:]

    # Arrival times
    du = np.diff(u)
    dv = np.diff(v)
    ell = np.sqrt(du**2 + dv**2 + 1e-12)
    g_cum = np.cumsum(ell)
    t_arrival = g_cum[:J] / s_d

    X_arr = X0_j + s_jx * t_arrival
    Y_arr = Y0_j + s_jy * t_arrival

    plt.figure(figsize=(9, 7))
    plt.plot(u, v, color='purple', linewidth=2, label='Optimized drone path')
    plt.scatter(X_arr, Y_arr, color='blue', marker='x', s=100,
                label='Cluster heads at arrival')
    plt.scatter(u[1:-1], v[1:-1], color='orange', s=40,
                label='Drone reception points')

    for j in range(J):
        plt.text(X_arr[j] + 0.02, Y_arr[j] + 0.02, str(j), color='blue',       fontsize=9)
        plt.text(u[j+1]  + 0.02, v[j+1]  + 0.02, str(j), color='darkorange', fontsize=9)

    plt.scatter(u[0],  v[0],  color='green', s=100, label='Start')
    plt.scatter(u[-1], v[-1], color='red',   s=100, label='End')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Final Optimized Path with Labeled Cluster Head Arrivals')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_final_path_smooth(
    u, v,
    X0_j, Y0_j,
    s_jx, s_jy,
    s_d,
    t_arrival=None,
    save_gif=True,
    target_fps=25,
    playback_speed=1.0
):
    """
    Animate the drone flying along the optimized path with moving cluster heads.

    Cluster heads move continuously until the drone arrives; after that they freeze.
    The animation can be saved as a GIF.

    Parameters
    ----------
    u, v : ndarray
        Drone trajectory coordinates (full path including endpoints).
    X0_j, Y0_j : ndarray, shape (J,)
        Initial cluster head positions.
    s_jx, s_jy : ndarray, shape (J,)
        Cluster head velocities.
    s_d : float
        Drone speed.
    t_arrival : ndarray, optional
        Pre-computed arrival times per cluster. If None, defaults to zeros
        (cluster heads treated as stationary).
    save_gif : bool
        If True, save the animation to 'drone_cluster_smooth.gif'.
    target_fps : int
        Frames per second for the animation.
    playback_speed : float
        Values > 1.0 speed up playback; values < 1.0 slow it down.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    pad = 0.5
    ax.set_xlim(min(u) - pad, max(u) + pad)
    ax.set_ylim(min(v) - pad, max(v) + pad)
    ax.set_aspect('equal')
    ax.grid(True)

    path_line,    = ax.plot([], [], color='purple', lw=2)
    drone_dot,    = ax.plot([], [], 'ro', markersize=6)
    cluster_scatter = ax.scatter(X0_j, Y0_j, marker='x', s=100, color='blue')

    # Cumulative time along the path
    du = np.diff(u)
    dv = np.diff(v)
    ell = np.sqrt(du**2 + dv**2 + 1e-12)
    t_path_segment = np.concatenate([[0], np.cumsum(ell)]) / s_d
    T_max = t_path_segment[-1]

    if t_arrival is None:
        t_arrival = np.zeros(len(X0_j))
    t_arrival_segment = np.minimum(t_arrival, T_max)

    # Adaptive frame count
    n_frames = max(int(T_max * target_fps / playback_speed), 100)
    frames = np.linspace(0, T_max, n_frames)

    def update(frame_time):
        t = frame_time

        idx = np.searchsorted(t_path_segment, t) - 1
        idx = np.clip(idx, 0, len(u) - 2)

        dt_seg = t_path_segment[idx+1] - t_path_segment[idx]
        frac = (t - t_path_segment[idx]) / dt_seg if dt_seg > 0 else 0.0

        drone_x = u[idx] + frac * (u[idx+1] - u[idx])
        drone_y = v[idx] + frac * (v[idx+1] - v[idx])

        path_line.set_data(list(u[:idx+1]) + [drone_x],
                           list(v[:idx+1]) + [drone_y])
        drone_dot.set_data([drone_x], [drone_y])

        # Cluster heads freeze at their arrival time
        X_now = [X0_j[j] + s_jx[j] * min(t, t_arrival_segment[j]) for j in range(len(X0_j))]
        Y_now = [Y0_j[j] + s_jy[j] * min(t, t_arrival_segment[j]) for j in range(len(Y0_j))]
        cluster_scatter.set_offsets(list(zip(X_now, Y_now)))

        return path_line, drone_dot, cluster_scatter

    ani = FuncAnimation(fig, update, frames=frames,
                        interval=1000 / target_fps, blit=True)

    if save_gif:
        ani.save('drone_cluster_smooth.gif', writer='pillow', fps=target_fps)

    plt.show()


def plot_optimizers_paths(results, s_d):
    """
    Plot drone trajectories for all ten optimizer variants in portrait orientation.

    Arranged as a 5x2 grid (5 rows, 2 columns): left column = forward variants,
    right column = reversed variants.

    For each cluster head, three positions are plotted and connected by a
    dotted line:
      - Initial position (green)
      - Position at drone arrival (red)
      - Final position at end of drone tour (blue)

    Parameters
    ----------
    results : dict
        Dictionary returned by test_optimizers. Keys are method names,
        each mapping to a dict with keys: 'u', 'v', 'X', 'Y', 'sx', 'sy',
        'reorders'.
    s_d : float
        Drone speed (used to compute arrival times from path distances).
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 24))
    axes = axes.flatten()

    methods = [
        'FinalXY',      'FinalXY_Greed',      'FinalUV',      'Predicted Rolling',      'Greedy Rolling',
        'FinalXY Rev',  'FinalXY_Greed Rev',  'FinalUV Rev',  'Predicted Rolling Rev',  'Greedy Rolling Rev',
    ]
    colors = [
        'tab:red', 'tab:cyan', 'tab:purple', 'tab:orange', 'tab:green',
        'tab:red', 'tab:cyan', 'tab:purple', 'tab:orange', 'tab:green',
    ]

    for idx, method in enumerate(methods):
        u  = results[method]['u']
        v  = results[method]['v']
        X  = results[method]['X']
        Y  = results[method]['Y']
        sx = results[method]['sx']
        sy = results[method]['sy']
        J  = len(X)

        # Arrival times at each cluster node
        du = np.diff(u)
        dv = np.diff(v)
        ell = np.sqrt(du**2 + dv**2)
        cumulative_dist = np.concatenate([[0], np.cumsum(ell)])
        t_arrival = cumulative_dist[1:J+1] / s_d
        t_final   = cumulative_dist[-1] / s_d

        X_init    = X
        Y_init    = Y
        X_arrival = X + sx * t_arrival
        Y_arrival = Y + sy * t_arrival
        X_final   = X + sx * t_final
        Y_final   = Y + sy * t_final

        ax = axes[idx]
        ax.plot(u, v, color=colors[idx], label='Drone path', lw=2)

        # Dotted lines connecting the three cluster positions
        for j in range(J):
            ax.plot(
                [X_init[j], X_arrival[j], X_final[j]],
                [Y_init[j], Y_arrival[j], Y_final[j]],
                'k:', lw=0.8
            )

        ax.scatter(X_init,    Y_init,    color='green', marker='o', s=50, zorder=5,
                   label='Initial position')
        ax.scatter(X_arrival, Y_arrival, color='red',   marker='o', s=50, zorder=5,
                   label='Position at drone arrival')
        ax.scatter(X_final,   Y_final,   color='blue',  marker='o', s=50, zorder=5,
                   label='Final position')
        ax.scatter([0], [0], color='black', marker='*', s=120, zorder=5,
                   label='Drone start')

        ax.set_title(f"{method} (Reorders: {results[method]['reorders']})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_best_solution(results, best_method, s_d, test_num=None):
    """
    Plot the best solution method for a test case in a clean, large format.
    
    Parameters
    ----------
    results : dict
        Dictionary returned by test_optimizers.
    best_method : str
        Name of the best method to plot.
    s_d : float
        Drone speed (used to compute arrival times from path distances).
    test_num : int, optional
        Test case number for the title.
    """
    u  = results[best_method]['u']
    v  = results[best_method]['v']
    X  = results[best_method]['X']
    Y  = results[best_method]['Y']
    sx = results[best_method]['sx']
    sy = results[best_method]['sy']
    f_val = results[best_method]['f']
    J  = len(X)

    # Arrival times at each cluster node
    du = np.diff(u)
    dv = np.diff(v)
    ell = np.sqrt(du**2 + dv**2)
    cumulative_dist = np.concatenate([[0], np.cumsum(ell)])
    t_arrival = cumulative_dist[1:J+1] / s_d
    t_final   = cumulative_dist[-1] / s_d

    X_init    = X
    Y_init    = Y
    X_arrival = X + sx * t_arrival
    Y_arrival = Y + sy * t_arrival
    X_final   = X + sx * t_final
    Y_final   = Y + sy * t_final

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot drone path
    ax.plot(u, v, color='darkblue', label='Drone path', lw=2.5)

    # Dotted lines connecting the three cluster positions
    for j in range(J):
        ax.plot(
            [X_init[j], X_arrival[j], X_final[j]],
            [Y_init[j], Y_arrival[j], Y_final[j]],
            'k:', lw=1, alpha=0.6
        )

    # Plot cluster head positions
    ax.scatter(X_init,    Y_init,    color='green', marker='o', s=100, zorder=5,
               label='Initial position', edgecolors='darkgreen', linewidths=1.5)
    ax.scatter(X_arrival, Y_arrival, color='red',   marker='s', s=100, zorder=5,
               label='Position at drone arrival', edgecolors='darkred', linewidths=1.5)
    ax.scatter(X_final,   Y_final,   color='cyan',  marker='^', s=100, zorder=5,
               label='Final position', edgecolors='darkcyan', linewidths=1.5)
    ax.scatter([0], [0], color='black', marker='*', s=300, zorder=5,
               label='Drone base', edgecolors='black', linewidths=1)

    title_str = f"Best Solution: {best_method} (f = {f_val:.4f})"
    if test_num is not None:
        title_str = f"Test Case {test_num}: {title_str}"
    
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (distance)', fontsize=12)
    ax.set_ylabel('Y (distance)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def plot_monte_carlo_paths(results, s_d):
    """Plot drone trajectories for 10 Monte Carlo optimizer variants.

    This is a thin wrapper around plot_optimizers_paths; Monte Carlo uses the
    same 10 method labels as in `monte_carlo.py`.
    """
    return plot_optimizers_paths(results, s_d)
