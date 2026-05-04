# monte_carlo.py
"""
Monte Carlo evaluation of drone path optimization algorithms.

Loops over a list of scenarios (each defined by L, sMax, nCH). For each
scenario, generates nTest random test cases, runs all optimizer variants,
and saves results and plots.

Usage
-----
    python monte_carlo.py

The ten methods (indices 0-9) are:
    0  FinalXY
    1  FinalXY_Greed
    2  FinalUV
    3  Predicted Rolling
    4  Greedy Rolling
    5  FinalXY Rev
    6  FinalXY_Greed Rev
    7  FinalUV Rev
    8  Predicted Rolling Rev
    9  Greedy Rolling Rev
"""
import os
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from comparison import test_optimizers
from visualization import plot_monte_carlo_paths


# =============================================================================
# Parameters
# =============================================================================
nTest          = 1      # number of random test cases per scenario
dInit          = 1000    # max distance from origin for cluster head placement
s_d            = 1.0     # drone speed
p              = 2.0     # path-loss exponent
seed           = 53      # RNG seed for reproducibility (first set has seed 33, second 43, third 53)
dtau           = 0.1   # pseudo-time step
n_steps        = 2000     # optimizer steps per solve
tsp_time_limit = 1       # OR-Tools GLS time limit per TSP call (seconds)

# Scenario parameter arrays — the m'th entry defines the m'th scenario
L_values    = [8000,4000,16000,8000,8000,6000]   # path-length constraints
sMax_values = [0.2,0.2,0.2,0.1,0.4,0.2]     # max cluster head speeds
nCH_values  = [20,10,40,20,20,20]      # cluster heads per scenario


# =============================================================================
# Method name list (must match comparison.py / visualization.py order)
# =============================================================================
METHOD_NAMES = [
    'FinalXY',
    'FinalXY_Greed',
    'FinalUV',
    'Predicted Rolling',
    'Greedy Rolling',
    'FinalXY Rev',
    'FinalXY_Greed Rev',
    'FinalUV Rev',
    'Predicted Rolling Rev',
    'Greedy Rolling Rev',
]

COMBINED_NAMES = ['FinalXY', 'FinalXY_Greed', 'FinalUV', 'Predicted Rolling', 'Greedy Rolling']
COMBINED_PAIRS = [
    ('FinalXY',           'FinalXY Rev'),
    ('FinalXY_Greed',     'FinalXY_Greed Rev'),
    ('FinalUV',           'FinalUV Rev'),
    ('Predicted Rolling', 'Predicted Rolling Rev'),
    ('Greedy Rolling',    'Greedy Rolling Rev'),
]

ITERATIVE_METHODS = ['FinalXY', 'FinalXY_Greed', 'FinalUV', 'FinalXY Rev', 'FinalXY_Greed Rev', 'FinalUV Rev']


# =============================================================================
# Helpers
# =============================================================================

def _scenario_tag(L, sMax, nCH):
    """Return a short string identifying a scenario's key parameters."""
    return f"L{L:.0f}_sMax{sMax:.2f}_nCH{nCH}"


def _halton(n, base):
    """Return the n-th element of the Halton low-discrepancy sequence (1-indexed)."""
    result, f, i = 0.0, 1.0 / base, n
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _halton_disk(n_points, radius, offset=0):
    """
    Generate n_points quasi-random 2-D points uniformly distributed inside
    a disk of the given radius, using a Halton sequence.
    """
    h1 = np.array([_halton(offset + i + 1, 2) for i in range(n_points)])
    h2 = np.array([_halton(offset + i + 1, 3) for i in range(n_points)])
    r     = radius * np.sqrt(h1)
    theta = 2.0 * np.pi * h2
    return r * np.cos(theta), r * np.sin(theta)


# =============================================================================
# Monte Carlo runner (single scenario)
# =============================================================================
def run_monte_carlo(
    nTest=5,
    nCH=10,
    dInit=5.0,
    sMax=0.25,
    L=10.0,
    s_d=1.0,
    p=4.0,
    seed=42,
    dtau=0.5,
    n_steps=500,
    tsp_time_limit=20,
    out_dir='.',
):
    """
    Run all optimizer variants on nTest randomly generated test cases
    for a single scenario defined by (L, sMax, nCH).

    Parameters
    ----------
    nTest   : int   - number of test cases
    nCH     : int   - cluster heads per test case
    dInit   : float - cluster heads placed within this distance of origin
    sMax    : float - maximum cluster head speed
    L       : float - drone path-length constraint
    s_d     : float - drone speed
    p       : float - path-loss exponent
    seed    : int   - RNG seed
    dtau    : float - pseudo-time step
    n_steps : int   - optimizer steps per solve
    out_dir : str   - directory for output files

    Returns
    -------
    best_f              : ndarray, shape (nTest,)
    best_idx            : list of lists
    all_f               : dict {method_name: ndarray(nTest,)}
    all_reorders        : dict {method_name: ndarray(nTest,)}
    all_times           : dict {method_name: ndarray(nTest,)}
    last_case_results   : dict
    initial_tsp_lengths : ndarray, shape (nTest,)
    initial_objectives  : ndarray, shape (nTest,)
    best_results_list   : list of dicts
    """
    tag = _scenario_tag(L, sMax, nCH)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f'checkpoint_{tag}.pkl')

    # --- Resume from checkpoint if available ---
    rng   = np.random.default_rng(seed)
    reOrd = list(range(nCH - 1))

    best_f_list       = []
    best_idx_list     = []
    all_f_list        = {name: [] for name in METHOD_NAMES}
    all_reorders_list = {name: [] for name in ITERATIVE_METHODS}
    all_times_list    = {name: [] for name in METHOD_NAMES}
    initial_tsp_lengths = []
    initial_objectives  = []
    best_results_list   = []
    last_case_results   = None
    start_i = 0

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as fh:
            ckpt = pickle.load(fh)
        start_i             = ckpt['completed']
        best_f_list         = ckpt['best_f_list']
        best_idx_list       = ckpt['best_idx_list']
        all_f_list          = ckpt['all_f_list']
        all_reorders_list   = ckpt['all_reorders_list']
        all_times_list      = ckpt['all_times_list']
        initial_tsp_lengths = ckpt['initial_tsp_lengths']
        initial_objectives  = ckpt['initial_objectives']
        best_results_list   = ckpt['best_results_list']
        last_case_results   = ckpt['last_case_results']
        # Advance RNG to match state after start_i completed cases
        rng = np.random.default_rng(seed)
        for _ in range(start_i):
            rng.uniform(0.0, sMax, nCH)
            rng.uniform(0.0, 2.0*np.pi, nCH)
        print(f"  Resumed from checkpoint: {start_i}/{nTest} test cases already done.")

    for i in range(start_i, nTest):
        X0_j, Y0_j = _halton_disk(nCH, dInit, offset=i * nCH)

        speeds     = rng.uniform(0.0, sMax,       nCH)
        directions = rng.uniform(0.0, 2.0*np.pi,  nCH)
        s_jx = speeds * np.cos(directions)
        s_jy = speeds * np.sin(directions)

        print(f"\n--- Monte Carlo test {i+1}/{nTest} | nCH={nCH}, L={L:.1f}, sMax={sMax} ---")

        results, initial_tsp_length, f_init = test_optimizers(
            X0_j, Y0_j, s_jx, s_jy,
            s_d, p, L,
            reOrd=reOrd,
            dtau=dtau,
            n_steps=n_steps,
            tsp_time_limit=tsp_time_limit,
        )

        last_case_results = results
        initial_tsp_lengths.append(initial_tsp_length)
        initial_objectives.append(f_init)

        f_values = [results[m]['f'] for m in METHOD_NAMES]
        min_f    = min(f_values)
        winners  = [j for j, fv in enumerate(f_values) if fv <= min_f * 1.0005]
        best_f   = min_f

        best_f_list.append(best_f)
        best_idx_list.append(winners)
        for name, fv in zip(METHOD_NAMES, f_values):
            all_f_list[name].append(fv)
            all_times_list[name].append(results[name]['time'])
        for name in ITERATIVE_METHODS:
            all_reorders_list[name].append(results[name]['reorders'])

        print(f"  Initial TSP path length: {initial_tsp_length:.4f}")
        print(f"  -> Winners: {[METHOD_NAMES[j] for j in winners]},  f = {best_f:.4f}")

        best_results_list.append(results)

        # Save checkpoint after each completed test case
        with open(ckpt_path, 'wb') as fh:
            pickle.dump({
                'completed':          i + 1,
                'best_f_list':        best_f_list,
                'best_idx_list':      best_idx_list,
                'all_f_list':         all_f_list,
                'all_reorders_list':  all_reorders_list,
                'all_times_list':     all_times_list,
                'initial_tsp_lengths': initial_tsp_lengths,
                'initial_objectives': initial_objectives,
                'best_results_list':  best_results_list,
                'last_case_results':  last_case_results,
            }, fh)

    # Remove checkpoint now that the scenario completed successfully
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Save plot_monte_carlo_paths for the last test case
    if last_case_results is not None:
        fig = plot_monte_carlo_paths(last_case_results, s_d)
        fig.savefig(os.path.join(out_dir, f'paths_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    all_f        = {name: np.array(vals) for name, vals in all_f_list.items()}
    all_reorders = {name: np.array(vals) for name, vals in all_reorders_list.items()}
    all_times    = {name: np.array(vals) for name, vals in all_times_list.items()}

    return (
        np.array(best_f_list),
        best_idx_list,
        all_f,
        all_reorders,
        all_times,
        last_case_results,
        np.array(initial_tsp_lengths),
        np.array(initial_objectives),
        best_results_list,
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_monte_carlo_results(best_f, best_idx, all_f, all_reorders,
                             combined_f, combined_best_idx, all_times,
                             initial_objectives, tag='', out_dir='.'):
    """
    Plot and save figures summarising Monte Carlo results for one scenario.

    Figure 1a     - Histogram of best f values across all test cases.
    Figure 1b     - Bar chart of how often each method achieved the best f.
    Figure 1c     - Mean +/- std with 5th/95th percentile lines of (f - best f).
    Figure 1c_pct - Mean +/- std of % improvement over initial ordering.
    Figure 2a     - Combined method win frequency.
    Figure 2b     - Mean +/- std of (f - best f) for combined methods.
    Figure 2b_pct - Mean +/- std of % improvement for combined methods.
    Figure 3      - Reorder count stacked bar.
    Figure 4      - Execution time horizontal bar chart.

    Parameters
    ----------
    best_f             : ndarray, shape (nTest,)
    best_idx           : list of lists
    all_f              : dict {method_name: ndarray(nTest,)}
    all_reorders       : dict {method_name: ndarray(nTest,)}
    combined_f         : dict {combined_name: ndarray(nTest,)}
    combined_best_idx  : ndarray, shape (nTest,)
    all_times          : dict {method_name: ndarray(nTest,)}
    initial_objectives : ndarray, shape (nTest,) - f_init per test case
    tag                : str - scenario identifier appended to filenames
    out_dir            : str - directory for saved figures
    """
    suffix = f'_{tag}' if tag else ''

    def _savefig(fig, name):
        fig.savefig(os.path.join(out_dir, f'{name}{suffix}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 1a - distribution of best f values
    # ------------------------------------------------------------------
    fig1a, ax1a = plt.subplots(figsize=(6, 5))
    ax1a.hist(best_f, bins='auto', color='steelblue', edgecolor='black')
    ax1a.set_xlabel('Best f value')
    ax1a.set_ylabel('Count')
    ax1a.set_title(f'Distribution of Best Objective Values\n{tag}')
    ax1a.grid(True, alpha=0.3)
    _savefig(fig1a, 'fig1a')

    # ------------------------------------------------------------------
    # Figure 1b - how often each method wins (unique vs tied)
    # ------------------------------------------------------------------
    fig1b, ax1b = plt.subplots(figsize=(8, 5))
    unique_wins = np.zeros(len(METHOD_NAMES), dtype=int)
    tied_wins   = np.zeros(len(METHOD_NAMES), dtype=int)
    for winners in best_idx:
        if len(winners) == 1:
            unique_wins[winners[0]] += 1
        else:
            for j in winners:
                tied_wins[j] += 1
    ax1b.bar(range(len(METHOD_NAMES)), unique_wins, color='steelblue', edgecolor='black', label='Unique win')
    ax1b.bar(range(len(METHOD_NAMES)), tied_wins, bottom=unique_wins, color='salmon', edgecolor='black', label='Tied win')
    ax1b.set_xticks(range(len(METHOD_NAMES)))
    ax1b.set_xticklabels(METHOD_NAMES, rotation=45, ha='right', fontsize=8)
    ax1b.set_xlabel('Method')
    ax1b.set_ylabel('Times best')
    ax1b.set_title(f'Best Method Frequency\n{tag}')
    ax1b.legend(fontsize=8)
    ax1b.grid(True, alpha=0.3, axis='y')
    _savefig(fig1b, 'fig1b')

    # ------------------------------------------------------------------
    # Shared data for Figures 1c and 1c_pct
    # ------------------------------------------------------------------
    f_matrix = np.column_stack([all_f[m] for m in METHOD_NAMES])  # (nTest, n_methods)
    xs = np.arange(len(METHOD_NAMES))

    # ------------------------------------------------------------------
    # Figure 1c - mean +/- std with 5th/95th percentile lines of (f - best f)
    # ------------------------------------------------------------------
    fig1c, ax1c = plt.subplots(figsize=(8, 5))
    excess = f_matrix - f_matrix.min(axis=1, keepdims=True)
    means  = excess.mean(axis=0)
    stds   = excess.std(axis=0)
    p05    = np.percentile(excess,  5, axis=0)
    p95    = np.percentile(excess, 95, axis=0)
    ax1c.errorbar(xs, means, yerr=stds, fmt='D', color='steelblue',
                  capsize=5, elinewidth=1.5, markersize=7, label='Mean +/- std')
    for k, (lo, hi) in enumerate(zip(p05, p95)):
        ax1c.plot([k - 0.3, k + 0.3], [lo, lo], color='tomato', lw=1.5)
        ax1c.plot([k - 0.3, k + 0.3], [hi, hi], color='tomato', lw=1.5)
    ax1c.plot([], [], color='tomato', lw=1.5, label='5th / 95th percentile')
    ax1c.set_xticks(xs)
    ax1c.set_xticklabels(METHOD_NAMES, rotation=45, ha='right', fontsize=8)
    ax1c.set_xlabel('Method')
    ax1c.set_ylabel('f - best f')
    ax1c.set_title(f'Mean +/- Std of (f - best f) per Method\n{tag}')
    ax1c.legend(fontsize=8)
    ax1c.grid(True, alpha=0.3, axis='y')
    _savefig(fig1c, 'fig1c')

    # ------------------------------------------------------------------
    # Figure 1c_pct - mean +/- std of % improvement over initial ordering
    # ------------------------------------------------------------------
    fig1c_pct, ax1c_pct = plt.subplots(figsize=(8, 5))
    pct_matrix = (initial_objectives[:, None] - f_matrix) / initial_objectives[:, None] * 100
    pct_means  = pct_matrix.mean(axis=0)
    pct_stds   = pct_matrix.std(axis=0)
    pct_p05    = np.percentile(pct_matrix,  5, axis=0)
    pct_p95    = np.percentile(pct_matrix, 95, axis=0)
    ax1c_pct.errorbar(xs, pct_means, yerr=pct_stds, fmt='D', color='steelblue',
                      capsize=5, elinewidth=1.5, markersize=7, label='Mean +/- std')
    for k, (lo, hi) in enumerate(zip(pct_p05, pct_p95)):
        ax1c_pct.plot([k - 0.3, k + 0.3], [lo, lo], color='tomato', lw=1.5)
        ax1c_pct.plot([k - 0.3, k + 0.3], [hi, hi], color='tomato', lw=1.5)
    ax1c_pct.plot([], [], color='tomato', lw=1.5, label='5th / 95th percentile')
    ax1c_pct.axhline(0, color='black', lw=0.8, linestyle='--')
    ax1c_pct.set_xticks(xs)
    ax1c_pct.set_xticklabels(METHOD_NAMES, rotation=45, ha='right', fontsize=8)
    ax1c_pct.set_xlabel('Method')
    ax1c_pct.set_ylabel('% improvement over initial ordering')
    ax1c_pct.set_title(f'Mean +/- Std of % Improvement over Initial Ordering\n{tag}')
    ax1c_pct.legend(fontsize=8)
    ax1c_pct.grid(True, alpha=0.3, axis='y')
    _savefig(fig1c_pct, 'fig1c_pct')

    # ------------------------------------------------------------------
    # Shared data for Figures 2a, 2b, 2b_pct
    # ------------------------------------------------------------------
    c_matrix = np.column_stack([combined_f[m] for m in COMBINED_NAMES])  # (nTest, n_combined)
    c_min_f  = c_matrix.min(axis=1, keepdims=True)
    cxs      = np.arange(len(COMBINED_NAMES))

    # ------------------------------------------------------------------
    # Figure 2a - combined method win frequency
    # ------------------------------------------------------------------
    fig2a, ax2a = plt.subplots(figsize=(6, 5))
    c_is_winner = (c_matrix == c_min_f)
    c_n_winners = c_is_winner.sum(axis=1)
    c_unique    = (c_is_winner & (c_n_winners[:, None] == 1)).sum(axis=0)
    c_tied      = (c_is_winner & (c_n_winners[:, None] >  1)).sum(axis=0)
    ax2a.bar(cxs, c_unique, color='darkorange', edgecolor='black', label='Unique win')
    ax2a.bar(cxs, c_tied, bottom=c_unique, color='gold', edgecolor='black', label='Tied win')
    ax2a.set_xticks(cxs)
    ax2a.set_xticklabels(COMBINED_NAMES, rotation=30, ha='right', fontsize=9)
    ax2a.set_xlabel('Method')
    ax2a.set_ylabel('Times best')
    ax2a.set_title(f'Best Combined Method Frequency\n{tag}')
    ax2a.legend(fontsize=8)
    ax2a.grid(True, alpha=0.3, axis='y')
    _savefig(fig2a, 'fig2a')

    # ------------------------------------------------------------------
    # Figure 2b - mean +/- std of (f - best f) for combined methods
    # ------------------------------------------------------------------
    fig2b, ax2b = plt.subplots(figsize=(6, 5))
    c_excess = c_matrix - c_min_f
    c_means  = c_excess.mean(axis=0)
    c_stds   = c_excess.std(axis=0)
    c_p05    = np.percentile(c_excess,  5, axis=0)
    c_p95    = np.percentile(c_excess, 95, axis=0)
    ax2b.errorbar(cxs, c_means, yerr=c_stds, fmt='D', color='steelblue',
                  capsize=5, elinewidth=1.5, markersize=7, label='Mean +/- std')
    for k, (lo, hi) in enumerate(zip(c_p05, c_p95)):
        ax2b.plot([k - 0.3, k + 0.3], [lo, lo], color='tomato', lw=1.5)
        ax2b.plot([k - 0.3, k + 0.3], [hi, hi], color='tomato', lw=1.5)
    ax2b.plot([], [], color='tomato', lw=1.5, label='5th / 95th percentile')
    ax2b.set_xticks(cxs)
    ax2b.set_xticklabels(COMBINED_NAMES, rotation=30, ha='right', fontsize=9)
    ax2b.set_xlabel('Method')
    ax2b.set_ylabel('f - best f')
    ax2b.set_title(f'Mean +/- Std of (f - best f) per Combined Method\n{tag}')
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3, axis='y')
    _savefig(fig2b, 'fig2b')

    # ------------------------------------------------------------------
    # Figure 2b_pct - mean +/- std of % improvement for combined methods
    # ------------------------------------------------------------------
    fig2b_pct, ax2b_pct = plt.subplots(figsize=(6, 5))
    c_pct_matrix = (initial_objectives[:, None] - c_matrix) / initial_objectives[:, None] * 100
    c_pct_means  = c_pct_matrix.mean(axis=0)
    c_pct_stds   = c_pct_matrix.std(axis=0)
    c_pct_p05    = np.percentile(c_pct_matrix,  5, axis=0)
    c_pct_p95    = np.percentile(c_pct_matrix, 95, axis=0)
    ax2b_pct.errorbar(cxs, c_pct_means, yerr=c_pct_stds, fmt='D', color='steelblue',
                      capsize=5, elinewidth=1.5, markersize=7, label='Mean +/- std')
    for k, (lo, hi) in enumerate(zip(c_pct_p05, c_pct_p95)):
        ax2b_pct.plot([k - 0.3, k + 0.3], [lo, lo], color='tomato', lw=1.5)
        ax2b_pct.plot([k - 0.3, k + 0.3], [hi, hi], color='tomato', lw=1.5)
    ax2b_pct.plot([], [], color='tomato', lw=1.5, label='5th / 95th percentile')
    ax2b_pct.axhline(0, color='black', lw=0.8, linestyle='--')
    ax2b_pct.set_xticks(cxs)
    ax2b_pct.set_xticklabels(COMBINED_NAMES, rotation=30, ha='right', fontsize=9)
    ax2b_pct.set_xlabel('Method')
    ax2b_pct.set_ylabel('% improvement over initial ordering')
    ax2b_pct.set_title(f'Mean +/- Std of % Improvement over Initial Ordering (Combined)\n{tag}')
    ax2b_pct.legend(fontsize=8)
    ax2b_pct.grid(True, alpha=0.3, axis='y')
    _savefig(fig2b_pct, 'fig2b_pct')

    # ------------------------------------------------------------------
    # Figure 3 - reorder count stacked bar
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    all_vals = np.concatenate([all_reorders[n] for n in ITERATIVE_METHODS])
    lo_r, hi_r = int(all_vals.min()), int(all_vals.max())
    reorder_values = np.arange(lo_r, hi_r + 1)
    cmap = plt.get_cmap('Blues')
    seg_colors = [cmap(0.25 + 0.75 * v / max(hi_r, 1)) for v in reorder_values]
    xr = np.arange(len(ITERATIVE_METHODS))
    bottoms = np.zeros(len(ITERATIVE_METHODS))
    for v, seg_color in zip(reorder_values, seg_colors):
        heights = np.array([np.sum(all_reorders[n] == v) for n in ITERATIVE_METHODS])
        ax3.bar(xr, heights, bottom=bottoms, color=seg_color,
                edgecolor='black', linewidth=0.5, label=str(v))
        bottoms += heights
    ax3.set_xticks(xr)
    ax3.set_xticklabels(ITERATIVE_METHODS, rotation=45, ha='right', fontsize=8)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Reorder Count Distribution per Method\n{tag}')
    ax3.legend(title='# reorders', fontsize=8, title_fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    _savefig(fig3, 'fig3')

    # ------------------------------------------------------------------
    # Figure 4 - execution time per method
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 7))
    t_means = np.array([all_times[m].mean() for m in METHOD_NAMES])
    t_std   = np.array([all_times[m].std()  for m in METHOD_NAMES])
    y_pos   = np.arange(len(METHOD_NAMES))
    ax4.barh(y_pos, t_means, xerr=t_std,
             color='steelblue', edgecolor='black',
             capsize=4, error_kw={'elinewidth': 1.5}, alpha=0.85)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(METHOD_NAMES, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel('Wall-clock time (s)')
    ax4.set_title(f'Mean +/- Std Execution Time per Method\n{tag}')
    ax4.grid(True, alpha=0.3, axis='x')
    _savefig(fig4, 'fig4')


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    # Validate scenario arrays
    n_scenarios = len(L_values)
    assert len(sMax_values) == n_scenarios and len(nCH_values) == n_scenarios, \
        "L_values, sMax_values, and nCH_values must all have the same length."

    # Save all scenario parameters to a single JSON file
    all_scenario_params = [
        {
            'scenario': m,
            'tag': _scenario_tag(L_values[m], sMax_values[m], nCH_values[m]),
            'L': L_values[m],
            'sMax': sMax_values[m],
            'nCH': nCH_values[m],
        }
        for m in range(n_scenarios)
    ]
    params_path = os.path.join(out_dir, 'scenario_params.json')
    with open(params_path, 'w') as fh:
        json.dump({
            'nTest': nTest,
            'dInit': dInit,
            's_d': s_d,
            'p': p,
            'seed': seed,
            'dtau': dtau,
            'n_steps': n_steps,
            'tsp_time_limit': tsp_time_limit,
            'scenarios': all_scenario_params,
        }, fh, indent=2)
    print(f"Saved scenario parameters to {params_path}")

    # ------------------------------------------------------------------
    # Run each scenario
    # ------------------------------------------------------------------
    for m in range(n_scenarios):
        L    = L_values[m]
        sMax = sMax_values[m]
        nCH  = nCH_values[m]
        tag  = _scenario_tag(L, sMax, nCH)

        print(f"\n{'='*70}")
        print(f"SCENARIO {m+1}/{n_scenarios}: {tag}")
        print(f"{'='*70}")

        (best_f, best_idx, all_f, all_reorders, all_times,
         last_case_results, initial_tsp_lengths, initial_objectives,
         best_results_list) = run_monte_carlo(
            nTest=nTest,
            nCH=nCH,
            dInit=dInit,
            sMax=sMax,
            L=L,
            s_d=s_d,
            p=p,
            seed=seed,
            dtau=dtau,
            n_steps=n_steps,
            tsp_time_limit=tsp_time_limit,
            out_dir=out_dir,
        )

        # Combined forward+reverse: element-wise minimum per test case
        combined_f = {
            name: np.minimum(all_f[fwd], all_f[rev])
            for name, (fwd, rev) in zip(COMBINED_NAMES, COMBINED_PAIRS)
        }
        combined_matrix   = np.column_stack([combined_f[n] for n in COMBINED_NAMES])
        combined_best_idx = np.argmin(combined_matrix, axis=1)

        # ----------------------------------------------------------------
        # Table 0: Initial TSP / objective
        # ----------------------------------------------------------------
        print(f"\n=== INITIAL TSP PATH LENGTHS  [{tag}] ===")
        print(f"Mean initial TSP path length: {initial_tsp_lengths.mean():.4f}")
        print(f"Std dev:                      {initial_tsp_lengths.std():.4f}")

        print(f"\n=== INITIAL ORDERING OBJECTIVES  [{tag}] ===")
        print(f"Mean initial objective (f):   {initial_objectives.mean():.4f}")
        print(f"Std dev:                      {initial_objectives.std():.4f}")

        # ----------------------------------------------------------------
        # Table 1: Individual methods
        # ----------------------------------------------------------------
        print(f"\n=== MONTE CARLO SUMMARY (individual methods)  [{tag}] ===")
        print(f"{'Method':25s}  {'Wins':>5s}  {'Mean f':>10s}  {'Std f':>10s}  {'Mean % Improv':>14s}")
        counts = np.zeros(len(METHOD_NAMES), dtype=int)
        for winners in best_idx:
            for j in winners:
                counts[j] += 1
        for idx, name in enumerate(METHOD_NAMES):
            pct = ((initial_objectives - all_f[name]) / initial_objectives * 100).mean()
            print(f"  [{idx}] {name:22s}  {counts[idx]:5d}  "
                  f"{all_f[name].mean():10.4f}  {all_f[name].std():10.4f}  {pct:14.2f}")
        print(f"\nBest f  -  mean: {best_f.mean():.4f},  "
              f"std: {best_f.std():.4f},  "
              f"min: {best_f.min():.4f},  "
              f"max: {best_f.max():.4f}")

        # ----------------------------------------------------------------
        # Table 2: Combined methods
        # ----------------------------------------------------------------
        print(f"\n=== MONTE CARLO SUMMARY (combined fwd+rev)  [{tag}] ===")
        print(f"{'Method':25s}  {'Wins':>5s}  {'Mean f':>10s}  {'Std f':>10s}  {'Mean % Improv':>14s}")
        comb_counts = np.bincount(combined_best_idx, minlength=len(COMBINED_NAMES))
        for idx, name in enumerate(COMBINED_NAMES):
            pct = ((initial_objectives - combined_f[name]) / initial_objectives * 100).mean()
            print(f"  [{idx}] {name:22s}  {comb_counts[idx]:5d}  "
                  f"{combined_f[name].mean():10.4f}  {combined_f[name].std():10.4f}  {pct:14.2f}")

        # ----------------------------------------------------------------
        # Table 3: Reorder counts
        # ----------------------------------------------------------------
        print(f"\n=== REORDER COUNTS (iterative methods)  [{tag}] ===")
        print(f"{'Method':25s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>5s}  {'Max':>5s}")
        for name in ITERATIVE_METHODS:
            r = all_reorders[name]
            print(f"  {name:23s}  {r.mean():8.2f}  {r.std():8.2f}  {r.min():5d}  {r.max():5d}")

        # ----------------------------------------------------------------
        # Table 4: Execution times
        # ----------------------------------------------------------------
        print(f"\n=== EXECUTION TIMES  [{tag}] ===")
        print(f"{'Method':25s}  {'Mean(s)':>8s}  {'Std(s)':>8s}")
        for name in METHOD_NAMES:
            t = all_times[name]
            print(f"  {name:23s}  {t.mean():8.3f}  {t.std():8.3f}")

        # ----------------------------------------------------------------
        # Save results
        # ----------------------------------------------------------------
        pkl_path = os.path.join(out_dir, f'results_{tag}.pkl')
        with open(pkl_path, 'wb') as fh:
            pickle.dump({
                'tag': tag,
                'L': L, 'sMax': sMax, 'nCH': nCH,
                'best_f': best_f,
                'best_idx': best_idx,
                'all_f': all_f,
                'all_reorders': all_reorders,
                'combined_f': combined_f,
                'combined_best_idx': combined_best_idx,
                'all_times': all_times,
                'initial_tsp_lengths': initial_tsp_lengths,
                'initial_objectives': initial_objectives,
            }, fh)
        print(f"\nSaved: {pkl_path}")

        try:
            npz_path = os.path.join(out_dir, f'results_{tag}.npz')
            np.savez(npz_path,
                     best_f=best_f,
                     best_idx=np.array(best_idx, dtype=object),
                     combined_best_idx=combined_best_idx,
                     initial_tsp_lengths=initial_tsp_lengths,
                     initial_objectives=initial_objectives)
            print(f"Saved: {npz_path}")
        except Exception as e:
            print(f'Warning: np.savez failed, but pickle saved results: {e}')

        # ----------------------------------------------------------------
        # Performance plots (all saved to out_dir with scenario tag)
        # ----------------------------------------------------------------
        plot_monte_carlo_results(
            best_f, best_idx, all_f, all_reorders,
            combined_f, combined_best_idx, all_times,
            initial_objectives, tag=tag, out_dir=out_dir,
        )

    print(f"\nAll results saved in: {os.path.abspath(out_dir)}/")
