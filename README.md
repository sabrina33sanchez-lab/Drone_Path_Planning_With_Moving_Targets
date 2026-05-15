# Dynamic Drone Routing for Moving Cluster Heads

This project extends a baseline drone routing framework for stationary cluster heads by introducing moving targets and dynamic path optimization strategies.

The system studies how a drone acting as a communication relay can efficiently route through moving cluster heads while minimizing communication cost under a constrained total path length.

---

## Overview

The framework combines:

- Time-dependent cluster motion models
- Constrained path optimization
- Dynamic reordering strategies
- Traveling Salesman Problem (TSP) methods
- Monte Carlo evaluation
- Visualization and animation tools

The project evaluates how different routing and reordering methods perform when cluster heads move over time.

---

## Main Features

- Dynamic cluster head motion
- Constrained drone trajectory optimization
- TSP and greedy reordering methods
- Rolling optimization frameworks
- Monte Carlo scenario evaluation
- Animated trajectory visualization
- Comparative optimizer analysis

---

## Repository Structure

### `monecarlo.py`
Primary user-facing execution script.  Generates randomize moving-cluster scenarios and statistically compares optimizer performance across many trials.

Includes:
- Random scenario generation
- Halton sequence sampling
- Runtime analysis
- Histogram/bar-chart generation
- Best-method tracking

---

### `geometry.py`
Provides geometric utilities for drone path construction and path-length projection.

Main functionality:
- Path length computation
- Initial path generation from cluster heads
- Projection onto fixed path-length constraints

---

### `kinematics.py`
Implements constant-velocity cluster head motion models.

Cluster positions evolve according to:

```math id="e9w8ev"
x(t)=x_0+v_xt,\qquad y(t)=y_0+v_yt
```

---

### `optimization.py`
Core constrained optimization engine.

Implements:
- Objective gradients
- Path-length constraint gradients
- Pseudo-time optimizaiton dynamices
This modeule performs the actual trajectory optimizaiton.

---

### `routing.py`
Implements routing and reordering strategies.

Includes:
- Exact TSP solvers
- Greedy predicted-position reorder
- Rolling optimization frameworks
- Iterative re-optimizaiton methods
This module coordinates how the drone chooses visitation orderings.

---

### `metrics.py`
Computes performance metrics for optimized trajectories.

Metrics include:
- Total path length
- Communication objective values
- Dynamic energy cost calcualtions
- Moving cluster evaluations
Used for quantitative comparison of optimization strategies.

---

### `visualization.py`
Visualization and animation utilities.

Provides:
- Multi-panel optimization plots
- Final path visualizations
- Arrival-order labeling
- Animated drone/cluster trajectories
- Side-by-side optimizer comparisons
Supports GIF exports for simulation playback.

---

### `comparison.py`
Framework for evaluating optimizer performance across scenarios.

Performs:
- Multi-method solver comparisons
- Objective and energy evaluation
- Runtime benchmarking
- Reorder statistics collection
- Forward/reverse ordering comparisons

---

### `main.py`

Responsible for:
- Configuring simulation parameters
- Defining moving cluster scenarios
- Running optimizer comparisons
- Generating visualizations

---

## Optimization Objective

The drone seeks to minimize communication cost while maintaining a constrained total trajectory length.  
Cluster positions are evaluated dynamically at estimated drone arrival times, making the optimization time-dependent.

---

## Example Workflow

1. Generate moving cluster head scenarios
2. Initialize feasible drone trajectory
3. Optimize trajectory under length constraints
4. Reorder cluster visitation dynamically
5. Evaluate objective and energy metrics
6. Visualize and compare optimizer performance

---

## Research Motivation

This framework models dynamic drone-assisted communication and routing problems relevant to:
- UAV communication relays
- Autonomous surveillance
- Mobile sensor networks
- Dynamic routing optimization
- Time-dependent TSP variants

---

## Future Work

Potential extensions include:
- Multi-drone coordination
- Reinforcement learning control
- Obstacle-aware planning
- Nonlinear motion prediction
- Real-time adaptive routing
- 3D trajectory optimization

---

# Authors

Sabrina Keller TAMUCT   
Christopher Thron TAMUCT
