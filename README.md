# Moving Target Drone Tracking Simulation

This project extends a baseline drone tracking simulation that originally used stationary cluster heads. The updated version introduces **moving targets**, enabling evaluation of drone tracking performance under dynamic conditions.

---

## Overview

This simulation models drone-based tracking of multiple targets in a 2D environment. Unlike the original implementation (which assumed stationary cluster heads), this version introduces target mobility, allowing for more realistic analysis of tracking performance under dynamic behavior.

The goal is to study how motion affects:
- Tracking accuracy
- Path efficiency
- Coverage performance over time

---

## Key Modification from Original Model

Original system:
- Stationary cluster heads
- Static environment assumptions

Updated system:
- Moving targets with configurable motion models
- Continuous repositioning of targets over time
- Dynamic tracking requirements for drones

---

## Features

- Multi-target simulation environment
- Moving target dynamics (speed and direction configurable)
- Drone path tracking logic
- Performance evaluation metrics
- Visualization of trajectories (if applicable)
- Adjustable simulation parameters

---

## Repository Structure

```text
src/              explain
models/           explain
utils/            explain
results/          explain
main.py           explain
