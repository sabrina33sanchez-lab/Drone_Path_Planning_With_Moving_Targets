# kinematics.py
"""
Cluster head kinematics.

Provides functions describing how each cluster head moves over time.
All cluster heads are assumed to move with constant velocity.
"""

import numpy as np


def Xj(X0_j, s_jx, t_j):
    """
    Return cluster head x-coordinate(s) at time(s) t_j.

    Vectorized: supports arrays for positions, velocities and times.

    Parameters
    ----------
    X0_j : float or ndarray
        Initial x-position(s).
    s_jx : float or ndarray
        x-velocity(ies).
    t_j : float or ndarray
        Time(s) at which to evaluate position.

    Returns
    -------
    float or ndarray
        x-coordinate(s) at time t_j.
    """
    return X0_j + s_jx * t_j


def Yj(Y0_j, s_jy, t_j):
    """
    Return cluster head y-coordinate(s) at time(s) t_j.

    Vectorized: supports arrays for positions, velocities and times.

    Parameters
    ----------
    Y0_j : float or ndarray
        Initial y-position(s).
    s_jy : float or ndarray
        y-velocity(ies).
    t_j : float or ndarray
        Time(s) at which to evaluate position.

    Returns
    -------
    float or ndarray
        y-coordinate(s) at time t_j.
    """
    return Y0_j + s_jy * t_j


def Xj_prime(s_jx):
    """
    Return the time-derivative of Xj (i.e. the x-velocity).

    Parameters
    ----------
    s_jx : float or ndarray
        x-velocity(ies).

    Returns
    -------
    float or ndarray
        x-velocity (same as input since motion is constant-speed).
    """
    return s_jx


def Yj_prime(s_jy):
    """
    Return the time-derivative of Yj (i.e. the y-velocity).

    Parameters
    ----------
    s_jy : float or ndarray
        y-velocity(ies).

    Returns
    -------
    float or ndarray
        y-velocity (same as input since motion is constant-speed).
    """
    return s_jy
