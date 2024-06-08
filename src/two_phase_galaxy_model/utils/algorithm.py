from __future__ import annotations
from numba import njit
import numpy as np
import numbers


@njit
def bisearch(x_sorted: np.ndarray, x_dst: numbers.Real) -> int:
    '''
    Return idx in [-1, len(x_sorted)), 
    so that x_sorted[idx] <= x_dst < x_sorted[idx+1].
    
    Requirements:
    - x_sorted is sorted in ascending order.
    - len(x_sorted) > 0.
    '''
    n_all = len(x_sorted)
    assert n_all > 0
    if x_dst < x_sorted[0]:
        return -1
    l, r = 0, n_all
    while r - l > 1:
        c = (l + r) // 2
        if x_dst < x_sorted[c]:
            r = c
        else:
            l = c
    return l


@njit
def bisearch_nearest(x_sorted: np.ndarray, x_dst: numbers.Real) -> int:
    '''
    Return idx in [0, len(x_sorted)), so that x_sorted[idx] is the nearest 
    to x_dst among all in x_sorted.
    
    Requirements: the same as bisearch(x_sorted, x_dst).
    '''
    l = bisearch(x_sorted, x_dst)
    r = l + 1
    n = len(x_sorted)
    if l < 0:
        return 0
    if r > n-1:
        return n-1
    dx_l = x_dst - x_sorted[l]
    dx_r = x_sorted[r] - x_dst
    return l if dx_l < dx_r else r


@njit
def bisearch_nearest_array(x_sorted: np.ndarray, x_dst: np.ndarray) -> int:
    idx = np.empty(x_dst.size, dtype=np.int64)
    for i, _x_dst in enumerate(x_dst):
        idx[i] = bisearch_nearest(x_sorted, _x_dst)
    return idx


@njit
def bisearch_interp_weight(
    x_sorted: np.ndarray, x_dst: numbers.Real
) -> tuple[int, int, float, float]:

    l = bisearch(x_sorted, x_dst)
    if l < 0:
        return 0, 0, 0.0, 1.0
    r = l + 1
    n = len(x_sorted)
    if r > n-1:
        return n-1, n-1, 1.0, 0.0

    dx_l = x_dst - x_sorted[l]
    dx_r = x_sorted[r] - x_dst
    dx = dx_l + dx_r
    w_l, w_r = dx_r / dx, dx_l / dx

    return l, r, w_l, w_r


@njit
def bisearch_interp(x_sorted: np.ndarray, y: np.ndarray, x_dst: numbers.Real):
    l, r, w_l, w_r = bisearch_interp_weight(x_sorted, x_dst)
    return y[l] * w_l + y[r] * w_r
