from __future__ import annotations
import numpy as np
import numba
from numba.experimental import jitclass
from .algorithm import bisearch_interp
from pyhipp.stats.random import _Rng as Rng


@jitclass
class PowerLaw:
    '''
    @beta: cannot be -1.
    '''
    beta: numba.float64
    x1: numba.float64
    x2: numba.float64

    _bp1: numba.float64
    _ibp1: numba.float64
    _y1: numba.float64
    _y2: numba.float64
    _dy: numba.float64
    _norm: numba.float64

    def __init__(self, beta: float, x1: float, x2: float) -> None:

        self.beta = beta
        self.x1 = x1
        self.x2 = x2
        self.set_up()

    def set_up_range(self, x1: float, x2: float) -> None:
        self.x1 = x1
        self.x2 = x2
        self.set_up()

    def set_up(self) -> None:
        bp1 = self.beta + 1
        ibp1 = 1. / bp1
        y1, y2 = self.x1 ** bp1, self.x2 ** bp1
        dy = y2 - y1
        assert dy != 0
        norm = bp1 / dy

        self._bp1 = bp1
        self._ibp1 = ibp1
        self._y1 = y1
        self._y2 = y2
        self._dy = dy
        self._norm = norm

    def p2q(self, p: np.ndarray):
        y = p * self._dy + self._y1
        q = y ** self._ibp1
        return q

    def rv(self, rng: Rng, size=None):
        u = rng.random(size=size)
        q = self.p2q(u)
        return q

    def pdf(self, q: np.ndarray):
        pdf = self._norm * q ** self.beta
        return pdf

    def pdf_unnorm(self, q: np.ndarray):
        pdf = q ** self.beta
        return pdf


@jitclass
class WeightedPowerLaw:

    dist: PowerLaw
    dist_p: PowerLaw

    def __init__(self, beta: float, beta_p: float,
                 x1: float, x2: float) -> None:

        self.dist = PowerLaw(beta, x1, x2)
        self.dist_p = PowerLaw(beta_p, x1, x2)

    def set_up_range(self, x1: float, x2: float) -> None:
        self.dist.set_up_range(x1, x2)
        self.dist_p.set_up_range(x1, x2)

    def p2q(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Return quantile, weight.'''
        q = self.dist_p.p2q(p)
        w = self.dist.pdf(q) / self.dist_p.pdf(q)
        return q, w

    def rv(self, rng: Rng, size=None):
        '''Return quantile, weight.'''
        u = rng.random(size=size)
        q, w = self.p2q(u)
        return q, w


@jitclass
class ExpPowerLaw:
    '''
    Sampling for x, where x = ln(t), t is power-law distributed as t^beta.
    @beta: cannot be -1.
    '''
    beta: numba.float64
    x1: numba.float64
    x2: numba.float64

    _bp1: numba.float64
    _ibp1: numba.float64
    _y1: numba.float64
    _y2: numba.float64
    _dy: numba.float64
    _norm: numba.float64

    def __init__(self, beta: float, x1: float, x2: float) -> None:

        self.beta = beta
        self.x1 = x1
        self.x2 = x2
        self.set_up()

    def set_up_range(self, x1: float, x2: float) -> None:
        self.x1 = x1
        self.x2 = x2
        self.set_up()

    def set_up(self) -> None:
        bp1 = self.beta + 1
        ibp1 = 1. / bp1
        y1, y2 = np.exp(bp1 * self.x1), np.exp(bp1 * self.x2)
        dy = y2 - y1
        assert dy != 0
        norm = bp1 / dy

        self._bp1 = bp1
        self._ibp1 = ibp1
        self._y1 = y1
        self._y2 = y2
        self._dy = dy
        self._norm = norm

    def p2q(self, p: np.ndarray):
        y = p * self._dy + self._y1
        q = np.log(y) * self._ibp1
        return q

    def rv(self, rng: Rng, size=None):
        u = rng.random(size=size)
        q = self.p2q(u)
        return q

    def pdf(self, q: np.ndarray):
        pdf = self._norm * np.exp(self._bp1 * q)
        return pdf

    def pdf_unnorm(self, q: np.ndarray):
        pdf = np.exp(self._bp1 * q)
        return pdf


@jitclass
class InterpolatedDist:

    _x_edges: numba.float64[:]
    _pdf: numba.float64[:]
    _cdf: numba.float64[:]

    def __init__(self, x_edges: np.ndarray, pdf: np.ndarray) -> None:
        '''
        @pdf: un-normalized pdf.
        '''
        n_edges = len(x_edges)
        assert n_edges == len(pdf)

        cdf = np.empty(n_edges, dtype=np.float64)
        cdf[0] = 0
        for i in range(1, n_edges):
            dx = x_edges[i] - x_edges[i-1]
            p = 0.5 * (pdf[i] + pdf[i-1])
            cdf[i] = p * dx + cdf[i-1]

        norm = cdf[-1]
        cdf = cdf / norm
        pdf = pdf / norm

        self._x_edges = x_edges
        self._pdf = pdf
        self._cdf = cdf

    def p2q(self, p: float):
        return bisearch_interp(self._cdf, self._x_edges, p)

    def rv(self, rng: Rng):
        u = rng.random()
        q = self.p2q(u)
        return q

    def pdf(self, q: float):
        return bisearch_interp(self._x_edges, self._pdf, q)
