from __future__ import annotations
import typing
from typing import Self, Optional, Literal
from ..recipes.abc import default_ctx
from .abc import Model, Parameter
from .subhalos import Branch, GroupTree, SubhaloTree
from ..utils.sampling import Rng as JitRng
import numba
from numba.experimental import jitclass
import numpy as np


class Initializer(Model):
    def __init__(self, bh_m_h_min=1.0e-1, bh_m_scale=0.0, bh_m_min=1.0e-9,
                 star_eps=0.01,
                 **kw):

        super().__init__(**kw)

        self.bh_m_h_min = Parameter(bh_m_h_min)
        self.bh_m_scale = Parameter(bh_m_scale)
        self.bh_m_min = Parameter(bh_m_min)
        self.star_eps = Parameter(star_eps)

    def __call__(self, b: Branch) -> None:
        r'''
        m_h -> 
        all: bh_seeded, m_bh, 
        first: m_g, m_s
        '''
        self.__seed_bh(
            b.data,
            self.bh_m_h_min.value.item(),
            self.bh_m_scale.value.item(),
            self.bh_m_min.value.item()
        )
        self.__seed_baryon(
            b.data,
            self.star_eps.value.item(),
            self.ctx.cosm.baryon_fraction0,
        )

    @staticmethod
    @numba.njit
    def __seed_bh(hs, bh_m_h_min, bh_m_scale, bh_m_min):
        seeded, m_bh = False, 0.
        for h in hs:
            if not seeded and h.m_h >= bh_m_h_min:
                seeded, m_bh = True, max(h.m_h * bh_m_scale, bh_m_min)
            h.bh_seeded, h.m_bh = seeded, m_bh

    @staticmethod
    @numba.njit
    def __seed_baryon(hs, star_eps, f_b) -> None:
        assert len(hs) > 0
        h0 = hs[0]
        m_b = h0.m_h * f_b
        h0.m_g = (1. - star_eps) * m_b
        h0.m_s = star_eps * m_b


@jitclass
class _FastPhase:
    '''
    Parameters all in internal units.
    '''
    m_h_cool: numba.float64
    beta_cool: numba.float64

    alpha_acc: numba.float64
    alpha_en: numba.float64
    m_h_en: numba.float64
    beta_en: numba.float64

    eps_s: numba.float64

    alpha_sn: numba.float64
    beta_sn: numba.float64
    v_w: numba.float64
    f_ej_sn: numba.float64

    alpha_agn: numba.float64
    c: numba.float64
    f_ej_agn: numba.float64

    f_b: numba.float64

    def __init__(
            self,
            m_h_cool, beta_cool, alpha_acc, alpha_en, m_h_en,
            beta_en, eps_s, alpha_sn, beta_sn, v_w,
            f_ej_sn, alpha_agn, c, f_ej_agn, f_b):

        self.m_h_cool = m_h_cool
        self.beta_cool = beta_cool
        self.alpha_acc = alpha_acc
        self.alpha_en = alpha_en
        self.m_h_en = m_h_en

        self.beta_en = beta_en
        self.eps_s = eps_s
        self.alpha_sn = alpha_sn
        self.beta_sn = beta_sn
        self.v_w = v_w

        self.f_ej_sn = f_ej_sn
        self.alpha_agn = alpha_agn
        self.c = c
        self.f_ej_agn = f_ej_agn
        self.f_b = f_b

    def steps(self, hs):
        '''
        @hs: fast phase subhalos.
        
        All hs: m_h, v_max, dm_h, dt.
        Initialized m_bh, bh_seeded, m_g, m_s.
        -> 
        All gas, star, dm properties, except r_sgc.
        
        Must be called after initialization.
        Fill the fast phase subhalos (at least 1).
        '''
        assert len(hs) > 0
        h_prev = hs[0]
        self._step(h_prev)
        for h in hs[1:]:
            h.m_g = h_prev.m_g
            h.m_s = h_prev.m_s
            if h_prev.bh_seeded:
                h.m_bh = h_prev.m_bh
            self._step(h)
            h_prev = h

    def _step(self, h):
        f_cool = self._cooling(h)
        f_en, eps_bh = self._bh_accretion(h)
        eps_s = self.eps_s
        f_sn = self._sn_feedback(h)
        f_agn = self._agn_feedback(h)

        if eps_s + eps_bh > 1.:
            eps_s = min(eps_s, 1.0)
            eps_bh = min(eps_bh, 1.0 - eps_s)

        dm_b = max(h.dm_h * self.f_b, 1.0e-9)
        dm_g_cool = dm_b * f_cool
        dm_g_hot = dm_b * (1.0 - f_cool)
        dm_g_sf = dm_g_cool * f_sn * f_agn
        dm_g_ej = dm_g_cool * (1.0 - f_sn) * self.f_ej_sn + \
            dm_g_cool * f_sn * (1.0 - f_agn) * self.f_ej_agn
        dm_g_prev = dm_g_cool * (1.0 - f_sn) * (1.0 - self.f_ej_sn) + \
            dm_g_cool * f_sn * (1.0 - f_agn) * (1.0 - self.f_ej_agn)

        dm_s = eps_s * dm_g_sf
        dm_bh = eps_bh * dm_g_sf
        dm_g = dm_g_sf * (1.0 - eps_s - eps_bh) + dm_g_prev

        h.f_en = f_en
        h.f_sn = f_sn
        h.f_agn = f_agn
        h.m_g = h.m_g + dm_g
        h.dm_g_hot = dm_g_hot
        h.dm_g_cool = dm_g_cool
        h.dm_g_sf = dm_g_sf
        h.dm_g_ej = dm_g_ej
        h.dm_g_prev = dm_g_prev
        h.dm_g = dm_g

        h.m_s = h.m_s + dm_s
        h.dm_s = dm_s
        h.sfr = dm_s / max(h.dt, 1.0e-6)
        h.m_s_b = h.m_s
        h.m_s_d = 0.

        h.m_bh = h.m_bh + dm_bh
        h.dm_bh = dm_bh

    def _cooling(self, h):
        '''For h: m_h'''
        f_cool = 1.0 / (1.0 + (h.m_h / self.m_h_cool)**self.beta_cool)
        return f_cool

    def _bh_accretion(self, h):
        '''For h: m_h, m_bh, m_g'''
        m = (h.m_h / self.m_h_en)**self.beta_en
        f_en = (self.alpha_en + m)/(1.0 + m)
        eps_bh = self.alpha_acc * (h.m_bh / max(h.m_g, 1.0e-10)) * f_en
        return f_en, eps_bh

    def _sn_feedback(self, h):
        '''For h: v_max'''
        f_sn_min = 1.0e-3
        f1 = (h.v_max / self.v_w)**self.beta_sn
        f_sn = (self.alpha_sn + f1)/(1.0 + f1)
        f_sn = max(f_sn, f_sn_min)
        return f_sn

    def _agn_feedback(self, h):
        '''For h: m_bh, m_g, v_max'''
        f_agn_min = 1.0e-2
        f_agn = 1.0 - self.alpha_agn * \
            (h.m_bh / max(h.m_g, 1.0e-10)) * (self.c/h.v_max)**2
        f_agn = max(f_agn, f_agn_min)
        return f_agn


class FastPhase(Model):
    def __init__(
        self,
        m_h_cool=1.0e3, beta_cool=4., alpha_acc=2.5, alpha_en=3.0,
        m_h_en=10**1.5, beta_en=2.0,
        eps_s=0.75,
        alpha_sn=0., beta_sn=2.5, v_w=250.0, f_ej_sn=0.75,
        alpha_agn=10**-3.0, f_ej_agn=0.75,
        **kw,
    ):
        super().__init__(**kw)

        # cooling
        self.m_h_cool = Parameter(m_h_cool)
        self.beta_cool = Parameter(beta_cool)

        # bh accretion
        self.alpha_acc = Parameter(alpha_acc)
        self.alpha_en = Parameter(alpha_en)
        self.m_h_en = Parameter(m_h_en)
        self.beta_en = Parameter(beta_en)

        # sf
        self.eps_s = Parameter(eps_s)

        # feedbacks
        self.alpha_sn = Parameter(alpha_sn)
        self.beta_sn = Parameter(beta_sn)
        self.v_w = Parameter(v_w)
        self.f_ej_sn = Parameter(f_ej_sn)

        self.alpha_agn = Parameter(alpha_agn)
        self.f_ej_agn = Parameter(f_ej_agn)

        self.fast_phase: Optional[_FastPhase] = None

    def set_up(self) -> None:
        super().set_up()

        kw = {}
        for k, v in self.parameters().items():
            kw[k] = v.value.item()

        cosm = self.ctx.cosm
        us = cosm.unit_system

        kw['f_b'] = float(cosm.baryon_fraction0)
        kw['c'] = float(us.c_light_speed)
        kw['v_w'] = float(self.v_w.value.item() / us.u_v_to_kmps)

        self.fast_phase = _FastPhase(**kw)

    def tear_down(self) -> None:
        self.fast_phase = None
        super().tear_down()


@jitclass
class _SlowPhase:
    '''
    Lu 14, Model II.
    @m_h_lb: Imposed by Lu14 with M_vir >= 2e9 Msun/h for a non-zero SFR.
    '''
    a_0: numba.float64
    a_p: numba.float64
    b_0: numba.float64
    g_a: numba.float64
    lg_eps: numba.float64
    lg_m_c: numba.float64
    lg_r: numba.float64
    sigma_lg_sfr: numba.float64

    f_ej: numba.float64

    m_h_lb: numba.float64
    f_b: numba.float64
    big_h0: numba.float64

    def __init__(
        self, a_0, a_p, b_0, g_a, lg_eps, lg_m_c, lg_r, sigma_lg_sfr,
        f_ej, m_h_lb, f_b, big_h0,
    ) -> None:
        self.a_0 = a_0
        self.a_p = a_p
        self.b_0 = b_0
        self.g_a = g_a
        self.lg_eps = lg_eps
        self.lg_m_c = lg_m_c
        self.lg_r = lg_r
        self.sigma_lg_sfr = sigma_lg_sfr

        self.f_ej = f_ej

        self.m_h_lb = m_h_lb
        self.f_b = f_b
        self.big_h0 = big_h0

    def steps(self, hs, h_prev):
        '''
        @hs: slow phase subhalos.
        @h_prev: last fast phase subhalos.
        
        All hs: m_h, z, dt
        h_prev: m_g, m_s, m_s_b, m_bh, bh_seeded
        ->
        All gas, star, dm properties, except r_sgc.
        
        If not central stage, fill nothing.
        '''
        for h in hs:
            h.m_g = h_prev.m_g
            h.m_s = h_prev.m_s
            h.m_s_b = h_prev.m_s_b
            if h_prev.bh_seeded:
                h.m_bh = h_prev.m_bh
            self._step(h)
            h_prev = h

    def _step(self, h):
        m_h, z, dt = h.m_h, h.z, h.dt

        X = m_h / (10.0**self.lg_m_c)
        Xp1 = 1. + X
        XpR = 10.0**self.lg_r + X
        alpha = self.a_0 * (z+1.) ** self.a_p
        beta = self.b_0
        gamma = self.g_a
        f_cool = Xp1**alpha if m_h >= self.m_h_lb else 0.
        f_sn = (X / XpR)**gamma
        f_agn = (XpR/Xp1)**beta
        f_fb = f_sn * f_agn

        inv_tau = self.big_h0 * 10.0 * (1.0 + z)**1.5
        dm_b = max(m_h * self.f_b * inv_tau * dt, 1.0e-9)

        dm_g_cool = dm_b * f_cool
        dm_g_hot = dm_b * (1.0 - f_cool)
        dm_g_sf = dm_g_cool * f_fb
        dm_g_ej = dm_g_cool * (1.0 - f_fb) * self.f_ej
        dm_g_prev = dm_g_cool * (1.0 - f_fb) * (1.0 - self.f_ej)

        dm_s = 10.0**self.lg_eps * dm_g_sf
        dm_g = dm_g_sf - dm_s + dm_g_prev

        h.f_en = 1.
        h.f_sn = f_sn
        h.f_agn = f_agn
        h.m_g = h.m_g + dm_g
        h.dm_g_hot = dm_g_hot
        h.dm_g_cool = dm_g_cool
        h.dm_g_sf = dm_g_sf
        h.dm_g_ej = dm_g_ej
        h.dm_g_prev = dm_g_prev
        h.dm_g = dm_g

        h.m_s = h.m_s + dm_s
        h.dm_s = dm_s
        h.sfr = dm_s / max(h.dt, 1.0e-6)
        h.m_s_d = h.m_s - h.m_s_b

        h.dm_bh = 0.


class SlowPhase(Model):
    def __init__(
        self,
        a_0=-3.6, a_p=-0.72, b_0=1.8, g_a=1.9,
        lg_eps=-0.27 + np.log10(0.6),
        lg_m_c=1.9, lg_r=-0.96, lsfr_sigma=.3,
        f_ej=1.0, m_h_lb=2.0e-1,
        **kw
    ) -> None:

        super().__init__(**kw)

        self.a_0 = Parameter(a_0)
        self.a_p = Parameter(a_p)
        self.b_0 = Parameter(b_0)
        self.g_a = Parameter(g_a)
        self.lg_eps = Parameter(lg_eps)
        self.lg_m_c = Parameter(lg_m_c)
        self.lg_r = Parameter(lg_r)
        self.sigma_lg_sfr = Parameter(lsfr_sigma)
        self.f_ej = Parameter(f_ej)
        self.m_h_lb = Parameter(m_h_lb)

        self.slow_phase: Optional[_SlowPhase] = None

    def set_up(self) -> None:
        super().set_up()

        kw = {}
        for k, v in self.parameters().items():
            kw[k] = v.value.item()

        cosm = self.ctx.cosm
        f_b, big_h0 = cosm.baryon_fraction0, cosm.big_hubble0
        kw['f_b'] = float(f_b)
        kw['big_h0'] = float(big_h0)

        self.slow_phase = _SlowPhase(**kw)

    def tear_down(self) -> None:
        self.slow_phase = None
        super().tear_down()


@jitclass
class _SatPhase:

    tau_delay: numba.float64
    tau: numba.float64

    '''
    A flattening, followed by an exponential decay, of SFR; and the frozon of 
    m_bh, for satellite galaxies.
    @tau_delay, tau: [Gyr/h]. tau is the decay rate in 10-based logarithmic 
    space.
    '''

    def __init__(self, tau_delay, tau):

        self.tau_delay = tau_delay
        self.tau = tau

    def steps(self, hs, h_inf):
        '''
        @hs: satellite phase subhalos.
        
        All hs: dm_h, dt, t_lb
        h_prev (last central, inf): 
        sfr, t_lb, is_fast, 
        m_g, m_s, m_s_b, m_s_d, m_bh, bh_seeded
        ->
        All gas, star, dm properties, except r_sgc.
        '''
        h_prev = h_inf
        for h in hs:
            h.m_g = h_prev.m_g
            h.m_s = h_prev.m_s
            h.m_s_d = h_prev.m_s_d
            h.m_s_b = h_prev.m_s_b
            if h_prev.bh_seeded:
                h.m_bh = h_prev.m_bh
            self._step(h, h_inf)
            h_prev = h

    def _step(self, h, h_inf):
        sfr_inf, t_lb_inf, is_fast_inf = (h_inf.sfr, h_inf.t_lb, h_inf.is_fast)
        t_lb, dt = h.t_lb, h.dt

        dtau = max(t_lb_inf - t_lb - self.tau_delay, 0.)
        sfr = sfr_inf * 10.0**(-dtau/self.tau)
        dm_s = min(sfr * dt, h.m_g)
        dm_g = -dm_s
        dm_g_cool = dm_s
        dm_g_sf = dm_s

        h.f_en = 1.
        h.f_sn = 1.
        h.f_agn = 1.
        h.m_g = h.m_g + dm_g
        h.dm_g_cool = dm_g_cool
        h.dm_g_hot = 0.
        h.dm_g_ej = 0.
        h.dm_g_prev = 0.
        h.dm_g_sf = dm_g_sf
        h.dm_g = dm_g

        h.m_s = h.m_s + dm_s
        h.dm_s = dm_s
        h.sfr = dm_s / max(dt, 1.0e-6)
        if is_fast_inf:
            h.m_s_b += dm_s
        else:
            h.m_s_d += dm_s

        h.dm_bh = 0.


class SatPhase(Model):
    def __init__(self, tau_delay=0., tau=4., **kw):
        super().__init__(**kw)

        self.tau_delay = Parameter(tau_delay)
        self.tau = Parameter(tau)

        self.sat_phase: Optional[_SatPhase] = None

    def set_up(self) -> None:
        super().set_up()

        kw = {}
        for k, v in self.parameters().items():
            kw[k] = v.value.item()

        self.sat_phase = _SatPhase(**kw)

    def tear_down(self) -> None:
        self.sat_phase = None
        super().tear_down()


@jitclass
class _Integrator:

    f_r_lb: numba.float64
    fast_phase: _FastPhase
    slow_phase: _SlowPhase
    sat_phase: _SatPhase

    def __init__(self, f_r_lb, fast_phase, slow_phase, sat_phase) -> None:

        self.f_r_lb = f_r_lb
        self.fast_phase = fast_phase
        self.slow_phase = slow_phase
        self.sat_phase = sat_phase

    def steps(self, hs, idx_fast, idx_sat):
        idx_slow = idx_fast + 1
        if idx_slow == 0:
            assert idx_sat == 0
            idx_slow = 1
            idx_sat = 1

        self.fast_phase.steps(hs[:idx_slow])
        self.slow_phase.steps(hs[idx_slow:idx_sat], hs[idx_slow-1])
        self.sat_phase.steps(hs[idx_sat:], hs[idx_sat-1])

        for h in hs:
            f_gas = h.m_g / h.m_h
            f_r = max(f_gas, self.f_r_lb)
            h.r_sgc = h.a * h.r_h * f_r


class GasPhaseMZR(Model):

    Source = Literal['Ma16', 'Choksi18', 'New', 'Chen24']

    def __init__(self,
                 src: Source = 'Choksi18',

                 **kw) -> None:

        super().__init__(**kw)

        if src == 'Ma16':
            src = 0
        elif src == 'Choksi18':
            src = 1
        elif src == 'New':
            src = 2
        elif src == 'Chen24':
            src = 3
        else:
            raise ValueError(f'Invalid source: {src}')

        self.src = Parameter(src)

    def on_subhalos(self, hs: np.ndarray):
        m_s, z = hs['m_s'], hs['z']
        lg_Z = self(z, m_s)
        Z = 10.0**lg_Z
        hs['Z'] = Z

    def __call__(self, z: np.ndarray, m_star: np.ndarray) -> np.ndarray:
        '''
        (redshift z, stellar mass m_star) -> lg(Z/Zsun).
        '''
        src = self.src.rvalue

        h = self.ctx.cosm.hubble
        m_star = m_star * 1.0e10 / h                # 10^10 Msun/h -> Msun
        m_star = np.clip(m_star, 1.0e-10, None)

        if src == 0:
            out = 0.35 * np.log10(m_star / 1.0e10) \
                + 0.93 * np.exp(-0.43 * z) \
                - 1.05
        elif src == 1 or src == 2:
            if src == 2:
                z = z.clip(4.)
            out = 0.35 * np.log10(m_star / 10**10.5) \
                - 0.9 * np.log10(1.+z)
            out.clip(max=0.3, out=out)
        elif src == 3:
            lm = np.log10(m_star / 1.0e9)
            lx = np.log10(1. + z)
            out = 0.3 * lm - lx - 0.5
            out.clip(max=0.3, out=out)
        else:
            raise ValueError(f'Invalid source: {src}')

        return out


@jitclass
class _GasRegulator:
    y: numba.float64

    a_l0: numba.float64
    a_l1: numba.float64
    z_l: numba.float64
    b_l: numba.float64
    a_h0: numba.float64
    a_h1: numba.float64
    z_h: numba.float64
    b_h: numba.float64
    v_esc: numba.float64
    b_v: numba.float64

    a_mix_l: numba.float64
    a_mix_h: numba.float64
    g_mix: numba.float64
    b_mix: numba.float64

    a_z_l: numba.float64
    a_z_h: numba.float64
    b_z: numba.float64
    z_c: numba.float64

    r_ej: numba.float64
    f_return: numba.float64
    sigma_metal: numba.float64
    solar_metal: numba.float64

    rng: JitRng

    def __init__(self, y, a_l0, a_l1, z_l, b_l, a_h0, a_h1, z_h, b_h,
                 v_esc, b_v, a_mix_l, a_mix_h, g_mix, b_mix,
                 a_z_l, a_z_h, b_z, z_c,
                 r_ej, f_return, sigma_metal, solar_metal,
                 rng: JitRng) -> None:

        self.y = y

        self.a_l0 = a_l0
        self.a_l1 = a_l1
        self.z_l = z_l
        self.b_l = b_l
        self.a_h0 = a_h0
        self.a_h1 = a_h1
        self.z_h = z_h
        self.b_h = b_h
        self.v_esc = v_esc
        self.b_v = b_v

        self.a_mix_l = a_mix_l
        self.a_mix_h = a_mix_h
        self.g_mix = g_mix
        self.b_mix = b_mix

        self.a_z_l = a_z_l
        self.a_z_h = a_z_h
        self.b_z = b_z
        self.z_c = z_c

        self.r_ej = r_ej
        self.f_return = f_return
        self.sigma_metal = sigma_metal
        self.solar_metal = solar_metal

        self.rng = rng

    def on_brh(self, hs: np.ndarray) -> None:
        m_Z = 0.
        for h in hs:
            m_Z = self.__on_subhalo(h, m_Z)

    def __on_subhalo(self, h, m_Z):
        x_z = (h.z / self.z_l)**self.b_l
        a_l = (self.a_l0 + self.a_l1 * x_z) / (1. + x_z)

        x_z = (h.z / self.z_h)**self.b_h
        a_h = (self.a_h0 + self.a_h1 * x_z) / (1. + x_z)

        x_v = (h.v_max / self.v_esc)**self.b_v
        f_esc = (a_l + a_h * x_v)/(1. + x_v)

        x_g = np.abs(h.gamma / self.g_mix)**self.b_mix
        f_mix = (self.a_mix_l + self.a_mix_h * x_g)/(1. + x_g)

        x_z = (h.z / self.z_c)**self.b_z
        f_z = (self.a_z_l + self.a_z_h * x_z) / (1. + x_z)

        y_eff = self.y * (1. - f_esc) * f_mix * f_z

        dm_s, dm_bh, dm_g_ej, m_g = (h.dm_s, h.dm_bh, h.dm_g_ej, h.m_g)
        m_g_tot = max(m_g + dm_bh + dm_s + dm_g_ej, 1.0e-6)

        f_lock = 1 - self.f_return
        dm_Z_yield = y_eff * dm_s / f_lock
        dm_Z_lock = (dm_s + dm_bh) * m_Z / m_g_tot

        m_Z = m_Z + dm_Z_yield - dm_Z_lock
        dm_Z_ej = self.r_ej * dm_g_ej * m_Z / m_g_tot
        m_Z = m_Z - dm_Z_ej

        dm_Z = dm_Z_yield - dm_Z_ej - dm_Z_lock
        rv = 10.0 ** self.rng.normal(scale=self.sigma_metal)
        Z = m_Z / max(m_g, 1.0e-6) / self.solar_metal * rv

        h.dm_Z = dm_Z
        h.m_Z = m_Z
        h.dm_Z_yield = dm_Z_yield
        h.dm_Z_ej = dm_Z_ej
        h.dm_Z_lock = dm_Z_lock
        h.Z = Z

        return m_Z


class GasRegulator(Model):
    '''
    @yield: oxygen yield, Portinari+ 98 stellar evolution model, Chabrier IMF, 
    at Z_i = 0.02.
    @v_esc: physical [km/s].
    @solar_metal: Solar metallicity (oxygen mass fraction). The valus is 
    obtained from Bergemann+ 2021, where A(O) is 8.75 \pm 0.03 dex.
    '''

    def __init__(
            self, y=0.0163, a_l0=1., a_l1=1., z_l=5., b_l=1., a_h0=0., a_h1=1.,
            z_h=5., b_h=1., v_esc=75.0, b_v=2., a_mix_l=1., a_mix_h=0.,
            g_mix=1., b_mix=2., a_z_l=1., a_z_h=1., b_z=1., z_c=1., r_ej=0.,
            f_return=0.4, sigma_metal=.1, solar_metal=0.0090, **kw):

        super().__init__(**kw)

        self.y = Parameter(y)

        self.a_l0 = Parameter(a_l0)
        self.a_l1 = Parameter(a_l1)
        self.z_l = Parameter(z_l)
        self.b_l = Parameter(b_l)
        self.a_h0 = Parameter(a_h0)
        self.a_h1 = Parameter(a_h1)
        self.z_h = Parameter(z_h)
        self.b_h = Parameter(b_h)
        self.v_esc = Parameter(v_esc)
        self.b_v = Parameter(b_v)

        self.a_mix_l = Parameter(a_mix_l)
        self.a_mix_h = Parameter(a_mix_h)
        self.g_mix = Parameter(g_mix)
        self.b_mix = Parameter(b_mix)

        self.a_z_l = Parameter(a_z_l)
        self.a_z_h = Parameter(a_z_h)
        self.b_z = Parameter(b_z)
        self.z_c = Parameter(z_c)

        self.r_ej = Parameter(r_ej)
        self.f_return = Parameter(f_return)
        self.sigma_metal = Parameter(sigma_metal)
        self.solar_metal = Parameter(solar_metal)

        self.gas_regulator: Optional[_GasRegulator] = None

    def set_up(self) -> None:
        super().set_up()

        kw = {k: v.rvalue for k, v in self.parameters().items()}

        uv = self.ctx.cosm.unit_system.u_v_to_kmps
        kw['v_esc'] = kw.pop('v_esc') / float(uv)
        kw['rng'] = self.ctx.jit_rng

        self.gas_regulator = _GasRegulator(**kw)

    def tear_down(self) -> None:
        self.gas_regulator = None
        super().tear_down()

    def __call__(self, b: Branch):
        self.gas_regulator.on_brh(b.data)


class GalaxiesInGroupTree(Model):

    StellarMzrSource = Literal['Chen24']

    def __init__(self, f_r_lb=0.01,
                 force_mzr=False,
                 cvt_stellar_mzr=False,
                 stellar_mzr_src: StellarMzrSource = 'Chen24',
                 solar_oxygen=8.75,
                 **kw):
        super().__init__(**kw)

        if stellar_mzr_src == 'Chen24':
            stellar_mzr_src = 0
        else:
            raise ValueError(f'Invalid stellar source: {stellar_mzr_src}')

        self.f_r_lb = Parameter(f_r_lb)
        self.force_mzr = Parameter(force_mzr)
        self.cvt_stellar_mzr = Parameter(cvt_stellar_mzr)
        self.stellar_mzr_src = Parameter(stellar_mzr_src)
        self.solar_oxygen = Parameter(solar_oxygen)

        self.initializer = Initializer()
        self.fast_phase = FastPhase()
        self.slow_phase = SlowPhase()
        self.sat_phase = SatPhase()
        self.gas_regulator = GasRegulator()
        self.gas_phase_mzr = GasPhaseMZR()

        self.integrator: Optional[_Integrator] = None

    def set_up(self):
        super().set_up()
        self.integrator = _Integrator(
            self.f_r_lb.rvalue,
            self.fast_phase.fast_phase,
            self.slow_phase.slow_phase,
            self.sat_phase.sat_phase)

    def tear_down(self):
        self.integrator = None
        super().tear_down()

    def __call__(self, grptr: GroupTree):
        initializer, integrator, gas_regulator = (
            self.initializer, self.integrator, self.gas_regulator)
        for subtr in grptr.subtrs:
            for b in subtr.brhs:
                initializer(b)
                integrator.steps(b.data, b.idx_fast, b.idx_sat)
                gas_regulator(b)

        if self.force_mzr.rvalue:
            self.gas_phase_mzr.on_subhalos(grptr.data)

        # YT Chen 2024, 12 + log10(O/H) -> [Fe/H].
        if self.cvt_stellar_mzr.rvalue:
            assert self.stellar_mzr_src.rvalue == 0
            hs = grptr.data
            lg_z2zsun = np.log10(hs['Z'])
            lgo2hp12 = lg_z2zsun + self.solar_oxygen.rvalue
            fe2h = 1.20 * lgo2hp12 - 10.92
            hs['Z'] = 10.0**fe2h
