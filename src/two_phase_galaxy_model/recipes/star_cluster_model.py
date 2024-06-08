from __future__ import annotations
from astropy import units, constants
from scipy.special import erfc
from .abc import Model, Parameter
from .subhalos import GroupTree
from .subclouds import SubcloudSet
import numba
from numba.experimental import jitclass
import numpy as np
from ..utils.algorithm import bisearch_nearest, bisearch_nearest_array
from ..utils.sampling import PowerLaw, WeightedPowerLaw, ExpPowerLaw, InterpolatedDist,\
    Rng as JitRng

@jitclass
class _Constants:
    # all in cgs
    hubble: numba.float64
    u_l: numba.float64
    u_t: numba.float64
    u_m: numba.float64
    u_rho: numba.float64
    u_v: numba.float64

    m_p: numba.float64
    mu: numba.float64
    m_mole: numba.float64

    # dimensionless
    u_m2msun: numba.float64
    u_l2pc: numba.float64

    def __init__(self, hubble, u_l, u_t, u_m, m_p, mu) -> None:

        m_sun = 1.988409870698051e33      # g
        pc = 3.0856775814913674e18        # cm

        self.hubble = hubble
        self.u_l = u_l
        self.u_t = u_t
        self.u_m = u_m
        self.u_rho = u_m / u_l**3
        self.u_v = u_l / u_t

        self.m_p = m_p
        self.mu = mu
        self.m_mole = self.m_p * self.mu

        self.u_m2msun = u_m / m_sun
        self.u_l2pc = u_l / pc

@jitclass
class _Sampler:
    # mass
    beta_m: numba.float64
    beta_mp: numba.float64
    m_g_min: numba.float64
    m_g_max: numba.float64
    m_g_trunc: numba.float64

    # density
    s: numba.float64[:]
    pdf: numba.float64[:]
    cdf: numba.float64[:]
    beta_sp: numba.float64
    f_r_gas: numba.float64
    eta: numba.float64
    fully_mixing: numba.float64
    cold_threshold: numba.float64
    beta_metal: numba.float64
    beta_metalp: numba.float64
    sigma_lmetal: numba.float64
    bias_lmetal: numba.float64
    sigma_metal: numba.float64
    bias_metal: numba.float64

    # SF
    n_sf_scale: numba.float64
    beta_n_sf: numba.float64
    beta_n_sfp: numba.float64
    force_shrink: numba.bool_
    eps_star: numba.float64
    beta: numba.float64
    gamma: numba.float64
    f_r_star: numba.float64
    f_return: numba.float64

    consts: _Constants
    rng: JitRng
    
    # working variables
    _dist_m_sc: WeightedPowerLaw
    _dist_sp: ExpPowerLaw
    _dist_metal: WeightedPowerLaw
    _is_cold: bool
    _n_sgc: numba.float64
    _n_shield_6: numba.float64
    _dist_n_sf: WeightedPowerLaw

    def __init__(
            self, beta_m, beta_mp,
            m_g_min, m_g_max, m_g_trunc, s, pdf, cdf, beta_sp,
            f_r_gas, eta, fully_mixing, cold_threshold,
            beta_metal, beta_metalp,
            sigma_lmetal, bias_lmetal, sigma_metal, bias_metal,
            n_sf_scale, beta_n_sf, beta_n_sfp, force_shrink,
            eps_star, beta, gamma, f_r_star, f_return, consts,
            rng):

        self.beta_m = beta_m
        self.beta_mp = beta_mp
        self.m_g_min = m_g_min
        self.m_g_max = m_g_max
        self.m_g_trunc = m_g_trunc

        self.s = s
        self.pdf = pdf
        self.cdf = cdf
        self.beta_sp = beta_sp
        self.f_r_gas = f_r_gas
        self.eta = eta
        self.fully_mixing = fully_mixing
        self.cold_threshold = cold_threshold
        self.beta_metal = beta_metal
        self.beta_metalp = beta_metalp
        self.sigma_lmetal = sigma_lmetal
        self.bias_lmetal = bias_lmetal
        self.sigma_metal = sigma_metal
        self.bias_metal = bias_metal

        self.n_sf_scale = n_sf_scale
        self.beta_n_sf = beta_n_sf
        self.beta_n_sfp = beta_n_sfp
        self.force_shrink = force_shrink
        self.eps_star = eps_star
        self.beta = beta
        self.gamma = gamma
        self.f_r_star = f_r_star
        self.f_return = f_return

        self.consts = consts
        self.rng = rng
        
        self._dist_m_sc = WeightedPowerLaw(beta_m, beta_mp, m_g_min, m_g_max)
        self._dist_sp = ExpPowerLaw(beta_sp, 1., 2.)
        self._dist_metal = WeightedPowerLaw(beta_metal, beta_metalp, 1., 2.)
        self._is_cold = True
        self._n_sgc = 0.
        self._n_shield_6 = 0.
        self._dist_n_sf = WeightedPowerLaw(beta_n_sf, beta_n_sfp, 1., 2.)

    def fill_scs(self, h, scs):
        '''
        Fill all sc properties, except for m_s_final.
        '''

        self._find_halo_property(h)
        m_tot = 0.
        for sc in scs:
            self._sample_m_sc(sc, h)
            self._sample_sc_density(sc, h)    
            self._star_formation(sc)
            m_tot += sc.m_s * sc.weight
        
        w = h.dm_s / m_tot if m_tot > 1.0e-10 else 1. 
        for sc in scs:
            sc.weight *= w
            
    def _find_halo_property(self, h):
        cc = self.consts
        
        m1, m2 = self.m_g_min, self.m_g_max
        dm_cls = h.m_g + h.dm_g_ej + h.dm_s
        m2 = min(m2, max(dm_cls, 1.0e-8))
        m1 = min(m1, 0.1 * m2)
        self._dist_m_sc.set_up_range(m1, m2)
        
        if not self.fully_mixing:
            self._dist_metal.set_up_range(1.0e-3, h.Z)
            
            x_z = ((1. + h.z) / 10.)**6.2
            x_m = h.m_h * 1.0e10 / cc.hubble / 10**10.8
            is_cold = x_m * x_z > self.cold_threshold
            self._is_cold = is_cold
        
        r_g = h.r_sgc * self.f_r_gas
        m_g = max(1.0e-8, h.m_g)
        rho_sgc = m_g / (4.0 / 3.0 * np.pi * r_g**3)
        _n_sgc = rho_sgc * cc.u_rho / cc.m_mole             # cm^-3
        self._n_sgc = _n_sgc
        
        sfr_1 = h.sfr * 10.0                                # [1.0 Msun/yr]
        eta_01 = self.eta / 0.1
        Z_002 = max(h.Z, 1.0e-3) / 0.02
        r_g_1 = r_g * cc.u_l2pc / 1.0e3                     # [1.0 kpc]
        _n_shield_6 = 94.71 * (eta_01 * sfr_1)**(3./7.) \
            * (Z_002 * r_g_1)**(-6./7.)                     # cm^-3, at m_sc_6 = 1
        self._n_shield_6 = _n_shield_6
    
    def _sample_m_sc(self, sc, h):
        '''
        Fill subcloud mass in [10^10 Msun/h] and weights.
        '''
        m_sc, w = self._dist_m_sc.rv(self.rng)
        if self.m_g_trunc > 0.:
            w = w * np.exp(-m_sc / self.m_g_trunc)
            
        sc.subhalo_id_in_grptr = h.id_in_grptr
        sc.snap = h.snap
        sc.z = h.z
        
        sc.m_g = m_sc
        sc.weight = w

    def _sample_sc_density(self, sc, h):
        cc = self.consts
        rng = self.rng
        
        if (not self.fully_mixing and h.is_fast 
            and h.Z > 2.0e-3 and self._is_cold):
            Z, w_Z = self._dist_metal.rv(rng)
        else:
            Z, w_Z = h.Z, 1.
        
        lZ_eps = rng.normal(self.bias_lmetal, self.sigma_lmetal)
        Z_eps = rng.normal(self.bias_metal, self.sigma_metal)
        Z = Z * 10**lZ_eps + Z_eps
        Z_002 = max(Z, 1.0e-3) / 0.02
        
        # shielding SN wind and SF density
        m_sc_6 = sc.m_g * 1.0e4 / cc.hubble                    # [10^6 Msun]
        _n_shield = self._n_shield_6 * m_sc_6**(-2./7.)        # cm^-3
        _n_sf = 3.4e3 * self.n_sf_scale * Z_002**(-2)

        # sample SC density
        # By definition, s = ln(rho/rho_0) = ln(n/n_0)
        _n_min = max(_n_shield, 1.0e-5)
        s_min = np.log(_n_min / self._n_sgc)
        P = self.cdf[bisearch_nearest(self.s, s_min)]
        P_hi = 1. - P
        w1 = P_hi
        
        s_max = self.s[-1]
        dist_sp = self._dist_sp
        dist_sp.set_up_range(s_min, s_max)
        s = dist_sp.rv(rng)
        w2 = self.pdf[bisearch_nearest(self.s, s)] / dist_sp.pdf_unnorm(s) 
        
        _n_sampled = self._n_sgc * np.exp(s)
        
        # let the SC shrink from _n_sampled to _n_sf
        if _n_sampled > _n_sf - 1.0e-5:
            _n_adopted = _n_sampled
            w_n_sf = 1.
        elif self.force_shrink:
            _n_adopted = _n_sf
            w_n_sf = 1.
        else:
            dist_n_sf = self._dist_n_sf
            dist_n_sf.set_up_range(_n_sampled, _n_sf)
            _n_adopted, w_n_sf = dist_n_sf.rv(rng)
            
        w = w1 * w2 * w_Z * w_n_sf
        
        sc.Z = Z
        sc.n_sampled = _n_sampled
        sc.n_sf = _n_sf
        sc.n_shield = _n_shield
        sc.n_adopted = _n_adopted
        sc.weight *= w
    
    def _star_formation(self, sc):
        cc = self.consts

        Z_002 = max(sc.Z, 1.0e-3) / 0.02
        _rho = sc.n_adopted * cc.m_mole        # g / cm^3
        rho = _rho / cc.u_rho
        r_sc = (sc.m_g / (4.0/3.0*np.pi*rho))**(1./3.)

        m_sc_6 = sc.m_g * cc.u_m2msun / 1.0e6     # 10^6 Msun
        r_sc_12 = r_sc * cc.u_l2pc / 12.0       # 12 pc
        eps = self.eps_star * (m_sc_6/r_sc_12)**self.beta * Z_002**self.gamma
        eps = min(max(eps, 0.05), 1.0)

        sc.n_sgc = self._n_sgc
        sc.rho_g = rho
        sc.r_g = r_sc
        sc.m_s = eps * (1.0 - self.f_return) * sc.m_g
        sc.r_s = self.f_r_star * r_sc

class Samplers(Model):
    '''
    @v_s: [km/s].
    '''

    def __init__(
            self, 
            m_batch=1.0e-6, n_m_min=4,
            f_s=1.0, tau_s=1., v_s=250.0, 
            stop_turb_slow_phase=False,
            stop_turb_sat_phase=False,
            stop_turb_below_gamma = -1.0e6,
            beta_m=-2., beta_mp=-0.75,
            m_g_min = 10.0**4, m_g_max = 10.0**8, m_g_trunc = 10**6.5,
            alpha_s=1.5, beta_sp=-1.25, f_r_gas=1.0, eta=0.1,
            fully_mixing=True, cold_threshold=0.1,
            beta_metal=0.75, beta_metalp=-0.75,
            sigma_lmetal=0.3, bias_lmetal=0., sigma_metal=0., bias_metal=0.,
            n_sf_scale=1.0, 
            beta_n_sf=-2.5, beta_n_sfp=-0.75,
            force_shrink=False,
            eps_star=1., beta=1., gamma=-0.9, f_r_star=0.5,
            f_return=0.4, **kw):

        super().__init__(**kw)

        self.m_batch = Parameter(m_batch)
        self.n_m_min = Parameter(n_m_min)
        self.f_s = Parameter(f_s)
        self.tau_s = Parameter(tau_s)
        self.v_s = Parameter(v_s)
        self.stop_turb_slow_phase = Parameter(stop_turb_slow_phase)
        self.stop_turb_sat_phase = Parameter(stop_turb_sat_phase)
        self.stop_turb_below_gamma = Parameter(stop_turb_below_gamma)

        self.beta_m = Parameter(beta_m)
        self.beta_mp = Parameter(beta_mp)
        self.m_g_min = Parameter(m_g_min)
        self.m_g_max = Parameter(m_g_max)
        self.m_g_trunc = Parameter(m_g_trunc)

        self.alpha_s = Parameter(alpha_s)
        self.beta_sp = Parameter(beta_sp)
        self.f_r_gas = Parameter(f_r_gas)
        self.eta = Parameter(eta)
        self.fully_mixing = Parameter(fully_mixing)
        self.cold_threshold = Parameter(cold_threshold)
        self.beta_metal = Parameter(beta_metal)
        self.beta_metalp = Parameter(beta_metalp)
        self.sigma_lmetal = Parameter(sigma_lmetal)
        self.bias_lmetal = Parameter(bias_lmetal)
        self.sigma_metal = Parameter(sigma_metal)
        self.bias_metal = Parameter(bias_metal)

        self.n_sf_scale = Parameter(n_sf_scale)
        self.beta_n_sf = Parameter(beta_n_sf)
        self.beta_n_sfp = Parameter(beta_n_sfp)
        self.force_shrink = Parameter(force_shrink)
        self.eps_star = Parameter(eps_star)
        self.beta = Parameter(beta)
        self.gamma = Parameter(gamma)
        self.f_r_star = Parameter(f_r_star)
        self.f_return = Parameter(f_return)

        self.samplers: list[_Sampler] = None
        self.sigma_s: np.ndarray = None

    def __call__(self, grptr: GroupTree):
        ctx = self.ctx
        
        # sampler index
        tau_s, v_s = self.tau_s.rvalue, self.v_s.rvalue
        u_v = ctx.cosm.unit_system.u_v_to_kmps
        v_h = grptr['v_h'] * u_v 
        c_s = 10.0                  # km/s
        M = self.f_s.value * (v_h / c_s) ** tau_s
        #M.clip(v_s/c_s, out=M)
        if self.stop_turb_slow_phase.rvalue:
            is_slow = ~grptr['is_fast']
            M[is_slow] = 0.
        gamma_crit = self.stop_turb_below_gamma.rvalue
        if gamma_crit > -1.0e5:
            sel = grptr['gamma'] < gamma_crit
            M[sel] = 0.
        #mach_sq = (M.clip(v_s/c_s))**2
        mach_sq = M**2 + (v_s / c_s)**2
        if self.stop_turb_sat_phase.rvalue:
            is_sat = ~grptr['is_cent']
            mach_sq[is_sat] = 0.
        
        sigma_s = np.log(1. + mach_sq)**(.5) 
        idx = bisearch_nearest_array(self.sigma_s, sigma_s)

        # determine no. of sampled SCs
        m_batch, n_m_min = self.m_batch.rvalue, self.n_m_min.rvalue
        dm_s = grptr['dm_s']
        n_sc = np.sqrt(dm_s / m_batch).astype(np.int64).clip(n_m_min)
        n_sc_all = n_sc.sum()
        sc_set = SubcloudSet.new_zeros(n_sc_all)
        
        # run sampling
        self.__sample(self.samplers, grptr.data, idx, n_sc, sc_set.data)

        return sc_set

    def set_up(self) -> None:

        super().set_up()

        cosm = self.ctx.cosm
        us = cosm.unit_system
        u_l = (us.u_l / units.cm).to(1).value
        u_m = (us.u_m / units.g).to(1).value
        u_t = (us.u_t / units.s).to(1).value
        m_p = (constants.m_p / units.g).to(1).value
        mu = 1.2
        cc = _Constants(cosm.hubble, u_l, u_t, u_m, m_p, mu)

        m_scale = cosm.hubble / 1.0e10
        kw = {
            k: v.rvalue for k, v in self.parameters().items()
        }
        kw['rng'] = self.ctx.jit_rng 
        for k in ('m_g_min', 'm_g_max', 'm_g_trunc'):
            v = kw[k]
            if v >= 0.:
                kw[k] = v * m_scale
        for k in ('m_batch', 'n_m_min', 'f_s', 'tau_s', 'v_s', \
            'stop_turb_slow_phase', 'stop_turb_sat_phase', 
            'stop_turb_below_gamma'):
            del kw[k]

        alpha_s = kw.pop('alpha_s')
        sigma_s = np.concatenate([
            np.linspace(0.1, 1., 21)[:-1],
            np.linspace(1., 10., 21)[:-1],
            np.linspace(10., 20., 10)
        ])
        s = np.linspace(0., 20., 256)
        samps = []
        for _sigma_s in sigma_s:
            pdf, cdf = self.__find_s_probdist(_sigma_s, alpha_s, s)
            samp = _Sampler(s=s, pdf=pdf, cdf=cdf, consts=cc, **kw)
            samps.append(samp)

        self.samplers = numba.typed.List(samps)
        self.sigma_s = sigma_s

    def tear_down(self) -> None:
        self.sigma_s = None
        self.samplers = None
        super().tear_down()

    @staticmethod
    def __find_s_probdist(sigma_s, alpha_s, s):
        sigma_s_sq = sigma_s * sigma_s
        s_t = (alpha_s - .5) * sigma_s_sq

        s_0 = -0.5 * sigma_s_sq
        p_lo = np.exp(-0.5 * (s - s_0)**2 / sigma_s_sq)

        C_s = np.exp(-0.5 * (s_t - s_0)**2 / sigma_s_sq)
        p_hi = C_s * np.exp(-alpha_s * (s-s_t).clip(0.))

        pdf = np.where(s > s_t, p_hi, p_lo)
        dp = pdf * (s[1] - s[0])
        cdf = np.cumsum(dp)
        norm = cdf[-1]

        pdf /= norm
        cdf /= norm

        return pdf, cdf

    @staticmethod
    @numba.njit
    def __sample(samplers, hs, idxs, n_scs, scs):
        b = 0
        for h, idx, n_sc in zip(hs, idxs, n_scs):
            sampler: _Sampler = samplers[idx]
            e = b + n_sc
            sampler.fill_scs(h, scs[b:e])
            b = e
        assert len(scs) == b
        
@jitclass
class _Disruption:
    
    alpha: numba.float64
    beta: numba.float64
    nu_0: numba.float64
    
    u_r_min: numba.float64
    u_r_max: numba.float64
    beta_r: numba.float64
    f_r_004: numba.float64
    
    f_return: numba.float64
    
    hubble: numba.float64
    
    rng: JitRng
    
    _dist_u: PowerLaw


    def __init__(self, alpha, beta, nu_0, u_r_min, u_r_max, beta_r, 
                 f_r_004, hubble, f_return, rng):
        
        self.alpha = alpha
        self.beta = beta
        self.nu_0 = nu_0
        self.u_r_min = u_r_min
        self.u_r_max = u_r_max
        self.beta_r = beta_r
        self.f_r_004 = f_r_004
        self.hubble = hubble
        self.f_return = f_return
        self.rng = rng
        
        self._dist_u = PowerLaw(beta_r, u_r_min, u_r_max)

    def on_scs(self, scs, hs, a_end):
        for sc in scs:
            h = hs[sc.subhalo_id_in_grptr]
            self._on_sc(h, sc, a_end)

    def _on_sc(self, h, sc, a_end):
        u_r, m_s_final = self.evolve_cls(sc.m_s, h.a, a_end)
        sc.u_r = u_r
        sc.m_s_final = m_s_final
        
    def evolve_cls(self, m_cls_beg, a_beg, a_end):
        '''
        Evolve a star cluster with initial stellar mass m_cls_beg
        from a = a_beg to a = a_end.
        Return u_r, m_cls_end.
        '''
        u_r = self._dist_u.rv(self.rng)
        r = u_r * self.f_r_004
        
        f_se = 1. - self.f_return
        m_ini = m_cls_beg / f_se
        m_ini_53 = m_ini * 1.0e10 / self.hubble  / 2.0e5
        m_ini_53 = max(m_ini_53, 1.0e-6)
        m_pow = m_ini_53 ** self.alpha
        
        dla = np.log(a_end / a_beg)
        
        _1mb = 1. - self.beta
        f1 = 1.0 - _1mb * self.nu_0 / r * m_pow * dla
        f1 = max(f1, 0.)
        
        m_cls_end = m_cls_beg * f1 ** (1. / _1mb)
        
        return u_r, m_cls_end

class Disruption(Model):
    def __init__(self,
                 alpha = -2./3., beta = -1./3., nu_0 = .55, 
                 u_r_min = 0.001, u_r_max = 1., beta_r = 0.0, f_r_004 = 1.0,
                 f_return = 0.4,
                 **kw):
        super().__init__(**kw)

        self.alpha = Parameter(alpha)
        self.beta = Parameter(beta)
        self.nu_0 = Parameter(nu_0)
        self.u_r_min = Parameter(u_r_min)
        self.u_r_max = Parameter(u_r_max)
        self.beta_r = Parameter(beta_r)
        self.f_r_004 = Parameter(f_r_004)
        self.f_return = Parameter(f_return)
        
        self.disruption: _Disruption = None
        
    def set_up(self) -> None:
        super().set_up()
        kw = {
            'hubble': self.ctx.cosm.hubble,
            'rng': self.ctx.jit_rng,
        }
        for k, v in self.parameters().items():
            kw[k] = v.rvalue
         
        self.disruption = _Disruption(**kw)
        
    def tear_down(self) -> None:
        self.disruption = None
        super().tear_down()

    def __call__(self, grptr: GroupTree, sc_set: SubcloudSet):
        a_end = grptr.data[0]['a']
        self.disruption.on_scs(sc_set.data, grptr.data, a_end)
        
@jitclass
class _GcsProfile:
    
    dist_fast: InterpolatedDist
    r_t_fast: numba.float64
    
    dist_slow: InterpolatedDist
    r_t_slow: numba.float64
    
    dist_ex: InterpolatedDist
    r_t_ex: numba.float64
    
    rng: JitRng
    
    def __init__(self, dist_fast, r_t_fast,
                 dist_slow, r_t_slow, dist_ex, r_t_ex, rng):
        
        self.dist_fast = dist_fast
        self.r_t_fast = r_t_fast
        
        self.dist_slow = dist_slow
        self.r_t_slow = r_t_slow
        
        self.dist_ex = dist_ex
        self.r_t_ex = r_t_ex
        
        self.rng = rng
    
    def on_scs(self, scs, hs):
        '''
        @scs: all SCs within a grptr.
        '''
        b, n_hs = 0, len(hs)
        root_hids = np.zeros(n_hs, dtype=np.int64)
        fast_hids = np.zeros(n_hs, dtype=np.int64)
        is_insitus = np.zeros(n_hs, dtype=np.bool_)
        while b < n_hs:
            h = hs[b]
            leaf_off = b + (h.leaf_id - h.id) + 1
            is_insitus[b:leaf_off] = True
            
            i_f = b
            while True:
                if hs[i_f].is_fast:
                    break
                i_f_next = i_f + 1
                if i_f_next == leaf_off:
                    break
                i_f = i_f_next
            
            e = b + (h.last_pro_id - h.id) + 1
            root_hids[b:e] = b
            fast_hids[b:e] = i_f
            
            b = e
        assert b == n_hs
        
        for sc in scs:
            hid = sc.subhalo_id_in_grptr
            h, is_insitu = hs[hid], is_insitus[hid]
            is_fast = h.is_fast
            
            h_root = hs[root_hids[hid]]
            r_root = h_root.r_h * h_root.a
            
            h_fast = hs[fast_hids[hid]]
            r_fast = h_fast.r_h * h_fast.a
            
            sc.r = self._on_sc(is_insitu, is_fast, sc, r_root, r_fast)
    
    def _on_sc(self, is_insitu, is_fast, sc, r_h, r_f):
        rv_u = self.rng.random()
        
        if not is_insitu:
            f_r = self.dist_ex.p2q(rv_u)
            r = f_r * self.r_t_ex * r_h
            return r
        
        if is_fast:
            f_r = self.dist_fast.p2q(rv_u)
            r = f_r * self.r_t_fast * r_f
        else:
            rv_u = sc.u_r
            assert rv_u >= 0.0 and rv_u <= 1.0
            f_r = self.dist_slow.p2q(rv_u)
            r = f_r * self.r_t_slow * r_h
        
        return r

class GcsProfile(Model):
    def __init__(self, f_gas = 0.04, 
                f_t_fast = .5, f_w_fast = 0.25, g_fast = -2.,
                shape_fast = 'exp',
                f_t_slow = 1., f_w_slow = 0.5, g_slow = -2.,
                shape_slow = 'exp',
                f_t_ex = 5., f_w_ex = 2.5, g_ex = -2.,
                shape_ex = 'exp',
            **kw):
        
        super().__init__(**kw)
        
        self.f_gas = Parameter(f_gas)
        self.f_t_fast = Parameter(f_t_fast)
        self.f_w_fast = Parameter(f_w_fast)
        self.g_fast = Parameter(g_fast)
        self.shape_fast = Parameter(shape_fast)
        
        self.f_t_slow = Parameter(f_t_slow)
        self.f_w_slow = Parameter(f_w_slow)
        self.g_slow = Parameter(g_slow)
        self.shape_slow = Parameter(shape_slow)
        
        self.f_t_ex = Parameter(f_t_ex)
        self.f_w_ex = Parameter(f_w_ex)    
        self.g_ex = Parameter(g_ex)    
        self.shape_ex = Parameter(shape_ex)
        
        self.gcs_profile: _GcsProfile = None
        
    def __call__(self, grptr: GroupTree, sc_set: SubcloudSet):
        self.gcs_profile.on_scs(sc_set.data, grptr.data)
        
    def set_up(self) -> None:
        super().set_up()
        
        f_gas = self.f_gas.rvalue
        rng = self.ctx.jit_rng
        
        f_t, f_w = self.f_t_fast.rvalue, self.f_w_fast.rvalue
        g = self.g_fast.rvalue
        shape = self.shape_fast.rvalue
        dist_fast = self.__get_dist(f_w / f_t, g, shape)
        r_t_fast = f_gas * f_t
        
        f_t, f_w = self.f_t_slow.rvalue, self.f_w_slow.rvalue
        g = self.g_slow.rvalue
        shape = self.shape_slow.rvalue
        dist_slow = self.__get_dist(f_w / f_t, g, shape)
        r_t_slow = f_gas * f_t
        
        f_t, f_w = self.f_t_ex.rvalue, self.f_w_ex.rvalue
        g = self.g_ex.rvalue
        shape = self.shape_ex.rvalue
        dist_ex = self.__get_dist(f_w / f_t, g, shape)
        r_t_ex = f_gas * f_t
        
        self.gcs_profile = _GcsProfile(
            dist_fast, r_t_fast, dist_slow, r_t_slow, 
            dist_ex, r_t_ex, rng)

    def tear_down(self) -> None:
        self.gcs_profile = None
        super().tear_down()
        
    def __get_dist(self, w, g, trunc='exp'):
        r_lb, r_ub = 1.0e-5, 5.0
        n_pts = 256
        
        g_r = g + 2.
        
        rs = np.linspace(r_lb, r_ub, n_pts)
        p_pow = rs**g_r
        if trunc == 'exp':
            p_trunc = np.exp( - rs / w ) 
        elif trunc == 'erfc':
            p_trunc = erfc( (rs - 1.0) / w )
        else:
            raise KeyError(trunc)
        pdfs = p_pow * p_trunc
        dist = InterpolatedDist(rs, pdfs)
        return dist

class SubcloudsInGroupTree(Model):
    def __init__(self, **kw):

        super().__init__(**kw)

        self.samplers = Samplers()
        self.disruption = Disruption()
        self.gcs_profile = GcsProfile()

    def __call__(self, grptr: GroupTree) -> SubcloudSet:
        sc_set = self.samplers(grptr)
        self.disruption(grptr, sc_set)
        self.gcs_profile(grptr, sc_set)
        return sc_set
