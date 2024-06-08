from __future__ import annotations
from .abc import Model, Parameter, record_type, RecordType
from .subhalos import SubhaloSet, GroupTree
import numpy as np
import numba
from numba.experimental import jitclass
from pyhipp.core import DataTable

@record_type
class _Subhalo(RecordType):
    fields = [
        # read from raw
        ('is_cent', 'bool'),
        ('t_lb', 'f4'),
        ('big_hubble', 'f4'),
        ('a', 'f4'),
        ('rho_h', 'f4'),
        ('m_h', 'f4'),
        ('v_max', 'f4'),

        # halo properties
        ('m_h_out', 'f4'),
        ('v_max_out', 'f4'),
        ('r_h_out', 'f4'),
        ('v_h_out', 'f4'),
        ('dm_h_out', 'f4'),
        ('dt_out', 'f4'),
        ('id_in_branch_out', 'i4'),
        ('id_in_subtr_out', 'i4'),
        ('id_in_grptr_out', 'i4'),

        # branch properties
        ('gamma_out', 'f4'),
        ('is_fast_out', 'bool'),
    ]


@jitclass
class _HaloProperties:

    f_kernel: numba.float64
    f_max: numba.float64

    m_h_lb: numba.float64
    v_h_lb: numba.float64
    r_h_lb: numba.float64

    big_g: numba.float64

    def __init__(self, f_kernel, f_max, m_h_lb, v_h_lb, r_h_lb, big_g) -> None:

        self.f_kernel = f_kernel
        self.f_max = f_max
        self.m_h_lb = m_h_lb
        self.v_h_lb = v_h_lb
        self.r_h_lb = r_h_lb
        self.big_g = big_g

    def run_on_branches(self, hs, brh_sizes, subtr_sizes):
        '''
        Fill _out properties.
        '''
        b = 0
        for brh_size in brh_sizes:
            e = b + brh_size
            self.__run_on_branch(hs[b:e][::-1])
            b = e
        assert len(hs) == b

        b = 0
        for subtr_size in subtr_sizes:
            e = b + subtr_size
            self.__run_on_subtr(hs[b:e])
            b = e
        assert len(hs) == b

        self.__run_on_grptr(hs)

    def __run_on_branch(self, hs):
        m_h_lb, v_h_lb = self.m_h_lb, self.v_h_lb
        for h in hs:
            h.m_h = max(h.m_h, m_h_lb)
            h.v_max = max(h.v_max, v_h_lb)
        self.__fill_sat(hs)
        self.__smooth(hs)

        r_h_lb, v_h_lb = self.r_h_lb, self.v_h_lb
        for i, h in enumerate(hs):
            vol = h.m_h_out / h.rho_h
            r_h = (vol / (4./3.*np.pi))**(1./3.)
            r_h_phy = r_h * h.a
            v_h = np.sqrt(self.big_g * h.m_h_out / r_h_phy)
            h.r_h_out = max(r_h, r_h_lb)
            h.v_h_out = max(v_h, v_h_lb)
            if i == 0:
                h.dt_out = 0.
                h.dm_h_out = 0.
            else:
                h.dt_out = hs[i-1].t_lb - h.t_lb
                h.dm_h_out = h.m_h_out - hs[i-1].m_h_out
            h.id_in_branch_out = i

    def __smooth(self, hs):
        n = len(hs)
        for i, h in enumerate(hs):
            t_lb = h.t_lb
            t_dyn = self.__find_t_dyn(h)
            dt_max = self.f_max * t_dyn
            dt_sig = self.f_kernel * t_dyn

            l, r = max(i-1, 0), min(i+1, n-1)
            while l > 0 and hs[l].t_lb - t_lb < dt_max:
                l -= 1
            while r < n-1 and t_lb - hs[r].t_lb < dt_max:
                r += 1
            r += 1

            hs_near = hs[l:r]
            dt = np.abs(hs_near.t_lb - t_lb)
            w = np.exp(-0.5 * (dt / dt_sig)**2)
            w /= w.sum()
            h.m_h_out = (hs_near.m_h * w).sum()
            h.v_max_out = (hs_near.v_max * w).sum()

    def __fill_sat(self, hs):
        n = len(hs)
        i, i_l, i_r = 0, 0, 0
        while i < n:
            if not hs[i].is_cent:
                i += 1
                continue
            i_r = i
            h_l, h_r = hs[i_l], hs[i_r]
            for j in range(i_l+1, i_r):
                h = hs[j]
                x_l, x_r = np.log(h_l.a), np.log(h_r.a)
                y_l, y_r = np.log(h_l.m_h), np.log(h_r.m_h)
                x = np.log(h.a)
                dx_l, dx_r = x - x_l, x_r - x
                dx = x_r - x_l
                w_l, w_r = dx_r / dx, dx_l / dx
                y = w_l * y_l + w_r * y_r
                h.m_h = np.exp(y)
            i_l = i
            i += 1
        for j in range(i_l+1, n):
            hs[j].m_h = hs[i_l].m_h

    def __find_t_dyn(self, h):
        vol = h.m_h / h.rho_h
        r_h_phy = (vol / (4./3.*np.pi))**(1./3.) * h.a
        v_h = np.sqrt(self.big_g * h.m_h / r_h_phy)
        t_dyn = r_h_phy / v_h
        return t_dyn

    def __run_on_subtr(self, hs):
        for i, h in enumerate(hs):
            h.id_in_subtr_out = i

    def __run_on_grptr(self, hs):
        for i, h in enumerate(hs):
            h.id_in_grptr_out = i


class HaloProperties(Model):
    def __init__(self, f_kernel=1.0, f_max=3.0,
                 m_h_lb=1.0e-4, v_h_lb=1.0e-4, r_h_lb=1.0e-4,
                 **kw):

        super().__init__(**kw)

        self.f_kernel = Parameter(f_kernel)
        self.f_max = Parameter(f_max)
        self.m_h_lb = Parameter(m_h_lb)
        self.v_h_lb = Parameter(v_h_lb)
        self.r_h_lb = Parameter(r_h_lb)

    def set_up(self) -> None:
        super().set_up()
        kw = {}
        for k, v in self.parameters().items():
            kw[k] = v.rvalue
        kw['big_g'] = self.ctx.cosm.unit_system.c_gravity
        self.halo_properties = _HaloProperties(**kw)

    def tear_down(self) -> None:
        self.halo_properties = None
        super().tear_down()

    def __call__(self, grptr_raw: DataTable) -> tuple[np.ndarray, ...]:
        impl = self.halo_properties

        n = len(grptr_raw['id'])
        hs = np.zeros(n, dtype=_Subhalo.np_dtype)
        keys = _Subhalo.keys
        for k in keys:
            if k[-4:] != '_out':
                hs[k] = grptr_raw[k]
        id, leaf_id, last_pro_id = grptr_raw['id', 'leaf_id', 'last_pro_id']
        brh_sizes = self.__find_chunk_sizes(id, leaf_id)
        subtr_sizes = self.__find_chunk_sizes(id, last_pro_id)
        impl.run_on_branches(hs, brh_sizes, subtr_sizes)

        return hs, brh_sizes, subtr_sizes

    @staticmethod
    @numba.njit
    def __find_chunk_sizes(ids, ptrs):
        '''
        @ptrs: e.g. leaf_ids, last_pro_ids.
        '''
        b, n = 0, len(ids)
        sizes = []
        while b < n:
            size = (ptrs[b] - ids[b]) + 1
            sizes.append(size)
            b += size
        return np.array(sizes)


@jitclass
class _BranchProperties:
    gamma_f: numba.float64
    disable_fitting: numba.bool_
    enable_f_a: numba.bool_
    m_h_lb: numba.float64
    z_ub: numba.float64
    n_lb: numba.int64
    big_g: numba.float64

    def __init__(self, gamma_f, disable_fitting, enable_f_a,
                 m_h_lb, z_ub, n_lb, big_g) -> None:

        self.gamma_f = gamma_f
        self.disable_fitting = disable_fitting
        self.enable_f_a = enable_f_a
        self.m_h_lb = m_h_lb
        self.z_ub = z_ub
        self.n_lb = n_lb
        self.big_g = big_g

    def run_on_branches(self, hs, brh_sizes):
        idxs = np.empty((brh_sizes.size, 2), dtype=np.int64)
        b = 0
        for i, brh_size in enumerate(brh_sizes):
            e = b + brh_size
            idxs[i] = self.__run_on_branch(hs[b:e][::-1])
            b = e
        assert len(hs) == b
        return idxs

    def __run_on_branch(self, hs):
        idx_sat = self.__find_idx_sat(hs)
        idx_fast = self.__fill_gamma(hs, idx_sat)
        return idx_sat, idx_fast

    def __find_idx_sat(self, hs):
        '''
        hs.is_cent -> idx_sat.
        '''
        idx_lc = -1
        i = len(hs)
        while i > 0:
            i -= 1
            if hs[i].is_cent:
                idx_lc = i
                break
        idx_sat = idx_lc + 1
        return idx_sat

    def __fill_gamma(self, hs, idx_sat):
        '''
        hs.{a, t_lb, big_hubble, m_h_out} -> hs.{is_fast_out, gamma_out}, idx_fast
        '''
        if idx_sat <= 1:
            hs.is_fast_out[:] = False
            hs.gamma_out[:] = 0.0
            if idx_sat == 1:
                hs.is_fast_out[0] = True
            return idx_sat - 1

        hs_c = hs[:idx_sat]
        a, t_lb, big_h = hs_c['a'], hs_c['t_lb'], hs_c['big_hubble']
        m = hs_c['m_h_out']
        z = np.float32(1.0) / a - np.float32(1.0)
        ln_zp1 = np.log(np.float32(1.0)+z)

        sel = (z <= self.z_ub) & (m >= self.m_h_lb)
        if not self.disable_fitting and sel.sum() >= self.n_lb:
            y = np.log(m)
            ones = np.ones_like(a)
            if self.enable_f_a:
                f_a = np.float32(1.0) - a
                f_la = ln_zp1 - f_a
                f_z = z - f_a
                X = np.stack((ones, f_a, f_la, f_z), axis=-1)
            else:
                f_la = ln_zp1
                f_z = z - f_la
                X = X = np.stack((ones, f_la, f_z), axis=-1)
            coef = np.linalg.lstsq(X[sel], y[sel])[0]
            y_pred = X @ coef
            y_pred.clip(-23., 23., out=y_pred)
            m = np.exp(y_pred)

        v = 100**(1./6.) * (self.big_g * m * big_h)**(1./3.)
        ln_v = np.log(v)
        loss = ln_v + self.gamma_f * ln_zp1
        idx_fast = loss.argmax()

        for i, h in enumerate(hs):
            h.is_fast_out = i <= idx_fast
        gamma = 0.
        for i, h in enumerate(hs):
            if i >= idx_sat:
                h.gamma_out = gamma
                continue
            i_r = max(i, 1)
            i_l = i_r - 1
            dln_v = ln_v[i_r] - ln_v[i_l]
            dt = t_lb[i_l] - t_lb[i_r]
            gamma = dln_v / dt / big_h[i_r]
            h.gamma_out = gamma

        return idx_fast


class BranchProperties(Model):
    def __init__(self, gamma_f = 3./16.,
                 disable_fitting=False, enable_f_a=True,
                 m_h_lb=1.0e-4, z_ub=15.0, n_lb=15, **kw):

        super().__init__(**kw)

        self.gamma_f = Parameter(gamma_f)
        self.disable_fitting = Parameter(disable_fitting)
        self.enable_f_a = Parameter(enable_f_a)
        self.m_h_lb = Parameter(m_h_lb)
        self.z_ub = Parameter(z_ub)
        self.n_lb = Parameter(n_lb)

    def set_up(self) -> None:
        super().set_up()
        kw = {k: v.rvalue for k, v in self.parameters().items()}
        kw['big_g'] = self.ctx.cosm.unit_system.c_gravity
        self.branch_properties = _BranchProperties(**kw)

    def tear_down(self) -> None:
        self.branch_properties = None
        super().tear_down()

    def __call__(self, hs: np.ndarray, brh_sizes: np.ndarray) -> np.ndarray:
        return self.branch_properties.run_on_branches(hs, brh_sizes)


class PreprocGroupTree(Model):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.halo_properties = HaloProperties()
        self.branch_properties = BranchProperties()

    def __call__(self, grptr_raw: DataTable) -> GroupTree:
        '''
        Required input arrays: 
        id, leaf_id, last_pro_id, snap, z, a, t_lb, x_t, big_hubble,
        is_cent, rho_h, m_h, v_max.
        '''

        hs, brh_sizes, subtr_sizes = self.halo_properties(grptr_raw)
        idxs = self.branch_properties(hs, brh_sizes)

        hs_out = SubhaloSet.new_zeros(hs.size).data
        keys = ('id', 'leaf_id', 'last_pro_id',
                'snap', 'z', 'a', 't_lb', 'x_t', 'big_hubble',
                'is_cent')
        for k in keys:
            hs_out[k] = grptr_raw[k]


        for k in _Subhalo.keys:
            if k[-4:] == '_out':
                k_out = k[:-4]
                hs_out[k_out] = hs[k]

        grptr = GroupTree(hs_out)
        i_brh = 0
        for subtr in grptr.subtrs:
            for b in subtr.brhs:
                b.idx_sat, b.idx_fast = idxs[i_brh]
                i_brh += 1
        assert i_brh == brh_sizes.size

        return grptr
