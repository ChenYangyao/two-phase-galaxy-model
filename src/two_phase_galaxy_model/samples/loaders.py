from __future__ import annotations
import typing
from typing import Self
from pyhipp.astro.cosmology.model import LambdaCDM
from pyhipp.core import abc, DataTable
import numpy as np


class GroupTreeLoader(abc.HasDictRepr):

    def __init__(self, cosm: LambdaCDM) -> None:

        self.cosm = cosm

    def from_dict(self, data: dict, copy=True) -> DataTable:
        '''
        @data: must contain items of 
            id, leaf_id, last_pro_id, 
            snap, z, is_cent, 
            m_h [10^10 Msun/h], v_max [km/s].
        '''

        data_out = DataTable({str(k): np.asarray(v)
                              for k, v in data.items()}, copy=copy)
        cosm = self.cosm
        us = cosm.unit_system
        ht = cosm.halo_theory

        z, v_max = data_out['z', 'v_max']

        a = 1.0 / (1.0 + z)
        x_t = np.log10(1.+z)
        big_hubble = cosm.big_hubble(z)
        v_max = v_max / us.u_v_to_kmps          # to internal unit defined by us
        rho_h = ht.rho_vir_crit(z=z)

        # make interpretation for lookback times
        zmin, zmax = max(z.min() - 0.1, 0.0), z.max() + 0.1
        z_list = np.linspace(zmin, zmax, 1024)
        t_lb_list = cosm.times.lookback_at(z_list)
        t_lb = np.interp(z, z_list, t_lb_list)

        data_out |= {
            'a': a, 'x_t': x_t, 'big_hubble': big_hubble,
            'rho_h': rho_h, 'v_max': v_max, 't_lb': t_lb
        }

        return data_out
