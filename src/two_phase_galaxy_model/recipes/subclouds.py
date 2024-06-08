from __future__ import annotations
from typing import Self
from .abc import record_type, RecordType
import numpy as np
from pyhipp.core import abc
from pyhipp.io import h5


@record_type
class Subcloud(RecordType):
    '''
    Units: 
    Z [Z_sun], n_{xxx} [cm^-3], m_{xxx} [10^10 Msun/h]
    '''
    fields = [
        ('subhalo_id_in_grptr', 'i4'),
        ('snap', 'i4'),
        ('z', 'f4'),
        ('Z', 'f4'),

        ('n_sgc', 'f4'),
        ('n_sampled', 'f4'),
        ('n_sf', 'f4'),
        ('n_shield', 'f4'),
        ('n_adopted', 'f4'),
        ('weight', 'f4'),

        ('m_g', 'f4'),
        ('rho_g', 'f4'),
        ('r_g', 'f4'),
        ('m_s', 'f4'),
        ('r_s', 'f4'),

        ('u_r', 'f4'),
        ('r', 'f4'),                # physical galactocentric distance
        ('m_s_final', 'f4'),
    ]


class SubcloudSet(abc.HasDictRepr):

    repr_attr_keys = ('size', 'keys')

    def __init__(self, data) -> None:
        super().__init__()

        self.data = np.asarray(data)

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def keys(self) -> list[str]:
        return Subcloud.keys

    def __getitem__(self, idx: int | str) -> (
            Subcloud | SubcloudSet | tuple[np.ndarray, ...] | np.ndarray):

        if isinstance(idx, str):
            return self.data[idx]
        elif isinstance(idx, tuple):
            return tuple(self.data[i] for i in idx)
        elif isinstance(idx, slice):
            return SubcloudSet(self.data[idx])
        else:
            return Subcloud(self.data[idx])

    @classmethod
    def new_zeros(cls, n, **kw):
        data = np.zeros(n, dtype=Subcloud.np_dtype)
        return cls(data, **kw)

    def save_h5(self, grp: h5.Group, d_flag='x'):
        '''
        Dump under `grp'.
        @d_flag: 'x' | 'ac'
        '''
        grp.datasets.create('data', self.data, flag=d_flag)

    @classmethod
    def load_h5(cls, grp: h5.Group) -> Self:
        data = grp.datasets['data']
        return cls(data)
