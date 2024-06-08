from __future__ import annotations
from typing import Self
import numpy as np
from pyhipp.core import abc
from pyhipp.io import h5
from .abc import record_type, RecordType

@record_type
class Subhalo(RecordType):
    fields = [
        # to be populated by preproc
        ('id', 'i8'),           # unique identifier across a simulation, e.g. SubLink id.
        ('leaf_id', 'i8'),      # `id` of main leaf
        ('last_pro_id', 'i8'),  # `id` of last progenitor

        ('id_in_branch', 'i4'), # index in the branch (from low z to high z)
        ('id_in_subtr', 'i4'),
        ('id_in_grptr', 'i4'),
        ('snap', 'i4'),
        ('z', 'f4'),
        ('a', 'f4'),
        ('t_lb', 'f4'),
        ('x_t', 'f4'),
        ('big_hubble', 'f4'),
        ('dt', 'f4'),           # first snap gets 0.

        ('m_h', 'f4'),          # lower-bound; for satellite, get an estimate
        ('dm_h', 'f4'),         # can be negative, first snap gets 0.
        ('r_h', 'f4'),          # comoving
        ('v_h', 'f4'),          # physical
        ('v_max', 'f4'),
        ('is_cent', 'bool'),
        ('is_fast', 'bool'),
        ('gamma', 'f4'),        # from fitted m_h history

        # to be populated by galaxy_model
        ('f_en', 'f4'),
        ('f_sn', 'f4'),
        ('f_agn', 'f4'),
        ('m_g', 'f4'),
        ('dm_g_hot', 'f4'),
        ('dm_g_cool', 'f4'),    # sf + ej + prev
        ('dm_g_sf', 'f4'),
        ('dm_g_ej', 'f4'),
        ('dm_g_prev', 'f4'),
        ('dm_g', 'f4'),         # sf (1 - bh - s) + prev
        ('r_sgc', 'f4'),        # f_gas * r_h * a

        ('m_s', 'f4'),
        ('dm_s', 'f4'),
        ('sfr', 'f4'),
        ('m_s_d', 'f4'),
        ('m_s_b', 'f4'),

        ('m_bh', 'f4'),
        ('dm_bh', 'f4'),
        ('bh_seeded', 'bool'),

        ('m_Z', 'f4'),         # [10^10 Msun/h]
        ('dm_Z', 'f4'),
        ('dm_Z_yield', 'f4'),
        ('dm_Z_ej', 'f4'),
        ('dm_Z_lock', 'f4'),
        ('Z', 'f4'),           # [Z_sun]
    ]


class SubhaloSet(abc.HasDictRepr):

    repr_attr_keys = ('size', 'keys')

    def __init__(self, data) -> None:
        super().__init__()

        self.data = np.asarray(data)

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def keys(self) -> list[str]:
        return Subhalo.keys

    def __getitem__(self, idx: int | str) -> (
            Subhalo | SubhaloSet | tuple[np.ndarray, ...] | np.ndarray):

        if isinstance(idx, str):
            return self.data[idx]
        elif isinstance(idx, tuple):
            return tuple(self.data[i] for i in idx)
        elif isinstance(idx, slice):
            return SubhaloSet(self.data[idx])
        else:
            return Subhalo(self.data[idx])

    @classmethod
    def new_zeros(cls, n, **kw):
        data = np.zeros(n, dtype=Subhalo.np_dtype)
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


class Branch(SubhaloSet):

    repr_attr_keys = SubhaloSet.repr_attr_keys + ('idx_sat', 'idx_fast')

    def __init__(self, data) -> None:
        super().__init__(data)

        self.idx_sat: int = None        # last central index + 1, [0, size]
        self.idx_fast: int = None       # last fast index, [-1, size-1]


class SubhaloTree(SubhaloSet):

    repr_attr_keys = SubhaloSet.repr_attr_keys + ('n_brhs', )

    def __init__(self, data) -> None:

        super().__init__(data)

        brh_offs = self.__find_meta()
        hs = self.data
        brhs = [Branch(hs[b:e][::-1])
                for b, e in zip(brh_offs[:-1], brh_offs[1:])]

        self.brhs = brhs
        self.n_brhs = len(brhs)

    def __find_meta(self):
        hs = self.data

        ids, leaf_ids = hs['id'], hs['leaf_id']
        b, n = 0, len(ids)
        brh_offs = [b]
        while b < n:
            e = b + (leaf_ids[b] - ids[b]) + 1
            brh_offs.append(e)
            b = e
        brh_offs = np.array(brh_offs)

        return brh_offs


class GroupTree(SubhaloSet):

    repr_attr_keys = SubhaloSet.repr_attr_keys + ('n_subtrs', )

    def __init__(self, data) -> None:

        super().__init__(data)

        subtr_offs = self.__find_meta()
        hs = self.data
        subtrs = [SubhaloTree(hs[b:e])
                  for b, e in zip(subtr_offs[:-1], subtr_offs[1:])]

        self.subtrs = subtrs
        self.n_subtrs = len(subtrs)

    def __find_meta(self):
        hs = self.data

        ids, last_pro_ids = hs['id'], hs['last_pro_id']
        b, n = 0, len(ids)
        subtr_offs = [b]
        while b < n:
            e = b + (last_pro_ids[b] - ids[b]) + 1
            subtr_offs.append(e)
            b = e
        subtr_offs = np.array(subtr_offs)

        return subtr_offs
