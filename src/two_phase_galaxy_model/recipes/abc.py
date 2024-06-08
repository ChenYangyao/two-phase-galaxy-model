from __future__ import annotations
from pyhipp.core import abc
from pyhipp.stats.random import Rng
from pyhipp.astro.cosmology.model import (
    LambdaCDM, predefined as predefined_cosm)
from typing import (Iterator, Any, Union, Tuple, Self,
                    Iterable, Mapping, ClassVar, get_type_hints)
import numpy as np
import numba
import numbers
from ..utils.sampling import Rng as JitRng


class RecordType(abc.HasDictRepr):

    fields: ClassVar[list[tuple]]
    keys: ClassVar[list[str]]
    np_dtype: ClassVar[np.dtype]
    numba_dtype: ClassVar[numba.types.Record]

    def __init__(self, data: np.void) -> None:

        super().__init__()

        self.data = data

    def to_simple_repr(self) -> dict:
        data = self.data
        return {k: data[k] for (k, *_) in self.fields}

    def __getitem__(self, key: str) -> np.ndarray | int | float | Any:
        return self.data[key]


def record_type(t: RecordType) -> RecordType:
    '''
    Populate t with np_dtype and numba_dtype, using t.fields.
    '''
    fields = t.fields

    keys = list(f[0] for f in fields)
    setattr(t, 'keys', keys)

    np_dtype = np.dtype(fields)
    setattr(t, 'np_dtype', np_dtype)

    numba_dtype = numba.from_dtype(np_dtype)
    setattr(t, 'numba_dtype', numba_dtype)

    return t


default_cosm = predefined_cosm['tng']
default_rng = Rng()
default_log = abc.HasLog(False)


class Context(abc.HasDictRepr):

    repr_attr_keys = ('cosm', 'rng')

    def __init__(self,
                 cosm=default_cosm,
                 rng=default_rng,
                 log=default_log) -> None:
        self.cosm = cosm
        self.rng = rng
        self.log = log

    @property
    def jit_rng(self):
        return JitRng(self.rng._np_rng)


default_ctx = Context()


class NestedName:
    def __init__(self, name: Union[str, Tuple[str], NestedName] = ()) -> None:
        if isinstance(name, NestedName):
            name = name.name
        elif isinstance(name, str):
            if len(name) == 0:
                name = ()
            else:
                name = tuple(name.split('.'))
        else:
            assert isinstance(name, tuple)

        self.name: Tuple[str] = name

    def __add__(self, sub_name: str) -> NestedName:
        return NestedName(self.name + (sub_name,))

    def __repr__(self) -> str:
        return '.'.join(self.name)


class Component(abc.HasDictRepr):

    copy_passed_keys = ('ctx', )

    def __init__(self, ctx=default_ctx):
        super().__init__()
        self.ctx = ctx

    def set_ctx(self, ctx: Context) -> None:
        self.ctx = ctx
        for _, c in self.attrs_typed(Component):
            c: Component
            c.set_ctx(ctx)

    def set_up(self) -> None:
        for _, c in self.attrs_typed(Component):
            c: Component
            c.set_up()

    def tear_down(self) -> None:
        for _, c in self.attrs_typed(Component):
            c: Component
            c.tear_down()

    def copied(self, **kw) -> Self:
        kw = {k: getattr(self, k) for k in self.copy_passed_keys} | kw
        return type(self)(**kw)

    def attrs_typed(self, _type: type | tuple[type, ...]
                    ) -> Iterator[tuple[str, Any]]:
        for key in dir(self):
            val = getattr(self, key)
            if isinstance(val, _type):
                yield key, val


class Parameter(Component):

    repr_attr_keys = Component.repr_attr_keys + ('value',)
    copy_passed_keys = Component.copy_passed_keys + ('value',)

    def __init__(self, value: np.ndarray, **base_kw) -> None:
        super().__init__(**base_kw)
        self.value = np.array(value)

    def set_value(self, value: np.ndarray) -> Self:
        self.value[...] = value
        return self

    @property
    def rvalue(self) -> numbers.Real | list[numbers.Real]:
        '''
        A right value, i.e., immutable and built-in typed.
        '''
        return self.value.tolist()


class Model(Component):

    def add_parameters(self, parameters: dict[str, np.ndarray]):
        for k, v in parameters.items():
            setattr(self, k, Parameter(v))

    def parameters(self) -> dict[str, Parameter]:
        return dict(self.attrs_typed(Parameter))

    def iter_parameters(self, start=NestedName()
                        ) -> Iterator[Tuple[NestedName, Parameter]]:
        for name, p in self.parameters().items():
            yield start+name, p.value
        for name, m in self.sub_models().items():
            yield from m.iter_parameters(start+name)

    def set_parameter(self, name: NestedName, value: np.ndarray) -> None:
        self.sub_component(name).set_value(value)

    def update_parameters(self,
                          parameters: Mapping
                          [str, Union[Mapping, np.ndarray]]) -> None:
        for name, value in parameters.items():
            if isinstance(value, Mapping):
                self.sub_component(name).update_parameters(value)
            else:
                self.set_parameter(name, value)

    def sub_models(self) -> dict[str, Model]:
        return dict(self.attrs_typed(Model))

    def sub_component(self, name: NestedName) -> Union[Parameter, Model]:
        c = self
        name = NestedName(name)
        for sub_name in name.name:
            c = getattr(c, sub_name)
        return c

    def sub_components(self) -> dict[str, Union[Parameter, Model]]:
        return dict(self.attrs_typed((Parameter, Model)))

    def copied(self, **kw):
        out = super().copied(**kw)
        for k, v in self.sub_components().items():
            setattr(out, k, v.copied())
        return out

    def to_simple_repr(self) -> dict:
        return super().to_simple_repr() | {
            'parameters': self.parameters(),
            'sub_models': self.sub_models(),
        }
