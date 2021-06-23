import typing

import numpy as np

from src.apis.extentions import Dict
from src.data.data_container import DataContainer


# when I write it, me and god knew how it works. Now, only god knows.
def reshape(shape: typing.Tuple):
    def _inner(a, b):
        if isinstance(b, DataContainer):
            return b.map(lambda x, y: _inner(x, y))
        return np.reshape(a, shape), b

    return _inner


def as_numpy(_, val: DataContainer):
    return val.as_numpy()


def as_tensor(_, val: DataContainer):
    return val.as_tensor()


def dict2dc(dc, key_val: typing.Tuple) -> DataContainer:
    dc = DataContainer([], []) if dc is None else dc
    return dc.concat(key_val[1])
