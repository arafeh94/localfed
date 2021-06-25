import typing

import numpy as np

from src.data.data_container import DataContainer


def reshape(shape: typing.Tuple):
    def _inner(a, b):
        if isinstance(b, DataContainer):
            return b.map(lambda x, y: _inner(x, y))
        return np.reshape(a, shape), b

    return _inner


def clients_features(nb_features):
    return lambda cid, data: DataContainer(data.x[:, 0:nb_features], data.y)


def empty(key, value):
    return len(value) > 0


def as_numpy(_, val: DataContainer):
    return val.as_numpy()


def as_tensor(_, val: DataContainer):
    return val.as_tensor()


def dict2dc(dc: DataContainer, key: int, val: DataContainer) -> DataContainer:
    dc = DataContainer([], []) if dc is None else dc
    return dc.concat(val)
