import typing

import numpy as np

from src.data.data_container import DataContainer


def reshape(shape: typing.Tuple) -> typing.Callable:
    def _inner(a, b):
        if isinstance(b, DataContainer):
            return b.reshape(shape)
        return np.reshape(a, shape), b

    return _inner


def transpose(shape: typing.Tuple) -> typing.Callable:
    def _inner(a, b):
        if isinstance(b, DataContainer):
            return b.transpose(shape)
        return np.transpose(a, shape), b

    return _inner


def take_only_features(nb_features) -> typing.Callable:
    return lambda cid, data: DataContainer(data.x[:, 0:nb_features], data.y)


def empty(_, value) -> bool:
    return len(value) > 0


def as_numpy(_, val: DataContainer) -> np.array:
    return val.as_numpy()


def as_tensor(_, val: DataContainer) -> 'Tensor':
    return val.as_tensor()


def dict2dc(dc: DataContainer, key: int, val: DataContainer) -> DataContainer:
    dc = DataContainer([], []) if dc is None else dc
    return dc.concat(val)


def dc_split(percentage, take0or1) -> typing.Callable:
    return lambda cid, data: data.split(percentage)[take0or1]
