import typing

import numpy as np

from src.data.data_container import DataContainer


def r28c28(x, y):
    return np.reshape(x, (28, 28)), y


def as_numpy(_, val: DataContainer):
    return val.as_numpy()


def as_tensor(_, val: DataContainer):
    return val.as_tensor()


def dict21dc(dc, key_val: typing.Tuple):
    dc = DataContainer([], []) if dc is None else dc
    return dc.concat(key_val[1])
