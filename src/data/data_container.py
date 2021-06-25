import copy
import typing

import numpy as np
import torch
from torch import Tensor

import src
from src.apis.extensions import Functional


class DataContainer(Functional):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def batch(self, batch_size):
        if len(self.x) == 0:
            return list()
        batch_data = list()
        batch_size = len(self.x) if batch_size <= 0 or len(self.x) < batch_size else batch_size
        for i in range(0, len(self.x), batch_size):
            batched_x = self.x[i:i + batch_size]
            batched_y = self.y[i:i + batch_size]
            batch_data.append((batched_x, batched_y))
        return batch_data

    def get(self):
        return self.x, self.y

    def is_empty(self):
        return self.x is None or len(self) == 0

    def __len__(self):
        return len(self.x)

    def is_tensor(self):
        return torch.is_tensor(self.x)

    def is_numpy(self):
        return type(self.x) == np.ndarray

    def as_tensor(self, convert_x: typing.Callable[[torch.Tensor], torch.Tensor] = None,
                  convert_y: typing.Callable[[torch.Tensor], torch.Tensor] = None) -> 'DataContainer':
        if convert_x is None:
            convert_x = lambda d: d.float()
        if convert_y is None:
            convert_y = lambda d: d.long()
        if self.is_tensor():
            return self
        if self.is_numpy():
            return DataContainer(convert_x(torch.from_numpy(self.x)), convert_y(torch.from_numpy(self.y)))
        return DataContainer(
            convert_x(torch.from_numpy(np.asarray(self.x))),
            convert_y(torch.from_numpy(np.asarray(self.y)))
        )

    def as_numpy(self, dtype=None) -> 'DataContainer':
        if self.is_tensor():
            return DataContainer(self.x.numpy(), self.y.numpy())
        if self.is_numpy():
            return self
        return DataContainer(np.asarray(self.x, dtype=dtype), np.asarray(self.y, dtype=dtype))

    def as_list(self) -> 'DataContainer':
        if self.is_numpy():
            return DataContainer(self.x.tolist(), self.y.tolist())
        if self.is_tensor():
            return DataContainer(self.x.numpy().tolist(), self.y.numpy().tolist())
        return self

    def split(self, train_freq) -> ('DataContainer', 'DataContainer'):
        total_size = len(self)
        train_size = int(total_size * train_freq)
        x_train = self.x[0:train_size]
        y_train = self.y[0:train_size]
        x_test = self.x[train_size:total_size]
        y_test = self.y[train_size:total_size]
        return DataContainer(x_train, y_train), DataContainer(x_test, y_test)

    def shuffle(self):
        dc = copy.deepcopy(self) if self.is_numpy() else self.as_numpy()
        p = np.random.permutation(len(dc.x))
        return DataContainer(dc.x[p], dc.y[p])

    def filter(self, predictor: typing.Callable[[typing.List, float], bool]) -> 'DataContainer':
        new_x = []
        new_y = []
        for x, y in zip(self.x, self.y):
            if predictor(x, y):
                new_x.append(x)
                new_y.append(y)
        return DataContainer(new_x, new_y)

    def map(self, mapper: typing.Callable[[typing.List, int], typing.Tuple[typing.List, int]]) -> 'DataContainer':
        new_x = []
        new_y = []
        for x, y in zip(self.x, self.y):
            nx, ny = mapper(x, y)
            new_x.append(nx)
            new_y.append(ny)
        return DataContainer(new_x, new_y)

    def for_each(self, func: typing.Callable[[typing.List, float], typing.NoReturn]):
        for x, y in zip(self.x, self.y):
            func(x, y)

    def reduce(self, func: typing.Callable[[typing.Any, typing.List, float], typing.Any]) -> 'DataContainer':
        first = None
        for x, y in zip(self.x, self.y):
            first = func(first, x, y)
        return first

    def select(self, keys) -> 'DataContainer':
        new_x = []
        new_y = []
        for key in keys:
            new_x.append(self.x[key])
            new_y.append(self.y[key])
        return DataContainer(new_x, new_y)

    def concat(self, other) -> 'DataContainer':
        new_x = other.x if self.is_empty() else np.concatenate((self.x, other.x))
        new_y = other.y if self.is_empty() else np.concatenate((self.y, other.y))
        return DataContainer(new_x, new_y)

    def distributor(self, verbose=0):
        from src.data.data_distributor import Distributor
        return Distributor(self, verbose)

    def __repr__(self):
        return f'Size:{len(self)}, Unique:{np.unique(self.y)}, Features:{len(self.x[0])}'
