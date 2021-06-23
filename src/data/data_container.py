import copy
import typing

import numpy as np
import torch
from torch import Tensor


class DataContainer:
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
                  convert_y: typing.Callable[[torch.Tensor], torch.Tensor] = None) -> (Tensor, Tensor):
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

    def as_numpy(self, dtype=None):
        if self.is_tensor():
            return DataContainer(self.x.numpy(), self.y.numpy())
        if self.is_numpy():
            return self
        return DataContainer(np.asarray(self.x, dtype=dtype), np.asarray(self.y, dtype=dtype))

    def split(self, train_freq):
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

    def filter(self, predictor: typing.Callable):
        new_x = []
        new_y = []
        for x, y in zip(self.x, self.y):
            if predictor(x, y):
                new_x.append(x)
                new_y.append(y)
        return DataContainer(new_x, new_y)

    def map(self, mapper: typing.Callable):
        new_x = []
        new_y = []
        for x, y in zip(self.x, self.y):
            nx, ny = mapper(x, y)
            new_x.append(nx)
            new_y.append(ny)
        return DataContainer(new_x, new_y)
