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

    def as_tensor(self) -> (Tensor, Tensor):
        if self.is_tensor():
            return self
        if self.is_numpy():
            return DataContainer(torch.from_numpy(self.x).long(), torch.from_numpy(self.y).long())
        return DataContainer(torch.from_numpy(np.asarray(self.x)).long(), torch.from_numpy(np.asarray(self.y)).long())

    def as_numpy(self):
        if self.is_tensor():
            return DataContainer(self.x.numpy(), self.y.numpy())
        if self.is_numpy():
            return self
        return DataContainer(np.asarray(self.x), np.asarray(self.y))

    def split(self, train_freq):
        total_size = len(self)
        train_size = int(total_size * train_freq)
        x_train = self.x[0:train_size]
        y_train = self.y[0:train_size]
        x_test = self.x[train_size:total_size]
        y_test = self.y[train_size:total_size]
        return DataContainer(x_train, y_train), DataContainer(x_test, y_test)
