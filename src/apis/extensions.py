import copy
import logging
import os
import pickle
import typing
from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
import tqdm
from sklearn import decomposition
from torch import nn

T = typing.TypeVar('T')
B = typing.TypeVar('B')
K = typing.TypeVar('K')
V = typing.TypeVar('V')


class Functional(typing.Generic[T]):
    @abstractmethod
    def for_each(self, func: typing.Callable[[T], typing.NoReturn]):
        pass

    @abstractmethod
    def filter(self, predicate: typing.Callable[[T], bool]):
        pass

    @abstractmethod
    def map(self, func: typing.Callable[[T], T]):
        pass

    @abstractmethod
    def reduce(self, func: typing.Callable[[B, T], B]):
        pass

    @abstractmethod
    def select(self, keys):
        pass

    @abstractmethod
    def concat(self, other):
        pass


class Dict(typing.Dict[K, V], Functional):

    def __init__(self, iter_map: typing.Dict[K, V] = {}):
        super().__init__(iter_map)

    def for_each(self, func: typing.Callable) -> typing.NoReturn:
        for key, val in self.items():
            func(key, val)

    def concat(self, other) -> 'Dict[K, V]':
        new_dict = copy.deepcopy(self)
        for item, val in other.items():
            new_dict[item] = val
        return new_dict

    def select(self, keys) -> 'Dict[K, V]':
        return Dict({key: self[key] for key in keys})

    def filter(self, predicate: typing.Callable[[K, V], bool]) -> 'Dict[K, V]':
        new_dict = Dict()
        for key, val in self.items():
            if predicate(key, val):
                new_dict[key] = self[key]
        return new_dict

    def map(self, func: typing.Callable[[K, V], V]) -> 'Dict[K, V]':
        new_dict = Dict()
        for key, val in self.items():
            new_val = func(key, val)
            new_dict[key] = new_val
        return new_dict

    def reduce(self, func: typing.Callable[[B, K, V], B]) -> 'B':
        first = None
        for key, val in self.items():
            first = func(first, key, val)
        return first


class Array(typing.List[V], Functional):
    def __init__(self, iter_=None):
        iter_ = iter_ if iter_ is not None else []
        super().__init__(iter_)

    def for_each(self, func: typing.Callable[[V], None]) -> typing.NoReturn:
        for item in self:
            func(item)

    def filter(self, predicate: typing.Callable[[V], bool]) -> 'Array[V]':
        new_a = Array()
        for item in self:
            if predicate(item):
                new_a.append(item)
        return new_a

    def map(self, func: typing.Callable[[V], V]) -> 'Array[V]':
        new_a = Array()
        for item in self:
            na = func(item)
            new_a.append(na)
        return new_a

    def reduce(self, func: typing.Callable[[B, V], B]) -> 'B':
        first = None
        for item in self:
            first = func(first, item)
        return first

    def select(self, indexes: typing.List[V]) -> 'Array[V]':
        new_a = Array()
        for index in indexes:
            new_a.append(self[index])
        return new_a

    def concat(self, other: 'Array[V]') -> 'Array[V]':
        return Array(self.copy().extend(other))


class Serializable:
    def __init__(self, file_path):
        self.file_path = file_path

    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        to_save = {}
        for key, item in self.__dict__.items():
            if not callable(item):
                to_save[key] = item
        pickle.dump(to_save, open(self.file_path, 'wb'))

    def load(self):
        if self.exists():
            try:
                for key, item in pickle.load(open(self.file_path, 'rb')).items():
                    self.__dict__[key] = item
            except Exception as e:
                print(e)

    def exists(self):
        return os.path.exists(self.file_path)

    def sync(self, func, *params):
        self.load()
        func(*params)
        self.save()


class TorchModel:
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger('TorchModel')

    def train(self, batched, **kwargs):
        r"""
        Args:
            batched:
            **kwargs:
                epochs (int)
                lr (float)
                momentum (float)
                optimizer (Optimizer)
                criterion (_WeightedLoss)
                verbose (int)
        Returns:

        """
        model = self.model
        epochs = kwargs.get('epochs', 10)
        learn_rate = kwargs.get('lr', 0.003)
        momentum = kwargs.get('momentum', 0)
        optimizer = kwargs.get('optimizer', torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum))
        criterion = kwargs.get('criterion', nn.CrossEntropyLoss())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        verbose = kwargs.get('verbose', 1)
        model.to(device)
        model.train()
        iterator = tqdm.tqdm(range(epochs), 'training') if verbose else range(epochs)

        for _ in iterator:
            for batch_idx, (x, labels) in enumerate(batched):
                x = x.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        weights = model.cpu().state_dict()
        return weights

    def infer(self, batched, **kwargs):
        verbose = kwargs.get('verbose', 1)
        model = self.model
        model.eval()
        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            iterator = tqdm.tqdm(enumerate(batched), 'inferring') if verbose else enumerate(batched)
            for batch_idx, (x, target) in iterator:
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc / test_total, test_loss / test_total

    def log(self, msg, level=1):
        self.logger.info(msg)

    def weights(self):
        return self.model.state_dict()

    def copy(self):
        return copy.deepcopy(self)

    def load(self, weights):
        self.model.load_state_dict(weights)

    def save(self, file_path):
        save_file = open(file_path, 'wb')
        pickle.dump(self.model, save_file)

    @staticmethod
    def open(file_path):
        if os.path.isfile(file_path):
            model = pickle.load(open(file_path, 'rb'))
            return TorchModel(model)
        else:
            return False

    def flatten(self):
        all_weights = []
        for _, weight in self.weights().items():
            all_weights.extend(weight.flatten().tolist())
        return np.array(all_weights)

    def compress(self, output_dim, n_components):
        weights = self.flatten().reshape(output_dim, -1)
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(weights)
        weights = pca.transform(weights)
        return weights.flatten()

    def extract(self):
        return self.model
