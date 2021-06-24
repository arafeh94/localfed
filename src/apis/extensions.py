import copy
import typing
from abc import abstractmethod

import numpy as np
import torch
import tqdm
from sklearn import decomposition
from torch import nn


class Functional:
    @abstractmethod
    def for_each(self, func: typing.Callable):
        pass

    @abstractmethod
    def filter(self, predicate: typing.Callable):
        pass

    @abstractmethod
    def map(self, func: typing.Callable):
        pass

    @abstractmethod
    def reduce(self, func: typing.Callable):
        pass

    @abstractmethod
    def select(self, keys):
        pass

    @abstractmethod
    def concat(self, other):
        pass


class Dict(dict, Functional):
    def __init__(self, iter_map: typing.Dict = {}):
        super().__init__(iter_map)

    def for_each(self, func: typing.Callable):
        new_dict = Dict()
        for key, val in self.items():
            new_dict[key] = func(key, val)
        return new_dict

    def concat(self, other):
        new_dict = copy.deepcopy(self)
        for item, val in other.items():
            new_dict[item] = val
        return new_dict

    def select(self, keys):
        return Dict({key: self[key] for key in keys})

    def filter(self, predicate: typing.Callable):
        new_dict = Dict()
        for key, val in self.items():
            if predicate(key, val):
                new_dict[key] = self[key]
        return new_dict

    def map(self, func: typing.Callable):
        new_dict = Dict()
        for key, val in self.items():
            new_val = func(key, val)
            new_dict[key] = new_val
        return new_dict

    def reduce(self, func: typing.Callable):
        first = None
        for key, val in self.items():
            first = func(first, key, val)
        return first


class Array(list, Functional):
    def __init__(self, iter_):
        super().__init__(iter_)

    def for_each(self, func: typing.Callable):
        for item in self:
            func(item)

    def filter(self, predicate: typing.Callable):
        new_a = []
        for item in self:
            if predicate(item):
                new_a.append(item)

    def map(self, func: typing.Callable):
        new_a = []
        for item in self:
            na = func(item)
            new_a.append(na)

    def reduce(self, func: typing.Callable):
        first = None
        for item in self:
            first = func(first, item)
        return first

    def select(self, indexes):
        new_a = []
        for index in indexes:
            new_a.append(self[index])

    def concat(self, other):
        return self.copy().extend(other)


class TorchModel:
    def __init__(self, model):
        self.model = model

    def train(self, batched, **kwargs):
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

    def weights(self):
        return self.model.state_dict()

    def copy(self):
        return copy.deepcopy(self)

    def load(self, weights):
        self.model.load_state_dict(weights)

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
