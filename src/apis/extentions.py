import copy
import typing
from abc import abstractmethod

import numpy as np


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
            first = func(first, (key, val))
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
