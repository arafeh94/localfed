import copy
import logging
import os
import random
import typing
from collections import defaultdict
from typing import NewType, Union

import numpy as np
import pickle

from libs.data_distribute import non_iid_partition_with_dirichlet_distribution
from src import tools, manifest
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.data.data_provider import DataProvider, PickleDataProvider

logger = logging.getLogger('data_generator')


class DataGenerator:
    def __init__(self, data_provider: DataProvider, shuffle=False):
        """
        :param data_provider: instance of data provider
        """
        self.data = data_provider.collect()
        if shuffle:
            self.data = self.data.shuffle()
        self.data = self.data.as_numpy()
        self.distributed = None

    def distribute_dirichlet(self, num_clients, num_labels, skewness=0.5) -> Dict:
        self.distributed = self.data.distributor().distribute_dirichlet(num_clients, num_labels, skewness)
        return self.distributed

    def distribute_percentage(self, num_clients, percentage=0.8, min_size=10, max_size=100) -> Dict:
        self.distributed = self.data.distributor().distribute_percentage(num_clients, percentage, min_size, max_size)
        return self.distributed

    def distribute_shards(self, num_clients, shards_per_client, min_size, max_size):
        self.distributed = self.data.distributor().distribute_shards(num_clients, shards_per_client, min_size, max_size)
        return self.distributed

    def distribute_continuous(self, num_clients, min_size, max_size):
        self.distributed = self.data.distributor().distribute_continuous(num_clients, min_size, max_size)
        return self.distributed

    def distribute_size(self, num_clients, min_size, max_size):
        self.distributed = self.data.distributor().distribute_size(num_clients, min_size, max_size)
        return self.distributed

    def describe(self, selection=None):
        if self.distributed is None:
            logging.getLogger('data_generator').error('you have to distribute first')
            return
        tools.detail(self.distributed, selection)

    def get_distributed_data(self):
        if self.distributed is None:
            logging.getLogger('data_generator').error('you have to distribute first')
            return None
        return Dict(self.distributed)

    def save(self, path):
        obj = copy.deepcopy(self)
        obj.data = []
        file = open(path, 'wb')
        pickle.dump(obj, file)


def load(path) -> DataGenerator:
    logging.getLogger('DataGenerator').debug('loaded data_generator has only @var.distributed available')
    file = open(path, 'rb')
    dg = pickle.load(file)
    return dg
