import copy
import logging
import os
import random
from collections import defaultdict

import numpy as np
import pickle

from libs.data_distribute import non_iid_partition_with_dirichlet_distribution
from src import tools, manifest
from src.data.data_container import DataContainer
from src.data.data_provider import DataProvider, PickleDataProvider

logger = logging.getLogger('data_generator')


class DataGenerator:
    def __init__(self, data_provider: DataProvider, xtt=None, ytt=None):
        """
        :param data_provider: instance of data provider
        :param xtt: x tensor type, callable to transform the current type to the desired type, by default float
        :param ytt: y tensor type, callable to transform the current type to the desired type, by default long
        """
        self.data = data_provider.collect().as_numpy()
        self.distributed = None
        self.xtt = xtt
        self.ytt = ytt

    def distribute_dirichlet(self, num_clients, num_labels, skewness=0.5) -> {int: DataContainer}:
        client_rows = non_iid_partition_with_dirichlet_distribution(self.data.y, num_clients, num_labels, skewness)
        clients_data = {}
        for client in client_rows:
            client_x = []
            client_y = []
            for pos in client_rows[client]:
                client_x.append(self.data.x[pos])
                client_y.append(self.data.y[pos])
            clients_data[client] = DataContainer(client_x, client_y).as_tensor(self.xtt, self.ytt)
        self.distributed = clients_data
        return clients_data

    def distribute_percentage(self, num_clients, percentage=0.8, min_size=10, max_size=100) -> {int: DataContainer}:
        clients_data = {}
        xs = self.data.x.tolist()
        ys = self.data.y.tolist()
        unique_labels = np.unique(ys)
        for i in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            selected_label = unique_labels[random.randint(0, len(unique_labels) - 1)]
            client_x = []
            client_y = []
            while len(client_y) / client_data_size < percentage:
                for index, item in enumerate(ys):
                    if item == selected_label:
                        client_x.append(xs.pop(index))
                        client_y.append(ys.pop(index))
                        break
            while len(client_y) < client_data_size:
                for index, item in enumerate(ys):
                    if item != selected_label:
                        client_x.append(xs.pop(index))
                        client_y.append(ys.pop(index))
                        break
            clients_data[i] = DataContainer(client_x, client_y).as_tensor(self.xtt, self.ytt)
        self.distributed = clients_data
        return clients_data

    def distribute_shards(self, num_clients, shards_per_client, min_size, max_size, verbose=0):
        clients_data = defaultdict(list)
        grouper = self.Grouper(self.data.x, self.data.y)
        for client_id in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            selected_shards = grouper.groups(shards_per_client)
            logging.getLogger('distribute_shards').info(f'generating data for {client_id}-{selected_shards}')
            client_x = None
            client_y = None
            for shard in selected_shards:
                rx, ry = grouper.get(shard, int(client_data_size / len(selected_shards)))
                client_x = rx if client_x is None else np.concatenate((client_x, rx))
                client_y = ry if client_y is None else np.concatenate((client_y, ry))
            clients_data[client_id] = DataContainer(client_x, client_y).as_tensor(self.xtt, self.ytt)
        self.distributed = clients_data
        return clients_data

    class Grouper:
        def __init__(self, x, y):
            self.grouped = defaultdict(list)
            self.selected = defaultdict(lambda: 0)
            self.label_cursor = 0
            for label, data in zip(y, x):
                self.grouped[label].append(data)
            self.all_labels = list(self.grouped.keys())

        def groups(self, count=None):
            if count is None:
                return self.all_labels
            selected_labels = []
            for i in range(count):
                selected_labels.append(self.all_labels[self.label_cursor])
                self.label_cursor = (self.label_cursor + 1) % len(self.all_labels)
            return selected_labels

        def get(self, label, size):
            x = self.grouped[label][self.selected[label]:self.selected[label] + size]
            y = [label] * len(x)
            self.selected[label] += size
            return x, y

    def distribute_continuous(self, num_clients, min_size, max_size):
        clients_data = {}
        xs = self.data.x.tolist()
        ys = self.data.y.tolist()
        group = {}
        for index in range(len(xs)):
            if ys[index] not in group:
                group[ys[index]] = []
            group[ys[index]].append(xs[index])
        for i in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            client_x = group[i][0:client_data_size]
            client_y = [i for _ in range(len(client_x))]
            clients_data[i] = DataContainer(client_x, client_y).as_tensor(self.xtt, self.ytt)
        self.distributed = clients_data
        return clients_data

    def distribute_size(self, num_clients, min_size, max_size):
        clients_data = {}
        xs = self.data.x.tolist()
        ys = self.data.y.tolist()
        data_pos = 0
        for i in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            client_x = xs[data_pos:data_pos + client_data_size]
            client_y = ys[data_pos:data_pos + client_data_size]
            data_pos += len(client_x)
            clients_data[i] = DataContainer(client_x, client_y).as_tensor(self.xtt, self.ytt)
        self.distributed = clients_data
        return clients_data

    def describe(self, selection=None):
        if self.distributed is None:
            logging.getLogger(self).error('you have to distribute first')
            return
        tools.detail(self.distributed, selection)

    def get_distributed_data(self):
        if self.distributed is None:
            logging.getLogger('data_generator').error('you have to distribute first')
            return None
        return self.distributed

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

