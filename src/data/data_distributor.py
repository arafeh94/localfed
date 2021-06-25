import logging
import random
from collections import defaultdict

import numpy
import numpy as np
import typing

from libs.data_distribute import non_iid_partition_with_dirichlet_distribution
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src import tools


class Distributor:
    def __init__(self, data_container: DataContainer, verbose=0):
        self.data = data_container
        self.verbose = verbose

    def log(self, msg, level=1):
        if self.verbose >= level:
            logging.getLogger('distributor').info(msg)

    def distribute_dirichlet(self, num_clients, num_labels, skewness=0.5) -> Dict[int, DataContainer]:
        self.data = self.data.as_list()
        client_rows = non_iid_partition_with_dirichlet_distribution(self.data.y, num_clients, num_labels, skewness)
        clients_data = {}
        for client in client_rows:
            client_x = []
            client_y = []
            for pos in client_rows[client]:
                client_x.append(self.data.x[pos])
                client_y.append(self.data.y[pos])
            clients_data[client] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

    def distribute_percentage(self, num_clients, percentage=0.8, min_size=10, max_size=100) -> Dict[int, DataContainer]:
        self.data = self.data.as_list()
        clients_data = {}
        xs = self.data.x
        ys = self.data.y
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
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

    def distribute_shards(self, num_clients, shards_per_client, min_size, max_size) -> Dict[int, DataContainer]:
        self.data = self.data.as_numpy()
        clients_data = defaultdict(list)
        grouper = self.Grouper(self.data.x, self.data.y)
        for client_id in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            selected_shards = grouper.groups(shards_per_client)
            self.log(f'generating data for {client_id}-{selected_shards}')
            client_x = []
            client_y = []
            for shard in selected_shards:
                rx, ry = grouper.get(shard, int(client_data_size / len(selected_shards)))
                if len(rx) == 0:
                    self.log(f'shard {round(shard)} have no more available data to distribute, skipping...')
                else:
                    client_x = rx if len(client_x) == 0 else np.concatenate((client_x, rx))
                    client_y = ry if len(client_y) == 0 else np.concatenate((client_y, ry))
            clients_data[client_id] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

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
                selected_labels.append(self.next())
            return selected_labels

        def next(self):
            temp = 0 if self.label_cursor >= len(self.all_labels) else self.label_cursor
            self.label_cursor = (self.label_cursor + 1) % len(self.all_labels)
            return self.all_labels[temp]

        def get(self, label, size):
            x = self.grouped[label][self.selected[label]:self.selected[label] + size]
            y = [label] * len(x)
            self.selected[label] += size
            if len(x) == 0:
                del self.all_labels[self.all_labels.index(label)]
            return x, y

    def distribute_continuous(self, num_clients, min_size, max_size) -> Dict:
        self.data = self.data.as_list()
        clients_data = Dict()
        xs = self.data.x
        ys = self.data.y
        group = {}
        for index in range(len(xs)):
            if ys[index] not in group:
                group[ys[index]] = []
            group[ys[index]].append(xs[index])
        for i in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            client_x = group[i][0:client_data_size]
            client_y = [i for _ in range(len(client_x))]
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return clients_data

    def distribute_size(self, num_clients, min_size, max_size) -> Dict[int, DataContainer]:
        self.data = self.data.as_list()
        clients_data = Dict()
        xs = self.data.x
        ys = self.data.y
        data_pos = 0
        for i in range(num_clients):
            client_data_size = random.randint(min_size, max_size)
            client_x = xs[data_pos:data_pos + client_data_size]
            client_y = ys[data_pos:data_pos + client_data_size]
            data_pos += len(client_x)
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)
