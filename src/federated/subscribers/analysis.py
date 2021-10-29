import logging
import os
import time
from collections import defaultdict
from typing import Dict
import numpy as np
from src import tools
from src.apis import plots
from src.data.data_container import DataContainer
from src.federated.events import FederatedEventPlug


class ShowDataDistribution(FederatedEventPlug):
    def __init__(self, label_count, per_round=False, save_dir=None):
        super().__init__()
        self.logger = logging.getLogger('data_distribution')
        self.label_count = label_count
        self.per_round = per_round
        self.save_dir = save_dir
        self.round_id = -1
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def on_federated_started(self, params):
        clients_data: Dict[int, DataContainer] = params['trainers_data_dict']
        self.plot(clients_data)

    def on_training_start(self, params):
        self.round_id = params['context'].round_id
        if self.per_round:
            clients_data = params['trainers_data']
            self.plot(clients_data)

    def plot(self, clients_data):
        tick = time.time()
        self.logger.info('building data distribution...')
        ids = list(clients_data.keys())
        id_mapper = lambda id: ids.index(id)

        client_label_count = np.zeros((len(clients_data), self.label_count))
        for client_id, data in clients_data.items():
            for y in data.y:
                client_label_count[id_mapper(client_id)][y] += 1
        save_dir = f"{self.save_dir}/round_{self.round_id}_dd.png" if self.save_dir is not None else None
        plots.heatmap(client_label_count, 'Clients Data Distribution', 'x:Client - y:Class', save_dir)
        self.logger.info(f'building data distribution finished {time.time() - tick}')


class ShowWeightDivergence(FederatedEventPlug):
    def __init__(self, show_log=False, include_global_weights=False, save_dir=None, plot_type='matrix'):
        """
        plot_type = matrix | linear
        Returns:
            object: FederatedEventPlug
        """
        super().__init__()
        self.logger = logging.getLogger('weights_divergence')
        self.show_log = show_log
        self.include_global_weights = include_global_weights
        self.trainers_weights = None
        self.global_weights = None
        self.save_dir = save_dir
        self.round_id = 0
        self.plot_type = plot_type
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_training_end(self, params):
        self.trainers_weights = params['trainers_weights']

    def on_aggregation_end(self, params):
        self.global_weights = params['global_weights']

    def on_round_end(self, params):
        tick = time.time()
        self.logger.info('building weights divergence...')
        self.round_id = params['context'].round_id
        save_dir = f"./{self.save_dir}/round_{self.round_id}_wd.png" if self.save_dir is not None else None
        acc = params['accuracy']
        trainers_weights = self.trainers_weights
        if self.include_global_weights:
            trainers_weights[len(trainers_weights)] = self.global_weights
        ids = list(trainers_weights.keys())
        self.logger.info(f'building weights divergence finished {time.time() - tick}')
        if self.plot_type == 'matrix':
            id_mapper = lambda id: ids.index(id)
            heatmap = np.zeros((len(trainers_weights), len(trainers_weights)))
            for trainer_id, weights in trainers_weights.items():
                for trainer_id_1, weights_1 in trainers_weights.items():
                    w0 = tools.flatten_weights(weights)
                    w1 = tools.flatten_weights(weights_1)
                    heatmap[id_mapper(trainer_id)][id_mapper(trainer_id_1)] = np.var(np.subtract(w0, w1))
            plots.heatmap(heatmap, 'Weight Divergence', f'Acc {round(acc, 4)}', save_dir)
            if self.show_log:
                self.logger.info(heatmap)
        elif self.plot_type == 'linear':
            weight_dict = defaultdict(lambda: [])
            for trainer_id, weights in trainers_weights.items():
                weights = tools.flatten_weights(weights)
                weights = tools.compress(weights, 10, 1)
                weight_dict[trainer_id] = weights
            plots.linear(weight_dict, "Model's Weights", f'R: {self.round_id}', save_dir)
        else:
            raise Exception('plot type should be a string with a value either "linear" or "matrix"')
