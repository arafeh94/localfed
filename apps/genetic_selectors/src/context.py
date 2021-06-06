import copy
import statistics
import threading

import numpy
import numpy as np
import torch
from matplotlib import pyplot
from sklearn.cluster import KMeans
import logging
from src import tools
from src.data.data_container import DataContainer


class Context:
    def __init__(self, clients_data: {int: DataContainer}, create_model: callable):
        self.clients_data: {int: DataContainer} = clients_data
        self.model_stats = {}
        self.models = {}
        self.sample_dict = {}
        self.create_model = create_model
        self.init_model = self.create_model()
        self.logging = logging.getLogger('context')

    def train(self, ratio=0):
        self.logging.info("Building Models --Started")

        for client_idx, data in self.clients_data.items():
            if ratio > 0:
                shuffled = data.shuffle().as_tensor()
                new_x = shuffled.x[0:int(len(data.x) * ratio)]
                new_y = shuffled.y[0:int(len(data.y) * ratio)]
                data = DataContainer(new_x, new_y)
            self.logging.info(f"Building Models --ClientID{client_idx}")
            model = copy.deepcopy(self.init_model)
            trained = tools.train(model, data.batch(8))
            self.model_stats[client_idx] = trained
            self.models[client_idx] = model
            self.sample_dict[client_idx] = len(data)
            for key in trained:
                if torch.prod(torch.isnan(trained[key])) != 0:
                    for image in data.as_tensor().x:
                        pyplot.imshow(image.view(28, 28))
                        pyplot.show()
        self.logging.info("Building Models --Finished")

    def cluster(self, cluster_size=10):
        self.logging.info("Clustering Models --Started")
        weights = []
        client_ids = []
        clustered = {}
        for client_id, stats in self.model_stats.items():
            client_ids.append(client_id)
            weights.append(tools.flatten_weights(stats))
        kmeans = KMeans(n_clusters=cluster_size).fit(weights)
        for i, label in enumerate(kmeans.labels_):
            clustered[client_ids[i]] = label
        self.logging.info("Clustering Models --Finished")
        return clustered

    def cosine(self, client_idx):
        aggregated = tools.aggregate(tools.dict_select(client_idx, self.model_stats),
                                     tools.dict_select(client_idx, self.sample_dict))
        influences = []
        first = next(iter(self.model_stats.values()))
        for idx in client_idx:
            influence = tools.influence_cos(first, self.model_stats[idx], aggregated)
            if influence != 1:
                influences.append(influence)

        print(influences)
        print("\t\t\t".join(str(i) for i in influences))
        # print("\t".join(str(i) for i in tools.Clusters(-1, 1, 50).get_clusters(influences)))
        # print(tools.Clusters(-1, 1, 50).count(influences))
        # fitness = statistics.variance(tools.normalize(influences))
        # fitness = fitness * 10 ** 5
        # if output:
        #     print("test case:", client_idx)
        #     print("selection fitness:", fitness)
        # return fitness
        return 0

    def aggregate_clients(self, client_idx):
        global_model_stats = tools.aggregate(tools.dict_select(client_idx, self.model_stats),
                                             tools.dict_select(client_idx, self.sample_dict))
        global_model = self.create_model()
        tools.load(global_model, global_model_stats)
        return global_model

    def test_selection_accuracy(self, client_idx, test_data: DataContainer, title='test accuracy', output=True):
        self.logging.info('-----------------' + title + '-----------------')
        global_model = self.aggregate_clients(client_idx)
        acc_loss = tools.infer(global_model, test_data.batch(8))
        if output:
            self.logging.info(f"test case:{client_idx}")
            self.logging.info(f"global model accuracy: {acc_loss[0]}, loss: {acc_loss[1]}")
        return acc_loss

    def ecl(self, client_idx):
        aggregated = tools.aggregate(tools.dict_select(client_idx, self.model_stats),
                                     tools.dict_select(client_idx, self.sample_dict))
        influences = []
        for key in client_idx:
            influence = tools.influence_ecl(aggregated, self.model_stats[key])
            influences.append(influence)
        fitness = statistics.variance(tools.normalize(influences))
        fitness = fitness * 10 ** 5
        return fitness

    def fitness(self, client_idx):
        return self.ecl(client_idx)
