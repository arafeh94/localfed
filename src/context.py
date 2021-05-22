import copy
import statistics
import threading
from sklearn.cluster import KMeans
import logging
from src import tools
from src.data_container import DataContainer


class args:
    def __init__(self):
        self.sql_host = "localhost"
        self.sql_user = "root"
        self.sql_password = "root"
        self.sql_database = "mnist"


class Context:
    def __init__(self, clients_data: {int: DataContainer}, test_data: DataContainer, create_model: callable):
        self.args = args()
        self.clients_data: {int: DataContainer} = clients_data
        self.test_data = test_data
        self.model_stats = {}
        self.models = {}
        self.sample_dict = {}
        self.create_model = create_model
        self.init_model = self.create_model()
        self.logging = logging.getLogger('context')

    def build(self):
        self.logging.debug("Building Models --Started")
        all_threads = []
        for client_idx, data in self.clients_data.items():
            thread = threading.Thread(target=self._train_models, args=(client_idx, data))
            thread.start()
            all_threads.append(thread)
        for thread in all_threads:
            thread.join()
        self.logging.debug("Building Models --Finished")

    def _train_models(self, client_idx, data):
        model = copy.deepcopy(self.init_model)
        trained = tools.train(model, data.batch(8))
        self.model_stats[client_idx] = trained
        self.models[client_idx] = model
        self.sample_dict[client_idx] = len(data)

    def cluster(self, cluster_size=10):
        self.logging.debug("Clustering Models --Started")
        weights = []
        client_ids = []
        clustered = {}
        for client_id, stats in self.model_stats.items():
            client_ids.append(client_id)
            weights.append(stats['linear.weight'].numpy().flatten())
        kmeans = KMeans(n_clusters=cluster_size).fit(weights)
        for i, label in enumerate(kmeans.labels_):
            clustered[client_ids[i]] = label
        self.logging.debug("Clustering Models --Finished")
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

    def test_selection_accuracy(self, client_idx, title='test accuracy', output=True):
        self.logging.debug('-----------------' + title + '-----------------')
        global_model = self.aggregate_clients(client_idx)
        acc_loss = tools.infer(global_model, self.test_data.batch(8))
        if output:
            self.logging.debug(f"test case:{client_idx}")
            self.logging.debug(f"global model accuracy: {acc_loss[0]}, loss: {acc_loss[1]}")
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
