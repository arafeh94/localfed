import copy

import torch

from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data import data_loader
from src.federated.components.trainers import TorchTrainer

clients_data = data_loader.cifar10_2shards_100c_3000min_3000max().select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) \
    .map(lambda client_id, dataset: dataset.map(lambda x, y: (x[0:1024], y)))
test_data = data_loader.cifar10_2shards_100c_3000min_3000max().select([10, 11, 12, 13, 14, 16, 15, 17, 18, 19]) \
    .map(lambda client_id, dataset: dataset.map(lambda x, y: (x[0:1024], y)))
client_samples = {}
clients_weights = {}

print(clients_data)
print(test_data)

clients_model = {}
global_model = LogisticRegression(32 * 32, 10)

for client_id, data in clients_data.items():
    client_model = copy.deepcopy(global_model)
    tools.train(client_model, data.batch(50), epochs=50)
    clients_model[client_id] = client_model
    clients_weights[client_id] = client_model.state_dict()
    client_samples[client_id] = 3000


def agg(models_weights, models_label):
    new_model = models_weights[0]
    for model_weight, model_label in zip(models_weights, models_label):
        for key in model_weight:
            new_model[key][model_label] = model_weight[key][model_label]
    return new_model


weights = []
for cid, m in clients_model.items():
    weights.append(m.state_dict())

aggregated = agg(weights, [6, 9, 4, 1, 2, 7, 8, 3, 5, 0])
global_model_rep = copy.deepcopy(global_model)
global_model_rep.load_state_dict(aggregated)

avg = tools.aggregate(models_dict=clients_weights, sample_dict=client_samples)
global_model_avg = copy.deepcopy(global_model)
global_model_avg.load_state_dict(avg)

for client_id1, model1 in clients_model.items():
    for client_id2, client_test_data in test_data.items():
        print(f"{client_id1}-{client_id2}: {tools.infer(model1, client_test_data.batch(50))}")
    print("------------------------------------")

for client_id2, client_test_data in test_data.items():
    print(f"g_rep-{client_id2}: {tools.infer(global_model_rep, client_test_data.batch(50))}")
print("------------------------------------")

for client_id2, client_test_data in test_data.items():
    print(f"g_avg-{client_id2}: {tools.infer(global_model_avg, client_test_data.batch(50))}")

print("total:-------")
print(f"g_rep: {tools.infer(global_model_rep, test_data.reduce(lambdas.dict2dc).as_tensor().batch(50))}")
print(f"g_avg: {tools.infer(global_model_avg, test_data.reduce(lambdas.dict2dc).as_tensor().batch(50))}")
