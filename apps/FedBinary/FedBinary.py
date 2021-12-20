import copy

from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.data.data_loader import preload

# clients_data = preload('xs1', 'cifar10', lambda dg: dg.distribute_shards(10, 1, 600, 600)) \
#     .map(lambdas.take_only_features(1024)).map(lambdas.dc_split(0.1, 1))
clients_data = preload('femnist_62c_1000mn_1000mx', 'femnist', lambda dg: dg.distribute_shards(62, 1, 600, 600)).map(
    lambdas.dc_split(0.1, 1))
test_data = preload('femnist_62c_1000mn_1000mx', 'femnist', lambda dg: dg.distribute_shards(62, 1, 600, 600)).map(
    lambdas.dc_split(0.1, 0))
# test_data = preload('xs1', 'cifar10', lambda dg: dg.distribute_shards(10, 1, 600, 600))\
#     .map(lambdas.take_only_features(1024)).map(lambdas.dc_split(0.1, 0))
print(clients_data)

clients_model = {}
client_samples = {}
clients_weights = {}


def get_client_aggregated_model(cid, rdx):
    if rdx == 0:
        return LogisticRegression(28 * 28, 2)
    else:
        positive = copy.deepcopy(clients_weights[client_id])
        negative_ids = list(filter(lambda _cid: _cid is not cid, list(clients_model.keys())))
        tmp_clients_weights = tools.dict_select(negative_ids, clients_weights)
        tmp_clients_sample = tools.dict_select(negative_ids, client_samples)
        negative = tools.aggregate(tmp_clients_weights, tmp_clients_sample)
        positive['linear.weight'][1] = copy.deepcopy(negative['linear.weight'][0])
        positive['linear.bias'][1] = copy.deepcopy(negative['linear.bias'][0])
        clients_model[cid].state_dict(positive)
        return clients_model[cid]


for round_id in range(1):
    for client_id, data in clients_data.items():
        client_model = get_client_aggregated_model(client_id, round_id)
        tools.train(client_model, data.map(lambda x, y: (x, 0)).batch(50), epochs=1)
        clients_model[client_id] = client_model
        clients_weights[client_id] = client_model.state_dict()
        client_samples[client_id] = 1000

for cid, model in clients_model.items():
    print(tools.infer(model, test_data[cid].map(lambda x, y: (x, 0)).batch(50)))

# print(tools.infer(clients_model[60], test_data[60].map(lambda x, y: (x, 0)).batch(50)))
