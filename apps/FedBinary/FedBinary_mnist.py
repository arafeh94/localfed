import collections
import copy
from random import randint, seed, uniform
from statistics import mode

import torch
from sklearn.metrics import confusion_matrix

from libs.model.collection import MLP
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.apis.plots import heatmap
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload
seed(60)
torch.manual_seed(98)
# clients_data = preload('xs1', 'cifar10', lambda dg: dg.distribute_shards(10, 1, 600, 600)) \
#     .map(lambdas.take_only_features(1024)).map(lambdas.dc_split(0.1, 1))

max_clients = 10
# all_data = preload('femnist', distributor=UniqueDistributor(62, 600, 600))
all_data = preload('mnist', distributor=UniqueDistributor(max_clients, 1000, 1000))

clients_data = all_data.map(lambdas.dc_split(0.4, 1))

other_data = all_data.map(lambdas.dc_split(0.4, 0))

test_data = other_data.map(lambdas.dc_split(0.1, 0))
warmup_data = other_data.map(lambdas.dc_split(0.1, 1)).reduce(lambdas.dict2dc)

# test_data = preload('xs1', 'cifar10', lambda dg: dg.distribute_shards(10, 1, 600, 600))\
#     .map(lambdas.take_only_features(1024)).map(lambdas.dc_split(0.1, 0))
print(clients_data)

clients_model = {}
client_samples = {}
clients_weights = {}


def get_client_aggregated_model(cid, rdx):
    print(rdx, cid)
    if rdx == 0:
        model = LogisticRegression(28 * 28, 2)
        wam_dt = warmup_data.filter(lambda x, y: y != cid).map(lambda x, y: (x, 1)).as_tensor()
        tools.train(model, wam_dt.batch(50), epochs=10)
        return model
        # return MLP(28 * 28, 2)
        # return CNN_OriginalFedAvg()
    else:
        positive = copy.deepcopy(clients_weights[client_id])
        negative_ids = list(filter(lambda _cid: _cid is not cid, list(clients_model.keys())))
        tmp_clients_weights = tools.dict_select(negative_ids, clients_weights)
        tmp_clients_sample = tools.dict_select(negative_ids, client_samples)
        negative = tools.mode(tmp_clients_weights, tmp_clients_sample)
        negative1 = tools.aggregate(tmp_clients_weights, tmp_clients_sample)
        positive['linear.weight'][1] = copy.deepcopy(negative['linear.weight'][0])
        positive['linear.bias'][1] = copy.deepcopy(negative['linear.bias'][0])
        clients_model[cid].state_dict(positive)
        return clients_model[cid]


for round_id in range(10):
    for client_id, data in clients_data.items():
        client_model = get_client_aggregated_model(client_id, round_id)
        tools.train(client_model, data.map(lambda x, y: (x, 0)).batch(128), epochs=10)
        clients_model[client_id] = client_model
        clients_weights[client_id] = client_model.state_dict()
        client_samples[client_id] = 6000

seed(60)

matrix = []
for i in range(max_clients):
    matrix.append([0] * max_clients)

for cid, model in clients_model.items():

    # random = randint(0, max_clients - 1)
    # while (random == cid):
    #     random = randint(0, max_clients - 1)
    results_TF = []
    results_FF = []
    final_results = []

    print('***test on itself*** ', cid)
    # infer provides a tuple of accuracy and loss
    tt = tools.infer(model, test_data[cid].map(lambda x, y: (x, 0)).batch(128))
    # tt = tools.infer_detailed(model, test_data[cid].map(lambda x, y: (x, 0)).batch(125))
    final_results.append(tt[0])
    matrix[cid][cid] = tt[0]
    ft = tools.infer(model, test_data[cid].map(lambda x, y: (x, 1)).batch(128))
    final_results.append(ft[0])

    # targets
    # y_true_tt_target = tt[2]
    # y_true_ft_target = tt[2]

    # predicted
    # y_true_tt_predicted = ft[3]
    # y_true_ft_predicted = ft[3]

    # confusion_matrix(y_true_tt_target, y_true_tt_predicted)

    # print(cid, False, ft)

    # >> > y_true = [2, 0, 2, 2, 0, 1]
    # >> > y_pred = [0, 0, 2, 2, 0, 2]
    # >> > confusion_matrix(y_true, y_pred)

    print('***test on the other datas ***')
    for i in range(max_clients):
        if i == cid:
            continue

        tf = tools.infer(model, test_data[i].map(lambda x, y: (x, 0)).batch(128))
        results_TF.append(tf[0])
        # print(cid, False, tf)

        ff = tools.infer(model, test_data[i].map(lambda x, y: (x, 1)).batch(128))
        matrix[cid][i] = tf[0]
        results_FF.append(ff[0])
        # print(cid, True, ff)

    avg_tf = sum(results_TF) / len(results_TF)
    avg_ff = sum(results_FF) / len(results_FF)

    final_results.append(avg_tf)
    final_results.append(avg_ff)

    print('*** Confusion Matrix For Client ID:', cid, '***')
    print(final_results[0], final_results[2])
    print(final_results[1], final_results[3])

    print('')

heatmap(matrix)
# print(tools.infer(clients_model[60], test_data[60].map(lambda x, y: (x, 0)).batch(128)))
