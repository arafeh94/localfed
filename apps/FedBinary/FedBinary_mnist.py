# multiclass classification - one model per client
import collections
import copy
import pickle
from random import randint, seed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.apis.plots import heatmap
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload

# clients_data = preload('xs1', 'cifar10', lambda dg: dg.distribute_shards(10, 1, 600, 600)) \
#     .map(lambdas.take_only_features(1024)).map(lambdas.dc_split(0.1, 1))

# one hot encode input variables
# onehot_encoder = OneHotEncoder(sparse=False)
# X = onehot_encoder.fit_transform(X)
from src.federated import federated
from src.federated.subscribers.analysis import ShowWeightDivergence

max_clients = 10
dataset_name = 'mnist'
# all_data = preload('femnist', distributor=UniqueDistributor(62, 600, 600))
all_data = preload(dataset_name, distributor=UniqueDistributor(max_clients, 6000, 6000))
# all_data = all_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

clients_data = all_data.map(lambdas.dc_split(0.8, 0))
test_data = all_data.map(lambdas.dc_split(0.8, 1))

# test_data = preload('xs1', 'cifar10', lambda dg: dg.distribute_shards(10, 1, 600, 600))\
#     .map(lambdas.take_only_features(1024)).map(lambdas.dc_split(0.1, 0))
print(clients_data)

clients_model = {}
client_samples = {}
clients_weights = {}


# def get_client_aggregated_model(client_id, round_id):
#     if round_id == 0:
#         return LogisticRegression(28 * 28, max_clients)
#         # return CNN_OriginalFedAvg()
#     else:
#         positive = copy.deepcopy(clients_weights[client_id])
#         # getting all clients except the positive client
#         negative_ids = list(filter(lambda _cid: _cid is not client_id, list(clients_model.keys())))
#         tmp_clients_weights = tools.dict_select(negative_ids, clients_weights)
#         tmp_clients_sample = tools.dict_select(negative_ids, client_samples)
#         negative = tools.aggregate(tmp_clients_weights, tmp_clients_sample)
#         positive['linear.weight'][1] = copy.deepcopy(negative['linear.weight'][client_id])
#         positive['linear.bias'][1] = copy.deepcopy(negative['linear.bias'][client_id])
#         clients_model[client_id].state_dict(positive)
#         return clients_model[client_id]

def get_client_aggregated_model(client_id, round_id):
    if round_id == 0:
        return LogisticRegression(28 * 28, 2)
        # return CNN_OriginalFedAvg()
    else:
        client_weights = copy.deepcopy(clients_weights[client_id])
        # getting all clients except the positive client
        negative_ids = list(filter(lambda _cid: _cid is not client_id, list(clients_model.keys())))

        tmp_clients_weights = tools.dict_select(negative_ids, clients_weights)
        tmp_clients_sample = tools.dict_select(negative_ids, client_samples) # todo check client_samples
        negative = tools.aggregate(tmp_clients_weights, tmp_clients_sample)

        client_weights['linear.weight'][1] = copy.deepcopy(negative['linear.weight'][0])
        client_weights['linear.bias'][1] = copy.deepcopy(negative['linear.bias'][0])
        clients_model[client_id].state_dict(client_weights)
        return clients_model[client_id]


max_rounds = 50
batch_size = 512
epochs = 20
lr = 0.0001
for round_id in range(max_rounds):
    for client_id, data in clients_data.items():
        client_model = get_client_aggregated_model(client_id, round_id)

        # assigning the class of training set as label 0
        print("Creating model for client ID", client_id, "In round", round_id)
        tools.train(client_model, data.map(lambda x, y: (x, 0)).batch(batch_size), epochs=epochs, lr=lr)
        clients_model[client_id] = client_model
        clients_weights[client_id] = client_model.state_dict()
        client_samples[client_id] = 6000

seed(60)

# saving the last model of the clients
for client_id in range(max_clients):
    client_model = get_client_aggregated_model(client_id, max_rounds)
    # # save the model to disk
    # filename = f'saved_models\{dataset_name}_model_{client_id}.sav'
    # pickle.dump(client_model, open(filename, 'wb'))

matrix = []
for i in range(max_clients):
    matrix.append([0] * max_clients)

for cid, model in clients_model.items():
    # total_client_false_positives = 0

    # random = randint(0, max_clients - 1)
    # while (random == cid):
    #     random = randint(0, max_clients - 1)
    results_TF = []
    results_FF = []
    # final_results = []

    print('***test on itself*** ', cid)
    # infer provides a tuple of accuracy and loss
    # tt = tools.infer_detailed(model, test_data[cid].map(lambda x, y: (x, cid)).batch(128))
    tt = tools.infer_detailed(model, test_data[cid].map(lambda x, y: (x, 0)).batch(batch_size))
    # final_results.append(tt[0])
    TP = tt[0]
    FN = 1 - TP
    matrix[cid][cid] = TP

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(tt[2], tt[3], normalize='true')
    # print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()

    # exit(0)

    # ft = tools.infer(model, test_data[cid].map(lambda x, y: (x, 1)).batch(128))
    # final_results.append(ft[0])

    print('***test on the other datas ***', cid)
    for i in range(max_clients):
        if i == cid:
            continue

        # res = tools.infer(model, test_data[i].map(lambda x, y: (x, cid)).batch(128))
        res = tools.infer(model, test_data[i].map(lambda x, y: (x, 0)).batch(batch_size))
        # results_TF.append(tf[0])
        FP = res[0]
        TN = 1 - FP
        matrix[cid][i] = FP

        precision = TP / (TP + FP)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(f'{cid}-{i}')
        print(f'Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, Accuracy: {accuracy}')
        print('')

        # total_client_false_positives = total_client_false_positives + FP

    print('')
    # total_accuracies = total_client_false_positives + TP
    # print(f"Average Error for client ID: {cid}", total_accuracies / max_clients)

heatmap(matrix)

print(f'configurations: max_rounds = {max_rounds}, batch_size = {batch_size}, epochs = {epochs}, lr = {lr}\n')

index = 0
TPs = []
FPs = []
for columns in matrix:
    FP = (sum(columns) - columns[index]) / (max_clients - 1)
    TP = columns[index]

    # Changing positional argument
    print(f'Label {index} => TP: {round(TP, 4)}, FP: {round(FP, 4)}')
    index = index + 1
    TPs.append(TP)
    FPs.append(FP)

print(f'\nTotal Models => TP:{round((sum(TPs)/max_clients), 4)}, FP:{round((sum(FPs)/max_clients),4)}')
