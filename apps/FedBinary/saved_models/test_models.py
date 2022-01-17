




# load the model from disk
import pickle

from src import tools
from src.apis import lambdas
from src.apis.plots import heatmap
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload

max_clients = 10
dataset_name = 'mnist'
# all_data = preload('femnist', distributor=UniqueDistributor(62, 600, 600))
all_data = preload(dataset_name, distributor=UniqueDistributor(max_clients, 6000, 6000))
# all_data = all_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

# clients_data = all_data.map(lambdas.dc_split(0.8, 0))
test_data = all_data.map(lambdas.dc_split(0.8, 1))


matrix = []
for i in range(max_clients):
    matrix.append([5] * max_clients)

for i in range(max_clients):
    print("printing result for client ID: ", i)
    results = []
    filename = f'{dataset_name}_model_{i}.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    for j in range(max_clients):

        # result = loaded_model.score(test_data[i], i)
        tt = tools.infer(loaded_model, test_data[j].map(lambda x, y: (x, i)).batch(128))
        results.append(tt)
        matrix[i][j] = tt[0]
        # print(tt)

    print(results)

heatmap(matrix)
