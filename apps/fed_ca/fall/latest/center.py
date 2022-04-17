import wandb

from libs.model.cv.cnn import Cnn1D
from src import manifest
from src.apis import lambdas
from src.apis.extensions import TorchModel, Dict
from src.data.data_container import DataContainer
from src.data.data_loader import preload
from src.federated.subscribers.wandb_logger import WandbLogger


def transformer(dt: Dict):
    dt = dt.map(lambda cid, dc: dc.map(lambda x, y: (x, cid)))
    dt = dt.map(lambdas.as_tensor)
    return dt


def add_wandb_log(project, federated, percentage_nb_client, model_name, batch_size, epochs, num_rounds,
                  learn_rate, dataset_used):
    federated.add_subscriber(WandbLogger(project=project,
                                         config={
                                             'lr': learn_rate,
                                             'batch_size': batch_size,
                                             'epochs': epochs,
                                             'num_rounds': num_rounds,
                                             'data_file': dataset_used,
                                             'model': model_name,
                                             'selected_clients': percentage_nb_client
                                         }))


learn_rate = 0.01
batch_size = 10000
epochs = 100
num_rounds = epochs
dataset_used = 'fall_ar_by_client'
model = TorchModel(Cnn1D(15))
model_name = type(model).__name__
percentage_nb_client = 15

print('loading dataset...')
data: Dict = preload(dataset_used, tag='fall_co', transformer=transformer)
# cl = len(data)
data: DataContainer = data.reduce(lambdas.dict2dc).filter(lambda x, y: y in range(0, 15)).as_tensor()
train, test = data.split(0.8)
print('training model...')
# model = TorchModel(Cnn1D(15))
model.train(train.batch(batch_size), lr=learn_rate, epochs=epochs)
print('inferring...')
res = model.infer(test.batch(batch_size))
print(res)

wandb.login(key=manifest.wandb_config['key'])
wandb.init(project='fall', entity=manifest.wandb_config['entity'], config={
    'lr': learn_rate, 'batch_size': batch_size,
    'epochs': epochs,
    'num_rounds': num_rounds, 'data_file': dataset_used,
    'model': model_name,
    'selected_clients': percentage_nb_client
})
