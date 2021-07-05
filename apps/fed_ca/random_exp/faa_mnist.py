import logging
import platform

import torchvision.models as models
from scipy.stats import entropy
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchsummary import summary
from torch import nn

from src import tools
from src.data import data_loader
from src.federated import subscribers
from src.federated.components import aggregators, metrics, client_selectors, trainers
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
dataset_name = 'mnist'
clients_nb = 10
min_samples = 1000
max_samples = 1000
shards_nb = 0
client_data = data_loader.pickle_distribute_continuous(dataset_name, clients_nb, min_samples, max_samples)
dataset_used = dataset_name + "_" + str(clients_nb) + "c_" + str(min_samples) + "mn_" + str(max_samples) + "mx.pkl"
tools.detail(client_data)

ui = []
ei = []
x = []
for trainer_id, data in client_data.items():
    data = data.shuffle().as_tensor()
    train, test = data.split(0.8)

    ui.append(torch.mean(train.x))
    ei.append(torch.var(train.x, unbiased=False))

    mean_value = torch.mean(train.x).numpy().max()
    variance_value = torch.var(train.x, unbiased=False).numpy().max()
    # number of samples 1000
    nd = random.normal(loc=mean_value, scale=variance_value, size=(1, 1000))
    x.append(nd)
    # sns.distplot(nd, hist=False)
    # plt.show()
    # exit(0)

y = []

# creating a dataset from the guassian distrubition and the labels,
# D = [xi = x, yi = y]
for i in range(len(x)):
    y.append(i)

data = [1, 2, 2, 3, 3, 3]

pd_series = pd.Series(data)
counts = pd_series.value_counts()
entropy = entropy(counts)

# print(entropy)

vgg16 = models.vgg16().cuda()
summary(vgg16, (3, 224, 224))

# # setting hyper parameters
batch_size = 64
epochs = 100
num_rounds = 80
learn_rate = 0.001
#  momentum 0.9
momentum = 0.9
optimizer = 'sgd'
criterion = 'cel'
print(f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}')
trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size, epochs=epochs,
                               optimizer=optimizer, criterion=criterion, lr=learn_rate, momentum=momentum)

federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.2),
    trainers_data_dict=client_data,
    initial_model=lambda: vgg16,
    num_rounds=num_rounds,
    desired_accuracy=0.99
)

federated.add_subscriber(subscribers.WandbLogger(config={
    'lr': learn_rate, 'batch_size': batch_size,
    'epochs': epochs,
    'num_rounds': num_rounds, 'data_file': dataset_used,
    'model': 'VGG16', 'os': platform.system(),
    'selected_clients': 'ALL'
}))

logger.info("----------------------")
logger.info("start federated")
logger.info("----------------------")
federated.start()
