import logging
from collections import defaultdict
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

import libs.model.collection
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data import data_loader
from src.data.data_generator import load
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider

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
for trainer_id, data in client_data.items():
    data = data.shuffle().as_tensor()
    train, test = data.split(0.8)

    ui.append(torch.mean(train.x))
    ei.append(torch.var(train.x, unbiased=False))

    mean_value = torch.mean(train.x).numpy().max()
    variance_value = torch.var(train.x, unbiased=False).numpy().max()
    x = random.normal(loc=mean_value, scale=variance_value, size=(1, 1000))

    sns.distplot(x, hist=False)
    plt.show()





# model = CNN_OriginalFedAvg(False)
#
#
#
# tools.train(model, train_data=train.batch(20), epochs=3)
# acc, loss = tools.infer(model, test.batch(20))
# print(acc, loss)