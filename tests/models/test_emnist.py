import logging
from collections import defaultdict

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

train, test = PickleDataProvider("../../datasets/pickles/femnist.pkl").collect().shuffle().as_tensor().split(0.8)

tools.detail(train)

# model = CNN_OriginalFedAvg(False)
model = resnet56(62, 1, 28)

tools.train(model, train_data=train.batch(20), epochs=3)
acc, loss = tools.infer(model, test.batch(20))
print(acc, loss)
# data_loader.mnist_2shards_100c_600min_600max()
