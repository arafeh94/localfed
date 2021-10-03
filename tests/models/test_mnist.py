import logging
from collections import defaultdict

import torch
from torch import nn

import libs.model.collection
from libs.model.cv.cnn import CNN_OriginalFedAvg, Net, CNN, CNN_DropOut
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data import data_loader
from src.data.data_generator import load
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider
from src.manifest import WandbAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')

train, test = PickleDataProvider("../../datasets/pickles/mnist.pkl").collect().shuffle().as_tensor().split(0.8)

tools.detail(train)

# model = Net()
# model = CNN_DropOut()
model = CNN_OriginalFedAvg()
# model = LogisticRegression(28*28, 10)
# model = CNN_OriginalFedAvg(False)
# model = resnet56(62, 1, 28)

tools.train(model, train_data=train.batch(32), epochs=1)
acc, loss = tools.infer(model, test.batch(32))
print(acc, loss)

import wandb
import atexit

wandb.login(key=WandbAuth.key)
wandb.init(project=WandbAuth.project, entity=WandbAuth.entity, config=None)
wandb = wandb
atexit.register(lambda: wandb.finish())


def on_round_end(self, params):
    self.wandb.log({'acc': params['accuracy'], 'loss': params['loss'], 'last_round': params['round'] + 1})


def on_federated_ended(self, params):
    self.wandb.finish()

data_loader.mnist_2shards_100c_600min_600max()
