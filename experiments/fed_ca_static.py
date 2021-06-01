import logging

from torch import nn

import libs.model.collection
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src.data import data_generator
from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider
from src.federated import plugins
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.trainer_manager import TrainerManager, SeqTrainerManager

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = "../datasets/pickles/10_6000_big_ca.pkl"
# custom test file contains only 20 samples from each client
# custom_test_file = '../datasets/pickles/test_data.pkl'

logger.info('generating data --Started')

dg = data_generator.load(data_file)
client_data = dg.distributed
dg.describe()

# # setting hyper parameters
batch_size = 64
epochs = 20
num_rounds = 20
learn_rate = 0.1
optimizer = 'sgd'
criterion = 'cel'
print(f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}')
trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=trainers.CPUChunkTrainer, batch_size=batch_size, epochs=epochs,
                               optimizer=optimizer, criterion=criterion, lr=learn_rate)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_params=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.All(),
    trainers_data_dict=client_data,
    # initial_model=lambda: LogisticRegression(28 * 28, 10),
    initial_model=lambda: libs.model.collection.MLP(28 * 28, 64, 10),
    # initial_model=lambda: CNN_OriginalFedAvg(),
    num_rounds=num_rounds,
    desired_accuracy=0.99
)

federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED]))
# federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
# federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(custom_test_file).collect().as_tensor(), 8))
# federated.plug(plugins.FedPlot())

# federated.plug(plugins.FL_CA())
federated.plug(plugins.WandbLogger(config={'lr': learn_rate, 'batch_size': batch_size, 'epochs': epochs,
                                           'num_rounds': num_rounds, 'data_file': data_file,
                                           'model': 'CNN'}))

logger.info("----------------------")
logger.info("start federated")
logger.info("----------------------")
federated.start()
