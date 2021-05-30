import logging
import pickle

from torch import nn

import src
from components import client_selectors, aggregators, trainers, testers, optims
from components.trainers import CPUTrainer
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider, LocalMnistDataProvider
from src.federated import plugins
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import TrainerManager, ADVTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../datasets/pickles/2_50_medium_shards.pkl'
test_file = '../datasets/pickles/test_data.pkl'

logger.info('Generating Data --Started')
# dg = src.data.data_generator.load(data_file)
# client_data = dg.distributed
dg = DataGenerator(LocalMnistDataProvider(limit=10000))
client_data = dg.distribute_size(10, 100, 100)
dg.describe()
logger.info('Generating Data --Ended')

trainer_manager = TrainerManager(trainers.CPUTrainer, batch_size=8, epochs=10, criterion=nn.CrossEntropyLoss(),
                                 optimizer=optims.sgd(0.1))

federated = FederatedLearning(
    trainer_manager=trainer_manager,
    aggregator=aggregators.AVGAggregator(),
    tester=testers.Normal(batch_size=8, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(10),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=10,
    desired_accuracy=0.99
)

federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
# federated.plug(plugins.FedPlot())
# federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))
# federated.plug(plugins.FedSave())
# federated.plug(plugins.WandbLogger(config={'num_rounds': 10}))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
