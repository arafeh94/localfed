import logging

from torch import nn

import src
from components import client_selectors, aggregators, trainers, testers, optims
from components.trainers import CPUTrainer
from libs.model.linear.lr import LogisticRegression
from libs.model.nlp.rnn import RNN_OriginalFedAvg
from src.data.data_provider import PickleDataProvider, LocalShakespeareDataProvider
from src.federated import plugins
from src.data.data_generator import DataGenerator
from src.federated.federated import Events as et
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import TrainerManager, ADVTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../datasets/pickles/2_50_medium_shards.pkl'
test_file = '../datasets/pickles/test_data.pkl'

logger.info('Generating Data --Started')
dg = DataGenerator(LocalShakespeareDataProvider(limit=1000))
client_data = dg.distribute_size(10, 10, 100)
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
    initial_model=lambda: RNN_OriginalFedAvg(),
    num_rounds=3,
    desired_accuracy=0.99
)

federated.plug(plugins.FederatedLogger([
    et.ET_ROUND_FINISHED, et.ET_TRAINER_SELECTED, et.ET_TRAINER_STARTED, et.ET_TRAINER_ENDED]))
federated.plug(plugins.FederatedTimer([et.ET_TRAINER_ENDED, et.ET_TRAIN_END]))
federated.plug(plugins.FedPlot())
# federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
