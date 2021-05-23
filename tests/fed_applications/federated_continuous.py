import logging

from torch import nn

from components import trainers, aggregators, testers, client_selectors, optims
from libs.model.linear.lr import LogisticRegression
from src.data import data_generator
from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider
from src.federated import plugins
from src.federated.federated import Events, FederatedLearning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = "../../datasets/pickles/continuous_unbalanced.pkl"
test_file = '../../datasets/pickles/test_data.pkl'

logger.info('generating data --Started')

dg = data_generator.load(data_file)
client_data = dg.distributed
dg.describe()

federated = FederatedLearning(
    trainer=trainers.CPUChunkTrainer(batch_size=20, epochs=30, criterion=nn.CrossEntropyLoss(),
                                     optimizer=optims.sgd(0.1)),
    aggregator=aggregators.AVGAggregator(),
    tester=testers.Normal(batch_size=8, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.All(),
    trainers_data_dict=client_data,
    create_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=50,
    desired_accuracy=0.99
)

federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
federated.plug(plugins.FedPlot())
federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))

logger.info("----------------------")
logger.info("start federated")
logger.info("----------------------")
federated.start()
