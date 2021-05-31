import logging

from torch import nn

from src.federated.components import testers, client_selectors, aggregators, optims, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import LocalKDDDataProvider
from src.federated import plugins
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import TrainerManager, SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
dg = DataGenerator(LocalKDDDataProvider())
client_data = dg.distribute_size(10, 5000, 5000)
dg.describe()
logger.info('Generating Data --Ended')

trainer_manager = SeqTrainerManager(trainers.CPUTrainer, batch_size=50, epochs=10, criterion=nn.CrossEntropyLoss(),
                                    optimizer=optims.sgd(0.1))

federated = FederatedLearning(
    trainer_manager=trainer_manager,
    aggregator=aggregators.AVGAggregator(),
    tester=testers.Normal(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(5),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(41, 2),
    num_rounds=50,
    desired_accuracy=0.99
)

federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
federated.plug(plugins.FederatedTimer([Events.ET_TRAINER_FINISHED]))
federated.plug(plugins.FedPlot())
# federated.plug(plugins.FedSave())
# federated.plug(plugins.WandbLogger(config={'num_rounds': 10}))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
