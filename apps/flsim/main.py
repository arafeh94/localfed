import logging

from torch import nn

import src
from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.data import data_generator
from src.federated.components import metrics, client_selectors, aggregators, params, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import plugins
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.trainer_manager import TrainerManager, SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../../datasets/pickles/mnist_10shards_100c_400mn_400mx.pkl'

logger.info('Generating Data --Started')
dg = data_generator.load(data_file)
client_data = dg.distributed
dg.describe()
logger.info('Generating Data --Ended')

trainer_params = TrainerParams(trainer_class=trainers.CPUTrainer, batch_size=50, epochs=20, optimizer='sgd',
                               criterion='cel', lr=0.1)

initial_model = LogisticRegression(28 * 28, 10)
booted_model, client_director, w_first = rl_module_creator(client_data, initial_model)
client_selector = RLSelector(10, client_director, w_first)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_params=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selector,
    trainers_data_dict=client_data,
    # initial_model=lambda: LogisticRegression(28 * 28, 10),
    initial_model=booted_model,
    num_rounds=1000,
    desired_accuracy=0.99
)

federated.plug(plugins.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.plug(plugins.FederatedTimer([Events.ET_ROUND_FINISHED]))
# federated.plug(plugins.FedPlot())
# federated.plug(plugins.FedSave())
# federated.plug(plugins.WandbLogger(config={'num_rounds': 10}))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()