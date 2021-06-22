import logging
import sys

from torch import nn

from libs.model.cv.cnn import CNN
from src.data.data_provider import PickleDataProvider

sys.path.append('../../')
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider
from src.federated.subscribers import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = data_loader.mnist_1shards_100c_600min_600max()
logger.info('Generating Data --Ended')

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=1, optimizer='sgd',
                               criterion='cel', lr=0.1)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(5),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=5,
    desired_accuracy=0.99,
)

# federated.add_subscriber(subscribers.Resumable('agv', federated, flush=True))
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
# federated.add_subscriber(subscribers.FedPlot())
# federated.add_subscriber(subscribers.FedSave('avgs'))
federated.add_subscriber(subscribers.ShowWeightDivergence())
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
