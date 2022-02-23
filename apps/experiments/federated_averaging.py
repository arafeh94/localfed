import logging
import sys

sys.path.append('../../')

from torch import nn
from src.federated.subscribers.fed_plots import RoundAccuracy, RoundLoss
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from libs.model.linear.lr import LogisticRegression
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('mnist', LabelDistributor(num_clients=100, label_per_client=5, min_size=600, max_size=600))
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=25, optimizer='sgd',
                               criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.4),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=50,
    desired_accuracy=0.99,
)

# (4)
federated.add_subscriber(FederatedLogger())
federated.add_subscriber(Timer())
federated.add_subscriber(RoundAccuracy(plot_ratio=1))
federated.add_subscriber(RoundLoss(plot_ratio=1))

# (5)
# RamTracker(delay=1).attach(federated)
# CPUTracker(delay=1).attach(federated)

logger.info("----------------------")
logger.info("start federated learning")
logger.info("----------------------")
federated.start()
