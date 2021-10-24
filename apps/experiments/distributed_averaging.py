# mpiexec -n 2 python distributed_averaging.py
import sys
from os.path import dirname

from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer

sys.path.append(dirname(__file__) + '../../')

from src.data import data_loader
import logging
from torch import nn
from src.federated.protocols import TrainerParams
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.linear.lr import LogisticRegression
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()

if comm.pid() == 0:
    logger.info('Generating Data --Started')
    client_data = data_loader.mnist_10shards_100c_400min_400max()
    logger.info('Generating Data --Ended')

    trainer_manager = MPITrainerManager()
    trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=20, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(50, criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(5),
        trainers_data_dict=client_data,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        num_rounds=0,
        desired_accuracy=0.99
    )

    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()
else:
    MPITrainerManager.mpi_trainer_listener(comm)
