# mpiexec -n 11 python distributed.py
import sys
from os.path import dirname

sys.path.append(dirname(__file__) + '../')
import logging
from torch import nn
import src
from src.federated.protocols import TrainerParams
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import subscribers
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()

if comm.pid() == 0:
    data_file = '../datasets/pickles/mnist_10shards_100c_400mn_400mx.pkl'
    test_file = '../datasets/pickles/test_data.pkl'

    logger.info('Generating Data --Started')
    dg = src.data.data_generator.load(data_file)
    client_data = dg.distributed
    dg.describe()
    logger.info('Generating Data --Ended')

    trainer_manager = MPITrainerManager()
    trainer_params = TrainerParams(trainer_class=trainers.CPUTrainer, batch_size=50, epochs=20, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(50, criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(5),
        trainers_data_dict=client_data,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        # initial_model=lambda: CNN_OriginalFedAvg(),
        num_rounds=0,
        desired_accuracy=0.99
    )

    federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(subscribers.FederatedTimer([Events.ET_TRAINER_FINISHED]))
    federated.add_subscriber(subscribers.FedPlot())
    # federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))
    # federated.plug(plugins.FedSave())
    # federated.plug(plugins.WandbLogger(config={'num_rounds': 10}))
    # federated.plug(plugins.MPIStopPlug())

    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()
else:
    MPITrainerManager.mpi_trainer_listener(comm)
