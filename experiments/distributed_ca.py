# mpiexec -n 11 py distributed_ca.py
import sys
from os.path import dirname

from src.federated.protocols import TrainerParams

sys.path.append(dirname(__file__) + '../')
import logging
from torch import nn
import src
from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.apis.mpi import Comm
from src.federated.components import testers, client_selectors, aggregators, optims, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import plugins
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()

config = {
    'criterion': nn.CrossEntropyLoss(),
    'batch_size': 128,
    'epochs': 100,
    'optimizer': optims.sgd(0.1)
}
if comm.pid() == 0:
    data_file = '../datasets/pickles/10_6000_big_ca.pkl'
    test_file = '../datasets/pickles/test_data.pkl'

    logger.info('Generating Data --Started')
    dg = src.data.data_generator.load(data_file)
    client_data = dg.distributed
    dg.describe()
    logger.info('Generating Data --Ended')

    trainer_manager = MPITrainerManager()
    trainer_params = TrainerParams(trainer_class=trainers.CPUChunkTrainer, batch_size=50, epochs=20, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_params=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        tester=testers.Normal(config['batch_size'], criterion=config['criterion']),
        client_selector=client_selectors.All(),
        trainers_data_dict=client_data,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        num_rounds=41,
        desired_accuracy=0.99
    )

    federated.plug(plugins.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.plug(plugins.FederatedTimer([Events.ET_TRAINER_FINISHED]))
    # federated.plug(plugins.FedPlot())
    # federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))
    # federated.plug(plugins.FedSave())
    # federated.plug(plugins.MPIStopPlug())
    federated.plug(plugins.WandbLogger(config=config))

    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()
else:
    MPITrainerManager.mpi_trainer_listener(comm)
