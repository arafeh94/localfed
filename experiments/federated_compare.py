import logging

from torch import nn

import src
from components import client_selectors, aggregators, trainers, testers, optims
from components.trainers import CPUTrainer
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import plugins, fedruns
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.fedruns import FedRuns
from src.federated.trainer_manager import TrainerManager, ADVTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../datasets/pickles/2_50_medium_shards.pkl'
test_file = '../datasets/pickles/test_data.pkl'

logger.info('Generating Data --Started')
dg = src.data.data_generator.load(data_file)
client_data = dg.distributed
dg.describe()
logger.info('Generating Data --Ended')

federated_configs = {
    'first': {
        'batch_size': 8,
        'epochs': 10,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': optims.sgd(0.1),
        'num_clients': 10,
        'num_rounds': 3,
        'desired_accuracy': 0.99,
        'model_init': lambda: LogisticRegression(28 * 28, 10),
        'clients_data': client_data,
    },
    'second': {
        'batch_size': 8,
        'epochs': 3,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': optims.sgd(0.1),
        'num_clients': 10,
        'num_rounds': 3,
        'desired_accuracy': 0.99,
        'model_init': lambda: LogisticRegression(28 * 28, 10),
        'clients_data': client_data,
    },
    'third': {
        'batch_size': 18,
        'epochs': 2,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': optims.sgd(0.1),
        'num_clients': 10,
        'num_rounds': 10,
        'desired_accuracy': 0.99,
        'model_init': lambda: LogisticRegression(28 * 28, 10),
        'clients_data': client_data,
    },
}

federated_runs = {}

for name, federated_params in federated_configs.items():
    trainer_manager = TrainerManager(
        trainers.CPUTrainer,
        batch_size=federated_params['batch_size'],
        epochs=federated_params['epochs'],
        criterion=federated_params['criterion'],
        optimizer=federated_params['optimizer'],
    )

    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        aggregator=aggregators.AVGAggregator(),
        tester=testers.Normal(batch_size=federated_params['batch_size'], criterion=federated_params['criterion']),
        client_selector=client_selectors.Random(federated_params['num_clients']),
        trainers_data_dict=federated_params['clients_data'],
        initial_model=federated_params['model_init'],
        num_rounds=federated_params['num_rounds'],
        desired_accuracy=federated_params['desired_accuracy']
    )

    federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED]))
    federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(),
                                               federated_params['batch_size'], False))

    logger.info("----------------------")
    logger.info("start federated " + name)
    logger.info("----------------------")
    federated.start()
    federated_runs[name] = federated.context

runs = FedRuns(federated_runs)
runs.compare_all()
runs.plot()
