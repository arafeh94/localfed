import logging

from torch import nn

import src
from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.apis import ga
from src.federated.components import testers, client_selectors, aggregators, params, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import plugins, fedruns
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import TrainerManager, SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../datasets/pickles/70_2_600_big_mnist.pkl'
test_file = '../datasets/pickles/test_data.pkl'

logger.info('Generating Data --Started')
dg = src.data.data_generator.load(data_file)
client_data = dg.distributed
# dg = DataGenerator(LocalMnistDataProvider(limit=10000))
# client_data = dg.distribute_size(10, 100, 100)
dg.describe()
logger.info('Generating Data --Ended')

configs = {
    'ga': {
        'batch_size': 25,
        'epochs': 8,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': params.sgd(0.1),
        'clients_per_round': 10,
        'num_rounds': 10,
        'desired_accuracy': 0.99,
        'test_on': FederatedLearning.TEST_ON_ALL,
        'model': lambda: LogisticRegression(28 * 28, 10),
        'ga_max_iter': 10,
        'ga_r_cross': 0.2,
        'ga_r_mut': 0.2,
        'ga_c_size': 10,
        'ga_p_size': 200,
    },
    'normal': {
        'batch_size': 25,
        'epochs': 8,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': params.sgd(0.1),
        'clients_per_round': 10,
        'num_rounds': 10,
        'desired_accuracy': 0.99,
        'test_on': FederatedLearning.TEST_ON_ALL,
        'model': lambda: LogisticRegression(28 * 28, 10),
    }
}
runs = {}
for name, config in configs.items():
    trainer_manager = SeqTrainerManager(trainers.CPUTrainer, batch_size=config['batch_size'], epochs=config['epochs'],
                                        criterion=config['criterion'], optimizer=config['optimizer'])

    initial_model = config['model']
    if 'ga_max_iter' in config:
        initial_model = ga.ga_module_creator(client_data, initial_model, max_iter=config['ga_max_iter'],
                                             r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
                                             c_size=config['ga_c_size'], p_size=config['ga_p_size'])

    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        aggregator=aggregators.AVGAggregator(),
        tester=testers.Normal(batch_size=config['batch_size'], criterion=config['criterion']),
        client_selector=client_selectors.Random(config['clients_per_round']),
        trainers_data_dict=client_data,
        initial_model=initial_model,
        num_rounds=config['num_rounds'],
        desired_accuracy=config['desired_accuracy'],
        test_on=FederatedLearning.TEST_ON_ALL
    )

    federated.plug(plugins.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.plug(plugins.FederatedTimer([Events.ET_TRAINER_FINISHED]))
    # federated.plug(plugins.FedPlot())
    # federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))
    # federated.plug(plugins.FedSave())
    # federated.plug(plugins.WandbLogger(config={'method': 'genetic', 'max_rounds': 10}))

    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()
    runs[name] = federated.context

fd = fedruns.FedRuns(runs)
fd.compare_all()
fd.plot()
