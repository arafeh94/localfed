import logging
import sys
from os.path import dirname

from torch import nn

sys.path.append(dirname(__file__) + './')

import src
from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.apis import ga
from src.apis.mpi import Comm
from src.federated.components import testers, client_selectors, aggregators, optims, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import plugins, fedruns
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import TrainerManager, SeqTrainerManager, MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()
batch_size = 5000
epochs = 1
clients_per_round = 3
num_rounds = 20
model = lambda: LogisticRegression(28 * 28, 10)
if comm.pid() == 0:
    data_file = '../datasets/pickles/70_2_600_big_mnist.pkl'
    test_file = '../datasets/pickles/test_data.pkl'

    logger.info('Generating Data --Started')
    dg = src.data.data_generator.load(data_file)
    client_data = dg.distributed
    dg.describe()
    logger.info('Generating Data --Ended')

    configs = {
        'ga': {
            'batch_size': batch_size,
            'epochs': epochs,
            'criterion': nn.CrossEntropyLoss(),
            'optimizer': optims.sgd(0.1),
            'clients_per_round': clients_per_round,
            'num_rounds': num_rounds,
            'desired_accuracy': 0.99,
            'test_on': FederatedLearning.TEST_ON_ALL,
            'model': model,
            'ga_max_iter': 10,
            'ga_r_cross': 0.05,
            'ga_r_mut': 0.1,
            'ga_c_size': 10,
            'ga_p_size': 200,
            'nb_clusters': 10,
            'ga_min_fitness': 0.2,
        },
        'normal': {
            'batch_size': batch_size,
            'epochs': epochs,
            'criterion': nn.CrossEntropyLoss(),
            'optimizer': optims.sgd(0.1),
            'clients_per_round': clients_per_round,
            'num_rounds': num_rounds,
            'desired_accuracy': 0.99,
            'test_on': FederatedLearning.TEST_ON_ALL,
            'model': model,
        },
        'clustered': {
            'batch_size': batch_size,
            'epochs': epochs,
            'criterion': nn.CrossEntropyLoss(),
            'optimizer': optims.sgd(0.1),
            'clients_per_round': clients_per_round,
            'num_rounds': num_rounds,
            'desired_accuracy': 0.99,
            'test_on': FederatedLearning.TEST_ON_ALL,
            'model': model,
            'c_size': 10,
            'nb_clusters': 10,
        },
    }
    runs = {}
    for name, config in configs.items():
        trainer_manager = MPITrainerManager()

        initial_model = config['model']
        if name == 'ga':
            initial_model = ga.ga_module_creator(client_data, initial_model, max_iter=config['ga_max_iter'],
                                                 r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
                                                 c_size=config['ga_c_size'], p_size=config['ga_p_size'],
                                                 clusters=config['nb_clusters'],
                                                 desired_fitness=config['ga_min_fitness'])
        elif name == 'clustered':
            initial_model = ga.cluster_module_creator(client_data, initial_model, config['nb_clusters'],
                                                      config['c_size'])

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
        # federated.plug(plugins.MPIStopPlug())
        logger.info("----------------------")
        logger.info(f"start federated {name}")
        logger.info("----------------------")
        federated.start()
        runs[name] = federated.context

    fd = fedruns.FedRuns(runs)
    fd.compare_all()
    fd.plot()
else:
    while True:
        model, train_data, context = comm.recv(0, 1)
        trainer = trainers.CPUTrainer(optimizer=optims.sgd(0.1), epochs=epochs,
                                      batch_size=batch_size, criterion=nn.CrossEntropyLoss())
        trained_weights, sample_size = trainer.train(model, train_data, context)
        comm.send(0, (trained_weights, sample_size), 2)
