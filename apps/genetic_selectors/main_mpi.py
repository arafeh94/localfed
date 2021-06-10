# mpiexec -n 4 python main_mpi.py
import logging
import sys
from os.path import dirname

from torch import nn

from src.federated.subscribers import Timer

sys.path.append(dirname(__file__) + '../../')
from src.data import data_generator
from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import CPUTrainer
from src.federated.protocols import TrainerParams
from apps.genetic_selectors.src import initializer
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators
from src.federated import subscribers, fedruns
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()
batch_size = 30
epochs = 20
clients_per_round = 3
num_rounds = 20
model = lambda: LogisticRegression(28 * 28, 10)
if comm.pid() == 0:
    data_file = '../../datasets/pickles/mnist_2shards_70c_600mn_600mx.pkl'
    test_file = '../../datasets/pickles/test_data.pkl'

    logger.info('Generating Data --Started')
    dg = data_generator.load(data_file)
    client_data = dg.distributed
    dg.describe()
    logger.info('Generating Data --Ended')

    configs = {
        'ga': {
            'batch_size': batch_size,
            'epochs': epochs,
            'clients_per_round': clients_per_round,
            'num_rounds': num_rounds,
            'desired_accuracy': 0.99,
            'test_on': FederatedLearning.TEST_ON_ALL,
            'model': model,
            'ga_max_iter': 10,
            'ga_r_cross': 0.05,
            'ga_r_mut': 0.1,
            'ga_c_size': 30,
            'ga_p_size': 200,
            'nb_clusters': 30,
            'ga_min_fitness': 0.45,
        },
        'normal': {
            'batch_size': batch_size,
            'epochs': epochs,
            'clients_per_round': clients_per_round,
            'num_rounds': num_rounds,
            'desired_accuracy': 0.99,
            'test_on': FederatedLearning.TEST_ON_ALL,
            'model': model,
        },
        'clustered': {
            'batch_size': batch_size,
            'epochs': epochs,
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

        initial_model = config['model']
        if name == 'ga':
            initial_model = initializer.ga_module_creator(client_data, initial_model, max_iter=config['ga_max_iter'],
                                                          r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
                                                          c_size=config['ga_c_size'], p_size=config['ga_p_size'],
                                                          clusters=config['nb_clusters'],
                                                          desired_fitness=config['ga_min_fitness'])
        elif name == 'clustered':
            initial_model = initializer.cluster_module_creator(client_data, initial_model, config['nb_clusters'],
                                                               config['c_size'])

        trainer_manager = MPITrainerManager()
        trainer_params = TrainerParams(trainer_class=CPUTrainer, optimizer='sgd', epochs=epochs, batch_size=batch_size,
                                       criterion='cel', lr=0.1)
        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            trainer_config=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion=nn.CrossEntropyLoss()),
            client_selector=client_selectors.Random(config['clients_per_round']),
            trainers_data_dict=client_data,
            initial_model=initial_model,
            num_rounds=config['num_rounds'],
            desired_accuracy=config['desired_accuracy'],
            test_on=FederatedLearning.TEST_ON_ALL
        )

        federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
        federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
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
    MPITrainerManager.mpi_trainer_listener(comm)
