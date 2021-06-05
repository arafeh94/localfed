# mpiexec -n 11 python compare.py
import copy
import logging
import sys
from os.path import dirname


sys.path.append(dirname(__file__) + '../')
from libs.model.collection import MLP
from src import tools
from src.apis.mpi import Comm
from torch import nn
from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from libs.model.linear.net import Net
from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import CPUTrainer
from src.federated.protocols import TrainerParams
from apps.genetic_selectors.src import initializer
from src.federated.components import metrics, client_selectors, aggregators
from src.federated import plugins, fedruns
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import SeqTrainerManager, MPITrainerManager
from src.data import data_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
is_mpi = False
comm = None
if is_mpi:
    comm = Comm()

if not is_mpi or is_mpi and comm.pid() == 0:
    batch_size = 50
    epochs = 20
    clients_per_round = 10
    num_rounds = 75
    shared_model = MLP(28 * 28, 128, 10)
    model = lambda: copy.deepcopy(shared_model)
    data_file = '../datasets/pickles/mnist_2shards_70c_600mn_600mx.pkl'

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
            'ga_c_size': 10,
            'ga_p_size': 200,
            'nb_clusters': 10,
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
        # 'clustered': {
        #     'batch_size': batch_size,
        #     'epochs': epochs,
        #     'clients_per_round': clients_per_round,
        #     'num_rounds': num_rounds,
        #     'desired_accuracy': 0.99,
        #     'test_on': FederatedLearning.TEST_ON_ALL,
        #     'model': model,
        #     'c_size': 10,
        #     'nb_clusters': 10,
        # },
        'rl': {
            'batch_size': batch_size,
            'epochs': epochs,
            'clients_per_round': clients_per_round,
            'num_rounds': num_rounds,
            'desired_accuracy': 0.99,
            'test_on': FederatedLearning.TEST_ON_ALL,
            'model': model,
        }

    }

    fd = fedruns.FedRuns({})
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
        elif name == 'rl':
            initial_model0 = initial_model()
            booted_model, client_director, w_first = rl_module_creator(client_data, initial_model0)
            client_selector = RLSelector(config['clients_per_round'], client_director, w_first)
            initial_model = booted_model

        trainer_manager = MPITrainerManager() if is_mpi else SeqTrainerManager()
        trainer_params = TrainerParams(trainer_class=CPUTrainer, optimizer='sgd', epochs=epochs, batch_size=batch_size,
                                       criterion='cel', lr=0.1)
        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            trainer_params=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion=nn.CrossEntropyLoss()),
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
        # federated.plug(plugins.FedSave())
        # federated.plug(plugins.WandbLogger(config={'method': 'genetic', 'max_rounds': 10}))
        # federated.plug(plugins.MPIStopPlug())
        logger.info("----------------------")
        logger.info(f"start federated {name}")
        logger.info("----------------------")
        federated.start()
        fd.append(name, federated)

    fd.compare_all()
    fd.plot()
else:
    if is_mpi:
        MPITrainerManager.mpi_trainer_listener(comm)
