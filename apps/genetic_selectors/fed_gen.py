# mpiexec -n 4 python fed_gen_mpi.py
import logging
import sys
from os.path import dirname

from torch import nn

sys.path.append(dirname(__file__) + '../../')
from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import CPUTrainer
from src.federated.protocols import TrainerParams
import src
from libs.model.cv.cnn import CNN_OriginalFedAvg
from apps.genetic_selectors import fed_comp
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators
from src.data.data_provider import PickleDataProvider
from src.federated import plugins, fedruns
from src.data.data_generator import DataGenerator
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.trainer_manager import MPITrainerManager, SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

batch_size = 30
epochs = 20
clients_per_round = 3
num_rounds = 5
model = lambda: LogisticRegression(28 * 28, 10)
data_file = '../../datasets/pickles/mnist_2shards_70c_600mn_600mx.pkl'

logger.info('Generating Data --Started')
dg = src.data.data_generator.load(data_file)
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
        initial_model = fed_comp.ga_module_creator(client_data, initial_model, max_iter=config['ga_max_iter'],
                                                   r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
                                                   c_size=config['ga_c_size'], p_size=config['ga_p_size'],
                                                   clusters=config['nb_clusters'],
                                                   desired_fitness=config['ga_min_fitness'])
    elif name == 'clustered':
        initial_model = fed_comp.cluster_module_creator(client_data, initial_model, config['nb_clusters'],
                                                        config['c_size'])

    trainer_manager = SeqTrainerManager()
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
    runs[name] = federated.context

fd = fedruns.FedRuns(runs)
fd.compare_all()
fd.plot()
