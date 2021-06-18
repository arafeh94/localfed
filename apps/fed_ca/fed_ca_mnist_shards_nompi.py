# to run MPI, u wull first need to change the terminal current working directory to D:\Github\my_repository\localfed\experiments>
# windows: mpiexec -n 11 python fed_ca_mnist_shards.py
# ubuntu: mpirun -np 11 --hostfile hosts python ./fed_ca.py
import logging
import platform
import sys
from os.path import dirname

import matplotlib.pyplot as plt
from torch import nn

from src import tools
from src.data.data_loader import mnist_2shards_100c_600min_600max

if platform.system() == 'Linux':
    # Linux
    sys.path.append(dirname(__file__) + './')
else:
    # windows
    sys.path.append(dirname(__file__) + '../../')

from src.federated import subscribers
from src.federated.components.trainer_manager import MPITrainerManager, SeqTrainerManager

from hp_generator import generate_configs, build_random, calculate_max_rounds
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg, CNN_DropOut
from libs.model.linear.lr import LogisticRegression
from libs.model.collection import MLP
from src.data import data_generator, data_loader
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

# data_file = "../../datasets/pickles/femnist_1shard_10c_1000min_1000max.pkl"
# data_file = "../../datasets/pickles/10_1000_big_ca.pkl"
# data_file = "../datasets/pickles/10_2400_3000_big_imbalanced_ca.pkl"

data_file = "femnist_1s_62c_2000min_2000max"
client_data = data_loader.femnist_1s_62c_2000min_2000max()
tools.detail(client_data)

logger.info('generating data --Started')

# dg = data_generator.load(data_file)
# client_data = dg.distributed

# dg.describe()

# building Hyperparameters
input_shape = 28 * 28
labels_number = 62
percentage_nb_client = 6

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    # 'CNN': CNN_OriginalFedAvg(False)
    'CNN':CNN_DropOut(False)
}

for model_name, gen_model in initial_models.items():

    """
      each params=(min,max,num_value)
    """
    batch_size = (50, 50, 1)
    epochs = (150, 20, 1)
    num_rounds = (1000, 1000, 1)

    hyper_params = build_random(batch_size=batch_size, epochs=epochs, num_rounds=num_rounds)
    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.1

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}, '
            f'initial_model={initial_model} ')
        trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size,
                                       epochs=epochs,
                                       optimizer='sgd', criterion='cel', lr=learn_rate)

        federated = FederatedLearning(
            trainer_manager=SeqTrainerManager(),
            trainer_config=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
            # client_selector=client_selectors.All(),
            client_selector=client_selectors.Random(percentage_nb_client),
            trainers_data_dict=client_data,
            initial_model=lambda: initial_model,
            num_rounds=num_rounds,
            desired_accuracy=0.99
        )

        federated.add_subscriber(subscribers.WandbLogger(config={
            'lr': learn_rate, 'batch_size': batch_size,
            'epochs': epochs,
            'num_rounds': num_rounds, 'data_file': data_file,
            'model': model_name, 'os': platform.system(),
            'selected_clients': percentage_nb_client
        }))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        # name = f"-{randint(0, 999)}"
        # runs[name] = federated.context

    # runs = fedruns.FedRuns(runs)
    # runs.compare_all()
    # runs.plot()
