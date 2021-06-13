# mpiexec -n 11 python fed_ca.py

import logging
import platform
import sys
from os.path import dirname
from random import randint

from torch import nn

from src.federated import subscribers, fedruns
from src.federated.components.trainer_manager import SeqTrainerManager

sys.path.append(dirname(__file__) + '../')

from hp_generator import generate_configs, build_random, calculate_max_rounds
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg, CNN_DropOut
from libs.model.linear.lr import LogisticRegression
from libs.model.collection import MLP
from src.data import data_generator, data_loader
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = "femnist"

logger.info('generating data --Started')
client_data = data_loader.femnist_1shard_62c_200min_2000max()

# building Hyperparameters
input_shape = 28 * 28
labels_number = 62
percentage_nb_client = 62

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    'CNN': CNN_DropOut(False)
}

runs = {}

for model_name, gen_model in initial_models.items():

    """
      each params=(min,max,num_value)
    """
    batch_size = (20, 20, 2)
    epochs = (100, 100, 1)
    num_rounds = (1000, 1000, 1)

    hyper_params = build_random(batch_size=batch_size, epochs=epochs, num_rounds=num_rounds)
    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)
    # for config in configs:
    #     print(config)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.001

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds},'
            f' initial_model={initial_model} ')
        trainer_manager = SeqTrainerManager()
        trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size,
                                       epochs=epochs, optimizer='sgd', criterion='cel', lr=learn_rate)

        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            trainer_config=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
            # client_selector=client_selectors.All(),
            client_selector=client_selectors.Random(percentage_nb_client),
            trainers_data_dict=client_data,
            initial_model=lambda: initial_model,
            # initial_model=lambda: libs.model.collection.MLP(28 * 28, 64, 10),
            # initial_model=lambda: LogisticRegression(28 * 28, 10),
            # initial_model=lambda: CNN_OriginalFedAvg(),
            num_rounds=num_rounds,
            desired_accuracy=0.99
        )

        federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))
        federated.add_subscriber(
            subscribers.WandbLogger(config={'lr': learn_rate, 'batch_size': batch_size, 'epochs': epochs,
                                            'num_rounds': num_rounds, 'data_file': data_file,
                                            'model': model_name, 'os': platform.system() + '',
                                            'selected_clients': percentage_nb_client}))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        runs[model_name] = federated.context

r = fedruns.FedRuns(runs)
r.plot()
