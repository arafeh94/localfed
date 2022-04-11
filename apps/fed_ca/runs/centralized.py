import atexit
import logging
from os import path

import wandb

from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src import tools, manifest
from src.apis import lambdas
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload
from src.data.data_provider import PickleDataProvider
from apps.fed_ca.utilities.hp_generator import generate_configs, build_random, calculate_max_rounds

from datetime import datetime

from src.federated.subscribers.wandb_logger import WandbLogger


def initialize_logs():
    start_time = datetime.now()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')
    return logger


def add_wandb_log(project, federated, percentage_nb_client, model_name, batch_size, epochs, num_rounds,
                  learn_rate, dataset_used):
    federated.add_subscriber(WandbLogger(project=project,
                                         config={
                                             'lr': learn_rate,
                                             'batch_size': batch_size,
                                             'epochs': epochs,
                                             'num_rounds': num_rounds,
                                             'data_file': dataset_used,
                                             'model': model_name,
                                             'selected_clients': percentage_nb_client
                                         }))


def get_client_data(dataset_name, labels_number, min_clients_samples, max_clients_samples, reshape=None,
                    distributed=False):
    if distributed:
        client_data = preload(dataset_name)
        # client_data = client_data.filter(lambda x[], y: (y in [0, 1, 3, 4, 5, 8, 15]))
        # client_data[0].filter(lambda x, y: (y in [1]))
    else:
        client_data = preload(dataset_name, UniqueDistributor(labels_number, min_clients_samples, max_clients_samples))

    if reshape is not None:
        client_data = client_data.map(lambdas.reshape(reshape[0])).map(lambdas.transpose(reshape[1]))

    return client_data