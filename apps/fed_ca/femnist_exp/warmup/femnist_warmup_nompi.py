# mpiexec -n 11 python femnist_mpi.py

import logging
import pickle
import platform
import sys
from os.path import dirname
from random import randint

from torch import nn

from apps.fed_ca.utilities.load_dataset import LoadData
from src import tools
from src.federated import subscribers, fedruns
from src.federated.components.trainer_manager import SeqTrainerManager

from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg, CNN_DropOut
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

ld = LoadData(dataset_name='femnist', shards_nb=0, clients_nb=62, min_samples=2000, max_samples=2000)
dataset_used = ld.filename
client_data = ld.pickle_distribute_continuous()
tools.detail(client_data)


def load_warmup():
    model = pickle.load(open("warmup_model_femnist200.pkl", 'rb'))
    return model


# building Hyperparameters
input_shape = 28 * 28
labels_number = 62
percentage_nb_client = 0.4

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
     'CNN': CNN_DropOut(False),
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg(False)
}

# runs = {}
for model_name, gen_model in initial_models.items():

    hyper_params = {'batch_size': [10], 'epochs': [5], 'num_rounds': [100_000]}
    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.1

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds},'
            f' initial_model={model_name} ')
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
            initial_model=load_warmup,
            num_rounds=num_rounds,
            desired_accuracy=0.99
        )

        federated.add_subscriber(subscribers.Resumable('warmup_femnist200', federated, flush=False))

        federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))
        federated.add_subscriber(
            subscribers.WandbLogger(config={'lr': learn_rate, 'batch_size': batch_size, 'epochs': epochs,
                                            'num_rounds': num_rounds, 'data_file': dataset_used,
                                            'model': model_name, 'os': platform.system() + '',
                                            'selected_clients': percentage_nb_client}))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        # runs[model_name] = federated.context

# r = fedruns.FedRuns(runs)
# r.plot()
