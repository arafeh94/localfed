import logging

from torch import nn

from src.federated.components import testers, client_selectors, aggregators, optims, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src.data import data_generator
from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider
from src.federated import plugins
from src.federated.federated import Events, FederatedLearning
from src.federated.trainer_manager import TrainerManager

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = "../datasets/pickles/10_1000_big_ca.pkl"
# custom test file contains only 20 samples from each client
# custom_test_file = '../datasets/pickles/test_data.pkl'

logger.info('generating data --Started')

dg = data_generator.load(data_file)
client_data = dg.distributed
dg.describe()

# # setting hyper parameters
criterion = nn.CrossEntropyLoss()

# Dictionary to change the two hyper parameters we are focusing on here.
hyper_params = {
    # "learning_rate": [.1],
    "batch_size": [None],
    "epochs": [None],
    "num_rounds": [None]
}

batch_sizes_min, batch_size_max = 8, 512
epochs_min, epochs_max = 5, 100
num_rounds_min, num_rounds_max = 5, 120
# learning_rate_min, learning_rate_max = 0.01, 0.2
learn_rate = 0.001

# How many parameters should we generate form the ranges above
no_parameters = 10

print(f"Number of runs for GridSearch will be:            {no_parameters ** 3}")

# Grid Search

# hyper_params["learning_rate"] = np.linspace(learning_rate_min, learning_rate_max, no_parameters)
hyper_params["batch_size"] = [int(n) for n in np.linspace(batch_sizes_min, batch_size_max, no_parameters)]
hyper_params["epochs"] = [int(n) for n in np.linspace(epochs_min, epochs_max, no_parameters)]
hyper_params["num_rounds"] = [int(n) for n in np.linspace(num_rounds_min, num_rounds_max, no_parameters)]

counter = 0
for k, hp_value in hyper_params.items():

    print(f"{k}\n{hp_value}")
    hyper_params[k] = hp_value
    for i in range(no_parameters):
        # j is the length of the hyper parameters in the dictionar and it will be used to loop through all the values of the parameters
        for j in range(len(hp_value)):
            counter = counter + 1
            # if k == 'learning_rate':
            #     learn_rate = hp_value[i]
            #     batch_size = hyper_params["batch_size"][j]
            #     epochs = hyper_params["epochs"][j]
            #     num_rounds = hyper_params["num_rounds"][j]
            # else:
            if k == 'batch_size':
                # learn_rate = hyper_params["learning_rate"][j]
                batch_size = hp_value[i]
                epochs = hyper_params["epochs"][j]
                num_rounds = hyper_params["num_rounds"][j]
            else:
                if k == 'epochs':
                    # learn_rate = hyper_params["learning_rate"][j]
                    batch_size = hyper_params["batch_size"][j]
                    epochs = hp_value[i]
                    num_rounds = hyper_params["num_rounds"][j]
                else:
                    # learn_rate = hyper_params["learning_rate"][j]
                    batch_size = hyper_params["batch_size"][j]
                    epochs = hyper_params["epochs"][j]
                    num_rounds = hp_value[i]

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds} ')
        trainer_manager = TrainerManager(trainers.CPUChunkTrainer, batch_size=batch_size, epochs=epochs,
                                         criterion=criterion,
                                         optimizer=optims.sgd(learn_rate))

        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            aggregator=aggregators.AVGAggregator(),
            tester=testers.Normal(batch_size=batch_size, criterion=criterion),
            client_selector=client_selectors.All(),
            trainers_data_dict=client_data,
            # initial_model=lambda: LogisticRegression(28 * 28, 10),
            initial_model=lambda: CNN_OriginalFedAvg(),
            num_rounds=num_rounds,
            desired_accuracy=0.99
        )

        # federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
        # federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
        # federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(custom_test_file).collect().as_tensor(), 8))
        # federated.plug(plugins.FedPlot())

        # federated.plug(plugins.FL_CA())
        federated.plug(plugins.WandbLogger(config={'lr': learn_rate, 'batch_size': batch_size, 'epochs': epochs,
                                                   'num_rounds': num_rounds, 'data_file': data_file,
                                                   'model': 'CNN'}))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        print()
