# windows: mpiexec -n 11 python fed_ca.py
# ubuntu: mpirun -np 11 python fed_ca.py
# ubuntu: mpirun -np 11 --hostfile hosts python ./fed_ca.py

import logging
import sys
from os.path import dirname
from random import randint

from torch import nn

sys.path.append(dirname(__file__) + '../')

from src.apis.hp_generator import generate_configs, build_grid, calculate_max_rounds
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src.data import data_generator
from src.federated import plugins, fedruns
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.trainer_manager import TrainerManager, SeqTrainerManager, MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()
if comm.pid() == 0:

    data_file = "../datasets/pickles/10_1000_big_ca.pkl"
    # custom test file contains only 20 samples from each client
    # custom_test_file = '../datasets/pickles/test_data.pkl'

    logger.info('generating data --Started')

    dg = data_generator.load(data_file)
    client_data = dg.distributed
    dg.describe()

    # building Hyperparameters
    hyper_params = build_grid(batch_size=(5, 128, 10), epochs=(1, 20, 5), num_rounds=(5, 80, 10))
    configs = generate_configs(hyper_params)
    for config in configs:
         print(config)

    logger.info(calculate_max_rounds(hyper_params))
    runs = {}
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        learn_rate = 0.1

        print(f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds} ')
        trainer_manager = MPITrainerManager()
        trainer_params = TrainerParams(trainer_class=trainers.CPUChunkTrainer, batch_size=batch_size, epochs=epochs,
                                       optimizer='sgd', criterion='cel', lr=learn_rate)

        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            trainer_params=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
            # client_selector=client_selectors.All(),
            client_selector=client_selectors.Random(3),
            trainers_data_dict=client_data,
            initial_model=lambda: LogisticRegression(28 * 28, 10),
            # initial_model=lambda: CNN_OriginalFedAvg(),
            num_rounds=num_rounds,
            desired_accuracy=0.99
        )

        # federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))
        # federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
        # federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(custom_test_file).collect().as_tensor(), 8))
        # federated.plug(plugins.FedPlot())

        # federated.plug(plugins.FL_CA())
        federated.plug(plugins.WandbLogger(config={'lr': learn_rate, 'batch_size': batch_size, 'epochs': epochs,
                                                   'num_rounds': num_rounds, 'data_file': data_file,
                                                   'model': 'LR', 'os': 'windows'}))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        name = f"-{randint(0, 999)}"
        runs[name] = federated.context

    runs = fedruns.FedRuns(runs)
    # runs.compare_all()
    # runs.plot()
else:
    MPITrainerManager.mpi_trainer_listener(comm)
