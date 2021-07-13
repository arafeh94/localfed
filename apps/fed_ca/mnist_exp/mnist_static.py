# windows: mpiexec -n 11 python mnist_static.py
import logging
import platform
import sys
from os.path import dirname

import libs
from libs.model.collection import MLP

sys.path.append(dirname(__file__) + '../../../')

from torch import nn
from apps.fed_ca.utilities.load_dataset import LoadData
from src import tools
from src.federated import subscribers
from src.federated.components.trainer_manager import MPITrainerManager, SeqTrainerManager
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

# comm = Comm()
#
# if comm.pid() == 0:

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

ld = LoadData(dataset_name='mnist', shards_nb=0, clients_nb=10, min_samples=1000, max_samples=1000)
dataset_used = ld.filename
client_data = ld.pickle_distribute_continuous()
tools.detail(client_data)
number_of_clients_per_round = 2

# # setting hyper parameters
batch_size = 10
epochs = 5
num_rounds = 1000
learn_rate = 0.1
optimizer = 'sgd'
criterion = 'cel'
print(f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}')
trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size, epochs=epochs,
                               optimizer=optimizer, criterion=criterion, lr=learn_rate)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(number_of_clients_per_round),
    trainers_data_dict=client_data,
    # initial_model=lambda: LogisticRegression(28 * 28, 10),
    initial_model=lambda: MLP(28 * 28, 64, 10),
    # initial_model=lambda: CNN_OriginalFedAvg(),
    num_rounds=num_rounds,
    desired_accuracy=0.99
)

# federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED]))
# federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
# federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(custom_test_file).collect().as_tensor(), 8))
# federated.plug(plugins.FedPlot())

# federated.plug(plugins.FL_CA())
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))
federated.add_subscriber(subscribers.WandbLogger(config={
    'lr': learn_rate, 'batch_size': batch_size,
    'epochs': epochs,
    'num_rounds': num_rounds, 'data_file': dataset_used,
    'model': 'MLP', 'os': platform.system(),
    'selected_clients': number_of_clients_per_round
}))

logger.info("----------------------")
logger.info("start federated")
logger.info("----------------------")
federated.start()
# else:
#     MPITrainerManager.mpi_trainer_listener(comm)
