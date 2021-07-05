# to run MPI, u wull first need to change the terminal current working directory to D:\Github\my_repository\localfed\experiments>
# windows: mpiexec -n 63 python femnist_mpi.py
import logging
import platform
import sys
from os.path import dirname
from torch import nn

# windows
sys_path = sys.path.append(dirname(__file__) + '../../../')
print(dirname(__file__))

from src import tools
from src.federated import subscribers
from src.federated.components.trainer_manager import MPITrainerManager
from apps.fed_ca.utilities.hp_generator import generate_configs, build_random, calculate_max_rounds
from src.apis.mpi import Comm
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from libs.model.collection import MLP
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from apps.fed_ca.utilities.load_dataset import LoadData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()
if comm.pid() == 0:

    ld = LoadData(dataset_name='femnist', shards_nb=0, clients_nb=62, min_samples=60000, max_samples=60000)
    dataset_used = ld.filename
    client_data = ld.pickle_distribute_continuous()
    tools.detail(client_data)

    # building Hyperparameters
    input_shape = 28 * 28
    labels_number = 62
    percentage_nb_client = 0.4

    # number of models that we are using
    initial_models = {
        # 'LR': LogisticRegression(input_shape, labels_number),
        # 'MLP': MLP(input_shape, labels_number)
        'CNN': CNN_OriginalFedAvg(False)
    }

    for model_name, gen_model in initial_models.items():

        """
          each params=(min,max,num_value)
        """
        batch_size = (64, 64, 1)
        epochs = (5, 5, 1)
        num_rounds = (1000, 1000, 1)

        hyper_params = build_random(batch_size=batch_size, epochs=epochs, num_rounds=num_rounds)
        configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

        logger.info(calculate_max_rounds(hyper_params))
        runs = {}
        for config in configs:
            batch_size = config['batch_size']
            epochs = config['epochs']
            num_rounds = config['num_rounds']
            initial_model = config['initial_model']
            learn_rate = 0.1

            print(
                f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}, '
                f'initial_model={initial_model} ')
            trainer_manager = MPITrainerManager()
            trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size,
                                           epochs=epochs,
                                           optimizer='sgd', criterion='cel', lr=learn_rate)

            federated = FederatedLearning(
                trainer_manager=trainer_manager,
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
                'num_rounds': num_rounds, 'data_file': dataset_used,
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
else:
    MPITrainerManager.mpi_trainer_listener(comm)
