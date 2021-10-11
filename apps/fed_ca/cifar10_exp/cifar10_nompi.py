# mpiexec -n 11 python cifar10_nompi.py

import logging
import platform
import sys
from os.path import dirname

from torch import nn

from apps.fed_ca.cifar10_exp.cifar10_models import CNN_85
from apps.fed_ca.utilities.load_dataset import LoadData
from libs.model.collection import CNNCifar
from libs.model.cv.cnn import CNN_DropOut, SimpleCNN
from libs.model.cv.resnet import resnet56, CNN_batch_norm_cifar10
from src import tools
from src.apis import lambdas
from src.federated import subscribers
from src.federated.components.trainer_manager import SeqTrainerManager

sys.path.append(dirname(__file__) + '../')

from apps.fed_ca.utilities.hp_generator import generate_configs, build_random, calculate_max_rounds
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

ld = LoadData(dataset_name='cifar10', shards_nb=0, clients_nb=10, min_samples=6000, max_samples=6000)
dataset_used = ld.filename
client_data = ld.pickle_distribute_continuous()
tools.detail(client_data)
client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

# building Hyperparameters
input_shape = 32 * 32
labels_number = 10
percentage_nb_client = 10

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    # 'CNNCifar':CNNCifar(labels_number)
    #  'CNN': CNN_DropOut(False)
    # 'ResNet': resnet56(labels_number, 3, 32)
    # 'SimpleCNN': SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
    # 'Cifar10': CNN_batch_norm_cifar10()
    'CNN_85': CNN_85()
}


# runs = {}

for model_name, gen_model in initial_models.items():

    hyper_params = {'batch_size': [64], 'epochs': [10], 'num_rounds': [10000]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.01

        print("------------------------------------------------------------------------------------------------")
        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds},'
            f' initial_model={model_name} ')
        print("------------------------------------------------------------------------------------------------")

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
            num_rounds=num_rounds,
            desired_accuracy=0.99
        )

        # use flush=True if you don't want to continue from the last round
        # federated.add_subscriber(
        #     subscribers.Resumable('cifar10_batch_normalization_test_10c_6000.pkl', federated, flush=True))

        # show weight divergence in each round
        # federated.add_subscriber(subscribers.ShowWeightDivergence(save_dir='./pics'))
        # show data distrubition of each clients
        # federated.add_subscriber(subscribers.ShowDataDistribution(label_count=62, save_dir='./pics'))

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
