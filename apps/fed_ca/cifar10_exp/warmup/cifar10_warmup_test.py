import logging
import platform
import sys

from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.resnet import CNN_Cifar10
from src import tools
from src.data.data_loader import preload

sys.path.append('../../../')

from typing import Callable
from torch import nn
from src.apis import lambdas, files
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider
from src.federated.subscribers import Timer, ShowWeightDivergence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
tag = 'federated_cifar10_240'
is_warmup = True

logger.info('Generating Data --Started')
client_data = data_loader.cifar10_10c_6000min_6000max()
logger.info('Generating Data --Ended')

dataset_used_wd = 'federated_cifar10_240'

# building Hyperparameters
input_shape = 32 * 32
labels_number = 10
percentage_nb_client = 10

if is_warmup:
    warmup_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.05)[0]).reduce(lambdas.dict2dc).as_tensor()
    tools.detail(warmup_data)
    warmup_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[1]).map(lambdas.as_tensor)

    tools.detail(client_data)
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    initial_model = TorchModel(CNN_Cifar10())
    initial_model.train(warmup_data.batch(100), epochs=1)
    initial_model = initial_model.extract()
    initial_models = {'CNN_Cifar10()': initial_model}
else:
    initial_models = {'CNN_Cifar10()': CNN_Cifar10() }


for model_name, gen_model in initial_models.items():

    hyper_params = {'batch_size': [100], 'epochs': [1], 'num_rounds': [800], 'learn_rate': [0.01]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = config['learn_rate']

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
            # accepted_accuracy_margin=0.05
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
                                            'num_rounds': num_rounds, 'data_file': dataset_used_wd,
                                            'model': model_name, 'os': platform.system() + '',
                                            'selected_clients': percentage_nb_client}))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        # runs[model_name] = federated.context

# r = fedruns.FedRuns(runs)
# r.plot()
