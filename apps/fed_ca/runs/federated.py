import logging

from torch import nn

from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import resnet56, CNN_batch_norm_cifar10, CNN_Cifar10
from src.apis import lambdas
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload
from src.federated.components import aggregators, metrics, client_selectors, trainers
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.wandb_logger import WandbLogger
from src.federated.components.trainer_manager import SeqTrainerManager


def initialize_logs():
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


def initialize_subscribers(federated):
    federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))


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


def initialize_exp(dataset_name, labels_number, min_clients_samples, max_clients_samples,
                   percentage_nb_client, model, batch_size, epochs, num_rounds, learn_rate, project, reshape=None,
                   distributed=False):
    logger = initialize_logs()

    client_data = get_client_data(dataset_name, labels_number, min_clients_samples, max_clients_samples, reshape,
                                  distributed)

    hyper_params = {'batch_size': [batch_size], 'epochs': [epochs], 'num_rounds': [num_rounds],
                    'learn_rate': [learn_rate]}

    logger.info(calculate_max_rounds(hyper_params))

    configs = generate_configs(model_param=model, hyper_params=hyper_params)

    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = config['learn_rate']

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}, '
            f'initial_model={type(model).__name__} ')

        trainer_manager = SeqTrainerManager()
        trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size,
                                       epochs=epochs,
                                       optimizer='sgd', criterion='cel', lr=learn_rate)

        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            trainer_config=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
            client_selector=client_selectors.Random(percentage_nb_client),
            trainers_data_dict=client_data,
            initial_model=lambda: initial_model,
            num_rounds=num_rounds
        )

        dataset_used = f'{dataset_name}_{labels_number}c_{min_clients_samples}mn_{max_clients_samples}mx'

        initialize_subscribers(federated)
        add_wandb_log(project, federated, percentage_nb_client, type(initial_model).__name__, batch_size, epochs,
                      num_rounds,
                      learn_rate, dataset_used)

        federated.start()


if __name__ == "__main__":

    if False:
        initialize_exp(dataset_name='mnist',
                       labels_number=10,
                       min_clients_samples=800,
                       max_clients_samples=800,
                       percentage_nb_client=0.2,
                       model=CNN_OriginalFedAvg(),
                       batch_size=32,
                       epochs=1,
                       num_rounds=5,
                       learn_rate=0.01,
                       project='mnist')

    if False:
        initialize_exp(dataset_name='cifar10',
                       labels_number=10,
                       min_clients_samples=200,
                       max_clients_samples=200,
                       percentage_nb_client=0.2,
                       model=CNN_Cifar10(),
                       batch_size=32,
                       epochs=1,
                       num_rounds=5,
                       learn_rate=0.01,
                       project='cifar10',
                       reshape=[(-1, 32, 32, 3), (0, 3, 1, 2)],
                       distributed=False)

    if False:
        initialize_exp(dataset_name='femnist',
                       labels_number=62,
                       min_clients_samples=200,
                       max_clients_samples=200,
                       percentage_nb_client=0.2,
                       model=CNN_OriginalFedAvg(False),
                       batch_size=32,
                       epochs=1,
                       num_rounds=5,
                       learn_rate=0.01,
                       project='femnist')

    if False:
        initialize_exp(dataset_name='umdaa02_fd_filtered_cropped',
                       labels_number=10,
                       min_clients_samples=200,
                       max_clients_samples=200,
                       percentage_nb_client=0.2,
                       model=resnet56(10, 3, 128),
                       batch_size=24,
                       epochs=1,
                       num_rounds=5,
                       learn_rate=0.01,
                       project='umdaa-02-fd-filtered-cropped')

    if False:
        initialize_exp(dataset_name='umdaa02_fd_filtered_cropped',
                       labels_number=10,
                       min_clients_samples=200,
                       max_clients_samples=200,
                       percentage_nb_client=0.2,
                       model=resnet56(10, 3, 128),
                       batch_size=24,
                       epochs=1,
                       num_rounds=5,
                       learn_rate=0.01,
                       project='umdaa-02-fd-filtered-cropped')

    # if True:
    #     initialize_exp(dataset_name='umdaa02touch',
    #                    labels_number=10,
    #                    min_clients_samples=200,
    #                    max_clients_samples=200,
    #                    percentage_nb_client=0.2,
    #                    model=resnet56(10, 1, 4),
    #                    batch_size=24,
    #                    epochs=1,
    #                    num_rounds=5,
    #                    learn_rate=0.01,
    #                    project='umdaa02touch',
    #                    distributed=True)
