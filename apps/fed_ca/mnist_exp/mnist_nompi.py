import logging

from torch import nn
from apps.fed_ca.utilities.load_dataset import LoadData
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.federated import subscribers
from src.federated.components.trainer_manager import SeqTrainerManager
from apps.fed_ca.utilities.hp_generator import generate_configs, build_random, calculate_max_rounds
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

ld = LoadData(dataset_name='mnist', shards_nb=0, clients_nb=10, min_samples=2400, max_samples=3000)
client_data = ld.pickle_distribute_continuous()

# ld = LoadData(dataset_name='mnist', shards_nb=2, clients_nb=100, min_samples=300, max_samples=300)
# client_data = ld.pickle_distribute_shards()

# ld = LoadData(dataset_name='mnist', shards_nb=10, clients_nb=100, min_samples=600, max_samples=600)
# client_data = ld.pickle_distribute_shards()

dataset_used = ld.filename
tools.detail(client_data)

# building Hyperparameters
input_shape = 28 * 28
labels_number = 10
percentage_nb_client = 0.2

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    'CNN_OriginalFedAvg': CNN_OriginalFedAvg()
    # 'CNN': CNN_DropOut(False)
}
for model_name, gen_model in initial_models.items():

    # hyper_params = {'batch_size': [10, 50, 1000], 'epochs': [1, 5, 20], 'num_rounds': [1200]}
    hyper_params = {'batch_size': [10], 'epochs': [1], 'num_rounds': [800]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.1

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}, '
            f'initial_model={initial_model} ')

        trainer_manager = SeqTrainerManager()
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
            desired_accuracy=1
        )

        federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

        federated.add_subscriber(subscribers.WandbLogger(config={
            'lr': learn_rate, 'batch_size': batch_size,
            'epochs': epochs,
            'num_rounds': num_rounds, 'data_file': dataset_used,
            'model': model_name,
            'selected_clients': percentage_nb_client
        }))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
