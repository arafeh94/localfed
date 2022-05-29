import logging

from torch import nn

from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.cnn import CNN_OriginalFedAvg, Cnn1D, ConvNet1D_test
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data.data_distributor import UniqueDistributor, LabelDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.wandb_logger import WandbLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

# client_data = PickleDataProvider(file_path).collect()
dataset_used = 'children_touch'
client_data = preload(dataset_used)
#
# dist = LabelDistributor(119, 3, 60, 60)
# client_data = preload(dataset_used, dist)


tools.detail(client_data)

# building Hyperparameters
labels_number = 64
percentage_nb_client = 0.2

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg()
    'ConvNet1D_test': ConvNet1D_test(labels_number)

    # 'CNN': CNN_DropOut(False)
}

for model_name, gen_model in initial_models.items():

    # hyper_params = {'batch_size': [10, 50, 1000], 'epochs': [1, 5, 20], 'num_rounds': [1200]}
    hyper_params = {'batch_size': [64], 'epochs': [1], 'num_rounds': [800]}

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
            client_selector=client_selectors.Random(percentage_nb_client),
            trainers_data_dict=client_data,
            initial_model=lambda: initial_model,
            num_rounds=num_rounds,
            desired_accuracy=1
            # accepted_accuracy_margin=0.05
        )
        # filename is used for both the wandb id and for model resume at checkpoint
        filename_id = f'lr={learn_rate}_batch_size={batch_size}_epochs={epochs}_num_rounds={num_rounds}_data_file={dataset_used}_model_name={model_name}'

        # use flush=True if you don't want to continue from the last round
        # federated.add_subscriber(subscribers.Resumable(federated, tag='002', save_each=5))

        federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

        # federated.add_subscriber(WandbLogger(project='children', config={
        #     'lr': learn_rate, 'batch_size': batch_size,
        #     'epochs': epochs,
        #     'num_rounds': num_rounds, 'data_file': dataset_used,
        #     'model': model_name,
        #     'selected_clients': percentage_nb_client
        # }))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
