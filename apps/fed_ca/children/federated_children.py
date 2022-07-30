import logging
from torch import nn
from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.collection import MLP
from src import tools
from src.apis import lambdas
from src.data.data_distributor import UniqueDistributor, LabelDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger

import visualizations
from src.federated.subscribers.wandb_logger import WandbLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
# total number of clients is 118
dataset_used = 'children_touch'
client_data = preload(dataset_used)
visualizations.visualize(client_data)

client_data = client_data.reduce(lambdas.dict2dc).as_tensor()
labels_number = 10

ud = LabelDistributor(labels_number, 1, 1000, 1000)
print(ud.id())
client_data = ud.distribute(client_data)

visualizations.visualize(client_data)
tools.detail(client_data)

# building Hyperparameters
percentage_nb_client = 1

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(3, labels_number),
    'MLP': MLP(3, 1000, labels_number)
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg()
    # 'ConvNet1D_test': ConvNet1D_test(labels_number)
    # 'CNN': CNN_DropOut(False)
}

for model_name, gen_model in initial_models.items():

    hyper_params = {'batch_size': [256], 'epochs': [1], 'num_rounds': [2],
                    'learn_rate': [0.001]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = config['learn_rate']

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
        )
        # filename is used for both the wandb id and for model resume at checkpoint
        filename_id = f'lr={learn_rate}_batch_size={batch_size}_epochs={epochs}_num_rounds={num_rounds}_data_file={dataset_used}_model_name={model_name}'

        # use flush=True if you don't want to continue from the last round
        # federated.add_subscriber(subscribers.Resumable(federated, tag='002', save_each=5))

        federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

        federated.add_subscriber(WandbLogger(project='children', config={
            'lr': learn_rate, 'batch_size': batch_size,
            'epochs': epochs,
            'num_rounds': num_rounds, 'data_file': dataset_used + '_federated'+'_'+ud.id(),
            'model': model_name,
            'selected_clients': percentage_nb_client,
            'labels_number': labels_number
        }))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
