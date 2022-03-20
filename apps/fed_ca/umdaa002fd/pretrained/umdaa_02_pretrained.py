import logging

import torchvision
from torch import nn

from apps.fed_ca.umdaa002fd.pretrained.inception_resnet_v1 import InceptionResnetV1
from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import resnet56, ResNet
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload
from src.data.data_provider import PickleDataProvider
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.wandb_logger import WandbLogger
from datetime import datetime

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
dataset = 'umdaa002fd'
# total number of clients from umdaa02-fd is 44
labels_number = 10
ud = UniqueDistributor(labels_number, 100, 100)
client_data = PickleDataProvider("../../../../datasets/pickles/umdaa02_fd.pkl").collect()
# tools.detail(client_data)
client_data = ud.distribute(client_data)
dataset_used = dataset + '_' + ud.id()

tools.detail(client_data)

# client_data = client_data.map(lambdas.reshape((-1, 3, 128, 128)))
client_data = client_data.map(lambdas.reshape((-1, 128, 128, 3))).map(lambdas.transpose((0, 3, 1, 2)))

# building Hyperparameters
input_shape = 128 * 128
percentage_nb_client = labels_number

vggface2 = InceptionResnetV1(pretrained='vggface2', num_classes=labels_number, classify=True, device='cuda').eval()

for param in list(vggface2.children()):
    param.requires_grad = False
for param in list(vggface2.children())[-5:]:
    param.requires_grad = True

# number of models that we are using
initial_models = {
    # 'resnet56': resnet56(labels_number, 3, 128),
    'vggface2': vggface2,
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg()
    # 'CNN': CNN_DropOut(False)
}
for model_name, gen_model in initial_models.items():

    # hyper_params = {'batch_size': [10, 50, 1000], 'epochs': [1, 5, 20], 'num_rounds': [1200]}
    hyper_params = {'batch_size': [10], 'epochs': [1], 'num_rounds': [100]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.001

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
        # filename_id = f'lr={learn_rate}_batch_size={batch_size}_epochs={epochs}_num_rounds={num_rounds}_data_file={dataset_used}_model_name={model_name}'

        # use flush=True if you don't want to continue from the last round
        # federated.add_subscriber(subscribers.Resumable(federated, tag='002', save_each=5))

        federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

        federated.add_subscriber(WandbLogger(config={
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

        end_time = datetime.now()
        print('Total Duration: {}'.format(end_time - start_time))

end_time = datetime.now()
print('Total Duration: {}'.format(end_time - start_time))
