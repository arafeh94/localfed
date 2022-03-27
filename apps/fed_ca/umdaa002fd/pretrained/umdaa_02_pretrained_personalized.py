import logging
import pickle

import torchvision
from torch import nn

from apps.fed_ca.umdaa002fd.pretrained.inception_resnet_v1 import InceptionResnetV1
from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import resnet56, ResNet
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data.data_container import DataContainer
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
dataset = 'umdaa02_fd_filtered_cropped'
# total number of clients from umdaa02-fd is 44
labels_number = 3
ud = UniqueDistributor(labels_number, 500, 500)
client_data = PickleDataProvider("../../../../datasets/pickles/umdaa02_fd_filtered_cropped.pkl").collect()
# tools.detail(client_data)
client_data = ud.distribute(client_data)
dataset_used = dataset + '_' + ud.id()

tools.detail(client_data)

# client_data = client_data.map(lambdas.reshape((-1, 3, 128, 128)))
client_data = client_data.map(lambdas.reshape((-1, 128, 128, 3))).map(lambdas.transpose((0, 3, 1, 2)))

# building Hyperparameters
input_shape = 128 * 128
percentage_nb_client = labels_number

# vggface2 = InceptionResnetV1(pretrained='vggface2', num_classes=labels_number, classify=True, device='cuda')
vggface2 = pickle.load(open('vggface2.pkl', 'rb'))

#  vggface2 as fixed feature extractor: Here, we will freeze the weights for all of the network except that of
#  the final fully connected layers. These last fully connected layers is replaced with a new ones with random weights
#  and only these layers are trained.
for param in list(vggface2.children()):
    param.requires_grad = False
for param in list(vggface2.children())[-5:]:
    param.requires_grad = True

# number of models that we are using
initial_models = {

    'vggface2': vggface2,

}

for model_name, gen_model in initial_models.items():

    # hyper_params = {'batch_size': [10, 50, 1000], 'epochs': [1, 5, 20], 'num_rounds': [1200]}
    hyper_params = {'batch_size': [128], 'epochs': [200]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        initial_model = config['initial_model']
        learn_rate = 0.001

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs} '
            f'initial_model={initial_model} ')

        test = DataContainer([], [])
        for i in range(labels_number):
            tdc = client_data[i].split(0.8)[1].as_list()
            test.x.extend(tdc.x)
            test.y.extend(tdc.y)

        test = test.as_tensor()
        # create a personalized model based on the global modal pretrained
        for i in range(labels_number):
            train, _ = client_data[i].split(0.8)
            t_model = TorchModel(gen_model)
            t_model.train(train.batch(batch_size), epochs=epochs, lr=learn_rate)
            acc, loss = t_model.infer(test.batch(batch_size))
            # tools.train(gen_model, train_data=train.batch(batch_size), epochs=epochs, lr=learn_rate)
            # acc, loss = tools.infer(gen_model, test.batch(batch_size))
            print(f'Printing Accuracy and Loss of client {i}')
            print(acc, loss)

        end_time = datetime.now()
        print('Total Duration: {}'.format(end_time - start_time))

end_time = datetime.now()
print('Total Duration: {}'.format(end_time - start_time))
