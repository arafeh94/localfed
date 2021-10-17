import logging
import sys

sys.path.append('../../')
import matplotlib.pyplot as plt
from torch import nn

import libs.model.cv.cnn


from libs.model.cv.cnn import CNN32
from libs.model.cv.resnet import ResNet, resnet56
from src.apis import lambdas
from src.data.data_provider import PickleDataProvider
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider
from src.federated.subscribers import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = data_loader.cifar10_10shards_100c_400min_400max()
logger.info('Generating Data --Ended')


def create_model(name):
    if name == 'resnet':
        return resnet56(10, 3, 32)
    else:
        global client_data
        # cifar10 data reduced to 1 dimension from 32,32,3. cnn32 model requires the image shape to be 3,32,32
        client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
        return libs.model.cv.cnn.Cifar10Model()


initialize_model = create_model('cnn')

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=1, optimizer='sgd',
                               criterion='cel', lr=0.1)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.2),
    trainers_data_dict=client_data,
    initial_model=lambda: initialize_model,
    num_rounds=50,
    desired_accuracy=0.99,
)
# federated.add_subscriber(subscribers.ShowDataDistribution(10, per_round=True, save_dir='./exp_pct2'))
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(subscribers.FedPlot(plot_each_round=True, show_loss=False))
# federated.add_subscriber(subscribers.ShowWeightDivergence(save_dir="./exp_pct2"))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
