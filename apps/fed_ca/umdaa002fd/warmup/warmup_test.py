import logging
import sys

from libs.model.cv.resnet import resnet56

sys.path.append('../../')

from src.federated.subscribers.analysis import ShowWeightDivergence
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from torch import nn
from src.apis import lambdas, files
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
tag = 'federated_test_basic_2'
labels_number = 10
batch_size = 24
epochs_warmup = 1
epochs_federated = 1
lear_rate = 0.001
rounds_nb = 2
is_warmup = True

logger.info('Generating Data --Started')
client_data = data_loader.umdaa02fd_1shard_10c_500min_500max()
logger.info('Generating Data --Ended')

if is_warmup:
    warmup_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[0]).reduce(lambdas.dict2dc).as_tensor()
    client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[1]).map(lambdas.as_tensor)
    initial_model = TorchModel(resnet56(labels_number, 3, 128))
    initial_model.train(warmup_data.batch(batch_size), epochs=epochs_warmup)
    initial_model = initial_model.extract()
else:
    initial_model = resnet56(labels_number, 3, 128)

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size, epochs=epochs_federated, optimizer='sgd',
                               criterion='cel', lr=lear_rate)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(labels_number),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=rounds_nb,
    desired_accuracy=0.99,
)
# federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
# federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
# federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear'))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()

# files.accuracies.save_accuracy(federated, tag)
