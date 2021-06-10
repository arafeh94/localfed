import logging

from torch import nn

from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.nlp.rnn import RNN_OriginalFedAvg
from src.data.data_provider import LocalShakespeareDataProvider
from src.federated import subscribers
from src.data.data_generator import DataGenerator
from src.federated.federated import Events as et
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
dg = DataGenerator(LocalShakespeareDataProvider(limit=1000), xtt=lambda x: x.long())
client_data = dg.distribute_size(10, 100, 100)
dg.describe()
logger.info('Generating Data --Ended')

trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=trainers.CPUTrainer, batch_size=50, epochs=20, optimizer='sgd',
                               criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=8, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(10),
    trainers_data_dict=client_data,
    initial_model=lambda: RNN_OriginalFedAvg(),
    num_rounds=3,
    desired_accuracy=0.99
)

federated.add_subscriber(subscribers.FederatedLogger([
    et.ET_ROUND_FINISHED, et.ET_TRAINER_SELECTED, et.ET_TRAINER_STARTED, et.ET_TRAINER_FINISHED]))
federated.add_subscriber(subscribers.FederatedTimer([et.ET_TRAINER_FINISHED, et.ET_TRAIN_END]))
federated.add_subscriber(subscribers.FedPlot())
# federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(), 8))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
