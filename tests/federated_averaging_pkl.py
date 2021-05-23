import logging

import src
from applications.fedavg import FedAVG
from libs.model.linear.lr import LogisticRegression
from src import plugins
from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider, PickleDataProvider
from src.federated import Events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
dg = src.data_generator.load('../datasets/2_50_big_shards.pkl')
client_data = dg.distributed
dg.describe()
logger.info('Generating Data --Ended')

federated = FedAVG(
    trainers_data_dict=client_data,
    create_model=lambda: LogisticRegression(28 * 28, 10),
    trainer_per_round=6,
    num_rounds=20,
    desired_accuracy=0.99,
    epochs=10,
    lr=0.1,
    batch_size=10,
)

federated.plug(plugins.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
federated.plug(plugins.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
federated.plug(plugins.FedPlot())
federated.plug(plugins.CustomModelTestPlug(PickleDataProvider('../datasets/test_data.pkl').collect().as_tensor()))

logger.info("----------------------")
logger.info("start federated")
logger.info("----------------------")
federated.start()
