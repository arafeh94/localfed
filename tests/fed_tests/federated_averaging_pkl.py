import logging

import src
from applications.fedavg import FedAVG
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import plugins
from src.data.data_generator import DataGenerator
from src.federated.federated import Events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../../datasets/pickles/2_50_medium_shards.pkl'
test_file = '../../datasets/pickles/test_data.pkl'

logger.info('Generating Data --Started')
dg = src.data.data_generator.load(data_file)
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
federated.plug(plugins.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor()))

logger.info("----------------------")
logger.info("start federated")
logger.info("----------------------")
federated.start()
