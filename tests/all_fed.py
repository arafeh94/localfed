import logging

from applications.fedavg import FedAVG
from applications.gafed import GAFedAVG
from libs.model.linear.lr import LogisticRegression
from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider
from src.plugins import FederatedLogger, FederatedTimer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')
logger.debug('generating data --Started')

test_data = LocalMnistDataProvider("select data,label from skewed where user_id=100").collect().as_tensor()
dg = DataGenerator(LocalMnistDataProvider(limit=10000))
client_data = dg.distribute_shards(num_clients=50, min_size=30, max_size=100, shards_per_client=2)
dg.describe()
logger.debug('generating data --Ended')

test_fed_avg = True
test_ga_fed = True

federated_config = {
    'trainers_data_dict': client_data,
    'create_model': lambda: LogisticRegression(28 * 28, 10),
    'test_data': test_data,
    'num_rounds': 1,
    'desired_accuracy': 0.9,
    'trainer_per_round': 15
}

if test_ga_fed:
    logger.debug("----------------------")
    logger.debug("start federated genetic")
    logger.debug("----------------------")
    federated = GAFedAVG(**federated_config)
    federated.plug(FederatedLogger([FedAVG.ET_ROUND_FINISHED, FedAVG.ET_TRAINER_SELECTED], detailed_selection=True))
    federated.plug(FederatedTimer([FedAVG.ET_ROUND_START, FedAVG.ET_TRAIN_END]))
    federated.start()

if test_fed_avg:
    logger.debug("----------------------")
    logger.debug("start normal federated")
    logger.debug("----------------------")
    federated = FedAVG(**federated_config)
    federated.plug(FederatedLogger([FedAVG.ET_ROUND_FINISHED, FedAVG.ET_TRAINER_SELECTED], detailed_selection=True))
    federated.plug(FederatedTimer([FedAVG.ET_ROUND_START, FedAVG.ET_TRAIN_END]))
    federated.start()
