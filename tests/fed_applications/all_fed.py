import logging

from components.fedavg import FedAVG
from components._need_update_ga import GAFedAVG
from libs.model.linear.lr import LogisticRegression
from src.data.data_generator import DataGenerator
from src.data.data_provider import LocalMnistDataProvider
from src.federated.federated import Events
from src.federated.plugins import FederatedLogger, FederatedTimer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
logger.info('generating data --Started')

dg = DataGenerator(LocalMnistDataProvider(limit=10000))
client_data = dg.distribute_shards(num_clients=50, min_size=30, max_size=100, shards_per_client=2)
dg.describe()
logger.info('generating data --Ended')

test_fed_avg = True
test_ga_fed = True

federated_config = {
    'trainers_data_dict': client_data,
    'create_model': lambda: LogisticRegression(28 * 28, 10),
    'num_rounds': 1,
    'desired_accuracy': 0.9,
    'trainer_per_round': 15
}

if test_ga_fed:
    logger.info("----------------------")
    logger.info("start federated genetic")
    logger.info("----------------------")
    federated = GAFedAVG(**federated_config)
    federated.plug(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED], detailed_selection=True))
    federated.plug(FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
    federated.start()

if test_fed_avg:
    logger.info("----------------------")
    logger.info("start normal federated")
    logger.info("----------------------")
    federated = FedAVG(**federated_config)
    federated.plug(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED], detailed_selection=True))
    federated.plug(FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
    federated.start()
