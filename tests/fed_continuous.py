import logging

from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider, PickleDataProvider

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')

logger.debug('generating data --Started')

test_data = PickleDataProvider('test.pkl').collect().as_tensor()

dg = DataGenerator(LocalMnistDataProvider(limit=10000))
client_data = dg.distribute_continuous(num_clients=10, min_size=30, max_size=100)
dg.describe()

logger.debug('generating data --Ended')
