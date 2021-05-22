import logging

from applications.fed_continuous import FedContinuous
from libs.model.linear.lr import LogisticRegression
from src.data_generator import DataGenerator
from src.data_provider import LocalMnistDataProvider, PickleDataProvider
from src.federated import Events
import src.plugins as plugs

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')

logger.debug('generating data --Started')

test_data = PickleDataProvider('test.pkl').collect().as_tensor()

dg = DataGenerator(LocalMnistDataProvider(limit=10000))
client_data = dg.distribute_continuous(num_clients=10, min_size=30, max_size=100)
dg.describe()

logger.debug("----------------------")
logger.debug("start normal federated")
logger.debug("----------------------")
federated = FedContinuous(
    trainers_data_dict=client_data,
    create_model=lambda: LogisticRegression(28 * 28, 10),
    test_data=test_data,
    num_rounds=5,
    desired_accuracy=0.9
)
federated.plug(plugs.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED], detailed_selection=True))
federated.plug(plugs.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
federated.plug(plugs.FedPlot())
federated.start()

logger.debug('generating data --Ended')
