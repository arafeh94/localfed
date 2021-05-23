import logging

import src.data.data_generator
from applications.fed_continuous import FedContinuous
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events
import src.federated.plugins as plugs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('generating data --Started')

dg = src.data.data_generator.load('../../datasets/continuous_unbalanced.pkl')
client_data = dg.distributed
dg.describe()

logger.info("----------------------")
logger.info("start normal federated")
logger.info("----------------------")
federated = FedContinuous(
    trainers_data_dict=client_data,
    create_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=5,
    desired_accuracy=0.9,
    epochs=10,
    lr=0.1,
)
federated.plug(plugs.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED], detailed_selection=False))
federated.plug(plugs.FederatedTimer([Events.ET_ROUND_START, Events.ET_TRAIN_END]))
federated.plug(plugs.FedPlot(per_rounds=False))
federated.start()

logger.info('generating data --Ended')
