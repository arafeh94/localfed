import calendar
import logging
import time

from libs.model.cv.cnn import Cnn1D
from src.apis import lambdas
from src.apis.extensions import Dict
from src.data.data_distributor import LabelDistributor, PipeDistributor
from src.data.data_loader import preload
from src.federated.components import aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer, CPUTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

pipes = [
    PipeDistributor.pick_by_label_id([1], 100, 1),
    PipeDistributor.pick_by_label_id([2], 300, 1)
]


# distributor = PipeDistributor(pipes)
#
# client_data = preload('fall_ar_by_client').map(lambda cid, dt: distributor.distribute(dt).reduce(lambdas.dict2dc))

def transformer(dt: Dict):
    dt = dt.map(lambda cid, dc: dc.map(lambda x, y: (x, cid)))
    dt = dt.map(lambdas.as_tensor)
    return dt


print('loading dataset...')
client_data: Dict = preload('fall_ar_by_client', tag='fall_co_1', transformer=transformer)
client_data = client_data.select(range(1, 16))
new_client_data = Dict()
for key, item in client_data.items():
    new_client_data[key - 1] = item
logger.info(client_data)

config = {
    'epochs': 10,
    'batch_size': 2048,
    'clients_per_round': 15,
    'initial_model': lambda: Cnn1D(15),
    'num_rounds': 50,
}

trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                               batch_size=config['batch_size'],
                               criterion='cel', lr=0.01)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion='cel'),
    client_selector=client_selectors.Random(0.9999),
    trainers_data_dict=new_client_data,
    initial_model=config['initial_model'],
    num_rounds=config['num_rounds'],
)
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(SQLiteLogger(str(calendar.timegm(time.gmtime())), 'res.db', config))
federated.add_subscriber(TqdmLogger())
logger.info("----------------------")
logger.info(f"start federated")
logger.info("----------------------")
federated.start()
