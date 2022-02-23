import calendar
import logging
import time

from libs.model.cv.cnn import CNN_OriginalFedAvg_fall
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src.data.data_loader import preload
from src.federated.components import aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer, CPUTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

# client_data = preload('fall_ar_by_client', tag='fall_unique',
#                       transformer=lambda dct: dct.map(
#                           lambda client_id, dc: dc.filter(lambda x, y: y in [0, 1, 3, 4, 5, 8, 15])
#                               .map(lambda x, y: x, client_id)))
client_data = preload('fall_013458_15')
logger.info(client_data)
epochs = 5
batch_size = 150
clients_per_round = len(client_data)
# number of clients are 33
# initial_model = lambda: LogisticRegression(11, len(client_data))
initial_model = lambda: CNN_OriginalFedAvg_fall(len(client_data), 11)
num_rounds = 10

print(client_data)
trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=epochs, batch_size=batch_size,
                               criterion='cel', lr=0.01)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=batch_size, criterion='cel', device='cuda'),
    client_selector=client_selectors.Random(clients_per_round),
    trainers_data_dict=client_data,
    initial_model=initial_model,
    num_rounds=num_rounds,
)
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(SQLiteLogger(str(calendar.timegm(time.gmtime())), 'res.db', 'fall_test'))
logger.info("----------------------")
logger.info(f"start federated")
logger.info("----------------------")
federated.start()
