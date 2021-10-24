import logging

from torch import nn

from src.apis.broadcaster import Subscriber
from src.app.session import Session
from src.app.settings import Settings
from src.data.data_loader import preload
from src.federated.components import trainers, aggregators, client_selectors, metrics
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.timer import Timer

logging.basicConfig(level=logging.INFO)


class FederatedApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = Session(settings)
        self.distributor = settings.get('distributor', absent_ok=False)
        self.dataset_name = settings.get('dataset', absent_ok=False)
        self.model = settings.get('model', absent_ok=False)
        self.trainer_params = TrainerParams(
            trainer_class=settings.get('trainer_class') or trainers.TorchTrainer,
            batch_size=settings.get('batch_size', absent_ok=False),
            epochs=settings.get('epochs', absent_ok=False),
            optimizer=settings.get('optimizer') or 'sgd',
            criterion=settings.get('criterion') or 'cel',
            lr=settings.get('lr', absent_ok=False)
        )
        self.aggregator = aggregators.AVGAggregator()
        self.metric = metrics.AccLoss(batch_size=self.settings.get('batch_size', absent_ok=False),
                                      criterion=nn.CrossEntropyLoss(), device=self.settings.get('device') or None)
        self.selector = client_selectors.Random(self.settings.get('client_ratio', absent_ok=False))
        self.logger = logging.getLogger('main')
        self.subscribers: list = self.settings.get('subscribers', absent_ok=True)
        self._check_subscribers()

    def _attach_subscribers(self, federated: FederatedLearning):
        self.logger.info('attaching subscribers...')
        for subs in self.subscribers:
            self.logger.info(f'attaching: {type(subs)}')
            federated.add_subscriber(subs)

    def default_subscribers(self):
        return [
            FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]),
            Timer([Timer.FEDERATED, Timer.ROUND]),
            Resumable(id=self.session.session_id(), save_ratio=5),
        ]

    def start_with_subscribers(self, subscribers=None):
        distributed_data = preload(self.dataset_name, self.distributor)
        federated = FederatedLearning(
            trainer_manager=SeqTrainerManager(),
            trainer_config=self.trainer_params,
            aggregator=self.aggregator,
            metrics=self.metric,
            client_selector=self.selector,
            trainers_data_dict=distributed_data,
            initial_model=lambda: self.model,
            num_rounds=self.settings.get('rounds'),
            desired_accuracy=0.99,
            accepted_accuracy_margin=self.settings.get('accepted_accuracy_margin') or -1,
        )
        if subscribers is not None:
            self.subscribers.extend(subscribers)
        self._attach_subscribers(federated)
        federated.start()

    def start(self):
        self.start_with_subscribers(None)

    def _check_subscribers(self):
        if self.subscribers is None:
            self.subscribers = self.default_subscribers()
        if len(self.subscribers) == 0:
            return
        if self.subscribers[0] == '..':
            self.subscribers.pop(0)
            self.subscribers = self.default_subscribers() + self.subscribers
        for subscriber in self.subscribers:
            if not isinstance(subscriber, Subscriber):
                raise Exception(f'unsupported subscriber of type {type(subscriber)}')
