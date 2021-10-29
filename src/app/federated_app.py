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
        self.logger = logging.getLogger('main')
        self.log_level = settings.get('log_level', absent_ok=True) or logging.INFO

    def init_federated(self, session):
        distributor = self.settings.get('distributor', absent_ok=False)
        dataset_name = self.settings.get('dataset', absent_ok=False)
        model = self.settings.get('model', absent_ok=False)
        trainer_params = TrainerParams(
            trainer_class=self.settings.get('trainer_class') or trainers.TorchTrainer,
            batch_size=self.settings.get('batch_size', absent_ok=False),
            epochs=self.settings.get('epochs', absent_ok=False),
            optimizer=self.settings.get('optimizer') or 'sgd',
            criterion=self.settings.get('criterion') or 'cel',
            lr=self.settings.get('lr', absent_ok=False)
        )
        aggregator = self.settings.get('aggregator') or aggregators.AVGAggregator()
        metric = metrics.AccLoss(batch_size=self.settings.get('batch_size', absent_ok=False),
                                 criterion=nn.CrossEntropyLoss(), device=self.settings.get('device') or None)
        selector = client_selectors.Random(self.settings.get('client_ratio', absent_ok=False))
        distributed_data = preload(dataset_name, distributor)
        federated = FederatedLearning(
            trainer_manager=SeqTrainerManager(),
            trainer_config=trainer_params,
            aggregator=aggregator,
            metrics=metric,
            client_selector=selector,
            trainers_data_dict=distributed_data,
            initial_model=lambda: model,
            num_rounds=self.settings.get('rounds'),
            desired_accuracy=0.99,
            accepted_accuracy_margin=self.settings.get('accepted_accuracy_margin') or -1,
        )
        return federated

    def start_with_subscribers(self, subscribers=None):
        if subscribers and not isinstance(subscribers, list):
            subscribers = [subscribers]

        session = Session(self.settings)
        federated = self.init_federated(session)
        configs_subscribers: list = session.settings.get('subscribers', absent_ok=True)
        subscribers = configs_subscribers + subscribers if subscribers else configs_subscribers
        self._attach_subscribers(federated, subscribers, session)

        federated.start()

    def start(self):
        self.start_with_subscribers(None)

    def start_all(self, subscribers=None):
        for index, st in enumerate(self.settings):
            self.logger.info(f'starting config {index}: {str(st.get_config())}')
            self.start_with_subscribers(subscribers)

    def _attach_subscribers(self, federated: FederatedLearning, subscribers, session):
        self.logger.info('attaching subscribers...')
        subscribers = self._check_subscribers(subscribers, session)
        for subs in subscribers:
            self.logger.info(f'attaching: {type(subs)}')
            federated.add_subscriber(subs)

    def _check_subscribers(self, subscribers, session):
        if subscribers is None:
            subscribers = self._default_subscribers(session)
        if len(subscribers) == 0:
            return
        if subscribers[0] == '..':
            subscribers.pop(0)
            subscribers = self._default_subscribers(session) + subscribers
        for subscriber in subscribers:
            if not isinstance(subscriber, Subscriber):
                raise Exception(f'unsupported subscriber of type {type(subscriber)}')
        return subscribers

    def _default_subscribers(self, session):
        return [
            FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]),
            Timer([Timer.FEDERATED, Timer.ROUND]),
            Resumable(io=session.cache),
        ]
