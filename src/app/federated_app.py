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
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.timer import Timer

logging.basicConfig(level=logging.INFO)


class FederatedApp:
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.logger = logging.getLogger('main')
        self.log_level = settings.get('log_level', absent_ok=True) or logging.INFO
        self.load_default_subscribers = kwargs.get('load_default_subscribers', True)
        self.kwargs = kwargs

    def init_federated(self, session):
        distributor = self.settings.get('data.distributor', absent_ok=False)
        dataset_name = self.settings.get('data.dataset', absent_ok=False)
        transformer = self.settings.get('data.transformer', absent_ok=True)
        model = self.settings.get('model', absent_ok=False)
        desired_accuracy = self.settings.get('desired_accuracy', absent_ok=True) or 0.99
        trainer_params = self.settings.get("trainer_config")
        aggregator = self.settings.get('aggregator', absent_ok=True) or aggregators.AVGAggregator()
        metric = metrics.AccLoss(batch_size=self.settings.get('trainer_config.batch_size', absent_ok=False),
                                 criterion=self.settings.get('trainer_config.criterion'),
                                 device=self.settings.get('device') or None)
        selector = self.settings.get('client_selector') or client_selectors.Random(
            self.settings.get('client_ratio', absent_ok=False))
        distributed_data = preload(dataset_name, distributor, transformer=transformer)
        federated = FederatedLearning(
            trainer_manager=self._trainer_manager(),
            trainer_config=trainer_params,
            aggregator=aggregator,
            metrics=metric,
            client_selector=selector,
            trainers_data_dict=distributed_data,
            initial_model=lambda: model,
            num_rounds=self.settings.get('rounds'),
            desired_accuracy=desired_accuracy,
            accepted_accuracy_margin=self.settings.get('accepted_accuracy_margin') or -1,
        )
        return federated

    def _trainer_manager(self):
        return SeqTrainerManager()

    def _start(self, subscribers=None):
        if subscribers and not isinstance(subscribers, list):
            subscribers = [subscribers]
        session = Session(self.settings)
        federated = self.init_federated(session)
        configs_subscribers: list = session.settings.get('subscribers', absent_ok=True) or []
        subscribers = configs_subscribers + subscribers if subscribers else configs_subscribers
        self._attach_subscribers(federated, subscribers, session)
        federated.start()

    def start(self, *subscribers):
        for index, st in enumerate(self.settings):
            self.logger.info(f'starting config {index}: {str(st.get_config())}')
            self._start([s for s in subscribers] if subscribers else None)

    def _attach_subscribers(self, federated: FederatedLearning, subscribers, session):
        self.logger.info('attaching subscribers...')
        subscribers = self._check_subscribers(subscribers, session)
        for subs in subscribers:
            self.logger.info(f'attaching: {type(subs)}')
            federated.add_subscriber(subs)

    def _check_subscribers(self, subscribers, session):
        attaching_subscribers = self._default_subscribers(session) if self.load_default_subscribers else []
        attaching_subscribers = (subscribers or []) + attaching_subscribers
        for subscriber in attaching_subscribers:
            if not isinstance(subscriber, Subscriber):
                raise Exception(f'unsupported subscriber of type {type(subscriber)}')
        return attaching_subscribers

    def _default_subscribers(self, session):
        return [
            FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]),
            Timer([Timer.FEDERATED, Timer.ROUND]),
            Resumable(io=session.cache),
            SQLiteLogger(session.session_id(), db_path='./cache/perf.db', tag=str(session.settings.get_config()))
        ]
