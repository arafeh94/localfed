import copy
import math
from collections import defaultdict
from functools import reduce
from typing import Dict

from src import tools
from src.data.data_container import DataContainer
from src.federated.events import Events, FederatedEventPlug
from src.federated.protocols import Aggregator, ClientSelector, ModelInfer, Trainer
from src.federated.trainer_manager import TrainerManager


class FederatedLearning:
    TEST_ON_ALL = 0
    TEST_ON_SELECTED = 1

    def __init__(self, trainer_manager: TrainerManager, trainer_params, aggregator: Aggregator,
                 client_selector: ClientSelector, metrics: ModelInfer, trainers_data_dict: Dict[int, DataContainer],
                 initial_model: callable, num_rounds=10, desired_accuracy=0.9, train_ratio=0.8,
                 ignore_acc_decrease=False, test_on=TEST_ON_ALL, optimizer: str = None, **kwargs):
        self.optimizer = optimizer
        self.trainer_params = trainer_params
        self.trainer_manager = trainer_manager
        self.aggregator = aggregator
        self.client_selector = client_selector
        self.metrics = metrics
        self.ignore_acc_decrease = ignore_acc_decrease
        self.trainers_data_dict = trainers_data_dict
        self.desired_accuracy = desired_accuracy
        self.initial_model = initial_model
        self.train_ratio = train_ratio
        self.num_rounds = num_rounds
        self.args = kwargs
        self.events = {}
        self._check_params()
        self.context = FederatedLearning.Context()
        self.trainers_test = {}
        self.trainers_train = {}
        self.test_on = test_on
        for trainer_id, data in trainers_data_dict.items():
            data = data.shuffle().as_tensor()
            train, test = data.split(train_ratio)
            self.trainers_train[trainer_id] = train
            self.trainers_test[trainer_id] = test

    def start(self):
        self.broadcast(Events.ET_FED_START, **self.configs())
        self.context.build(self)
        self.broadcast(Events.ET_INIT, global_model=self.context.model)
        while True:
            self.broadcast(Events.ET_ROUND_START, round=self.context.round_id)

            trainers_ids = self.client_selector.select(list(self.trainers_data_dict.keys()), self.context.round_id)
            self.broadcast(Events.ET_TRAINER_SELECTED, trainers_ids=trainers_ids)

            trainers_train_data = tools.dict_select(trainers_ids, self.trainers_train)
            self.broadcast(Events.ET_TRAIN_START, trainers_data=trainers_train_data)
            trainers_weights, sample_size_dict = self.train(trainers_train_data)
            self.broadcast(Events.ET_TRAIN_END, trainers_weights=trainers_weights, sample_size=sample_size_dict)

            global_weights = self.aggregator.aggregate(trainers_weights, sample_size_dict, self.context.round_id)
            tools.load(self.context.model, global_weights)
            self.broadcast(Events.ET_AGGREGATION_END, global_weights=global_weights, global_model=self.context.model)

            test_data = self.trainers_test if self.test_on == FederatedLearning.TEST_ON_ALL else \
                tools.dict_select(trainers_ids, self.trainers_test)
            accuracy, loss, local_acc, local_loss = self.infer(self.context.model, test_data)
            self.broadcast(Events.ET_ROUND_FINISHED, round=self.context.round_id, accuracy=accuracy, loss=loss,
                           local_acc=local_acc, local_loss=local_loss)

            self.context.store(acc=accuracy, loss=loss, lacc=local_acc, lloss=local_loss)
            self.context.new_round()
            if self.context.stop(accuracy):
                self.broadcast(Events.ET_FED_END, aggregated_model=self.context.model)
                break
        return self.context.model

    def train(self, trainers_train_data: Dict[int, DataContainer]):
        for trainer_id, train_data in trainers_train_data.items():
            self.broadcast(Events.ET_TRAINER_STARTED, trainer_id=trainer_id, train_data=train_data)
            model_copy = copy.deepcopy(self.context.model)
            self.trainer_manager.train_req(trainer_id, model_copy, train_data, self.context, self.trainer_params)
        return self.trainer_manager.resolve()

    def infer(self, model, trainers_data: Dict[int, DataContainer]):
        local_accuracy = {}
        local_loss = {}
        sample_size = {}

        for trainer_id, test_data in trainers_data.items():
            acc, loss = self.metrics.infer(model, test_data)
            self.client_selector.on_client_selected(trainer_id, accuracy=acc, loss=loss)
            local_accuracy[trainer_id] = acc
            local_loss[trainer_id] = loss
            sample_size[trainer_id] = len(test_data)

        weighted_accuracy = [local_accuracy[tid] * sample_size[tid] for tid in local_accuracy]
        weighted_loss = [local_loss[tid] * sample_size[tid] for tid in local_loss]

        total_accuracy = sum(weighted_accuracy) / sum(sample_size.values())
        total_loss = sum(weighted_loss) / sum(sample_size.values())

        return total_accuracy, total_loss, local_accuracy, local_loss

    def compare(self, other, verbose=1):
        local_history = self.context.history
        other_history = other.context.history
        performance_history = defaultdict(lambda: [])
        diff = {}
        for round_id, first_data in local_history.items():
            if round_id not in other_history:
                continue
            second_data = other_history[round_id]
            for item in first_data:
                if type(first_data[item]) in [int, float, str]:
                    performance_history[item].append(first_data[item] - second_data[item])
        for item, val in performance_history.items():
            diff[item] = math.fsum(val) / len(val)
        if verbose == 1:
            return diff
        else:
            return diff, performance_history

    def _check_params(self):
        pass

    def configs(self):
        named = {
            "trainer": self.trainer_manager,
            "aggregator": self.aggregator,
            "client_selector": self.client_selector,
            "ignore_acc_decrease": self.ignore_acc_decrease,
            "trainers_data_dict": self.trainers_data_dict,
            "desired_accuracy": self.desired_accuracy,
            "create_model": self.initial_model,
            "train_ratio": self.train_ratio,
            "num_rounds": self.num_rounds
        }
        return reduce(lambda x, y: dict(x, **y), (named, self.args))

    def broadcast(self, event_name: str, **kwargs):
        args = reduce(lambda x, y: dict(x, **y), ({'context': self.context}, kwargs))
        if event_name in self.events:
            for item in self.events[event_name]:
                item(args)

    def register_event(self, event_name: str, action: callable):
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(action)

    def plug(self, plugin: FederatedEventPlug):
        events = plugin.as_events()
        for event_name, call in events.items():
            if plugin.only is not None and event_name not in plugin.only:
                if event_name not in plugin.force():
                    continue
            self.register_event(event_name, call)

    class Context:
        def __init__(self):
            self.round_id = 0
            self.model = None
            self.num_rounds = None
            self.desired_accuracy = None
            self.history = {}

        def new_round(self):
            self.round_id += 1

        def stop(self, acc: float):
            return (0 < self.num_rounds <= self.round_id) or acc >= self.desired_accuracy

        def build(self, federated):
            self.reset()
            self.model = federated.initial_model()
            self.num_rounds = federated.num_rounds
            self.desired_accuracy = federated.desired_accuracy

        def reset(self):
            self.round_id = 0
            self.history.clear()

        def store(self, **kwargs):
            if self.round_id not in self.history:
                self.history[self.round_id] = {}
            self.history[self.round_id] = tools.Dict.concat(self.history[self.round_id], kwargs)
