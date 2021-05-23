import copy
from abc import abstractmethod, ABC
from functools import reduce
from itertools import chain

from torch import nn

from src import tools
from src.data_container import DataContainer


class Events:
    ET_FED_START = 'federated_learning_start'
    ET_INIT = 'init'
    ET_ROUND_START = 'round_start'
    ET_TRAINER_SELECTED = 'trainers_selected'
    ET_TRAIN_START = 'training_started'
    ET_TRAIN_END = 'training_finished'
    ET_AGGREGATION_END = 'aggregation_finished'
    ET_ROUND_FINISHED = 'round_finished'
    ET_FED_END = 'federated_learning_end'


class FederatedEventPlug(ABC):
    def __init__(self, only: None or [] = None):
        self.only = only

    def on_federated_started(self, params):
        pass

    def on_federated_ended(self, params):
        pass

    def on_init(self, params):
        pass

    def on_training_start(self, params):
        pass

    def on_training_end(self, params):
        pass

    def on_aggregation_end(self, params):
        pass

    def on_round_end(self, params):
        pass

    def on_round_start(self, params):
        pass

    def on_trainers_selected(self, params):
        pass

    def force(self) -> []:
        return []

    def as_events(self):
        return {
            Events.ET_FED_START: self.on_federated_started,
            Events.ET_INIT: self.on_init,
            Events.ET_ROUND_START: self.on_round_start,
            Events.ET_TRAINER_SELECTED: self.on_trainers_selected,
            Events.ET_TRAIN_START: self.on_training_start,
            Events.ET_TRAIN_END: self.on_training_end,
            Events.ET_AGGREGATION_END: self.on_aggregation_end,
            Events.ET_ROUND_FINISHED: self.on_round_end,
            Events.ET_FED_END: self.on_federated_ended,
        }


class AbstractFederated:

    def __init__(self, trainers_data_dict: {int: DataContainer}, create_model: callable,
                 num_rounds=10, desired_accuracy=0.9, trainer_per_round=4, batch_size=8, train_ratio=0.8,
                 epochs=10, lr=0.1, ignore_acc_decrease=False, **kwargs):
        self.ignore_acc_decrease = ignore_acc_decrease
        self.trainer_per_round = trainer_per_round
        self.create_model = create_model
        self.trainers_data_dict = trainers_data_dict
        self.aggregated_model = create_model()
        self.num_rounds = num_rounds
        self.desired_accuracy = desired_accuracy
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.lr = lr
        self.args = kwargs
        self.events = {}

    def start(self):
        self.broadcast(Events.ET_FED_START, **self.configs())
        global_weights = self.init()
        self.broadcast(Events.ET_INIT, global_weights=global_weights)
        num_round = 0
        while True:
            self.broadcast(Events.ET_ROUND_START, round=num_round)
            trainers_ids = self.select(list(self.trainers_data_dict.keys()), num_round)
            self.broadcast(Events.ET_TRAINER_SELECTED, trainers_ids=trainers_ids)
            selected_trainers = tools.dict_select(trainers_ids, self.trainers_data_dict)
            trainers_train_data, trainers_test_data = self.split(selected_trainers)
            self.broadcast(Events.ET_TRAIN_START, trainers_data=trainers_train_data)
            trainers_weights, sample_size = self.train(global_weights, trainers_train_data, num_round)
            self.broadcast(Events.ET_TRAIN_END, trainers_weights=trainers_weights, sample_size=sample_size)
            global_weights = self.aggregate(trainers_weights, sample_size, num_round)
            tools.load(self.aggregated_model, global_weights)
            self.broadcast(Events.ET_AGGREGATION_END, global_weights=global_weights, global_model=self.aggregated_model)
            accuracy, loss, local_acc, local_loss = self.infer(self.aggregated_model, trainers_test_data)
            self.broadcast(Events.ET_ROUND_FINISHED, round=num_round, accuracy=accuracy, loss=loss, local_acc=local_acc,
                           local_loss=local_loss)
            num_round += 1
            if (num_round > 0 and num_round >= self.num_rounds) or accuracy >= self.desired_accuracy:
                self.broadcast(Events.ET_FED_END, aggregated_model=self.aggregated_model)
                break
        return self.aggregated_model

    def configs(self):
        named = {
            "ignore_acc_decrease": self.ignore_acc_decrease,
            "trainer_per_round": self.trainer_per_round,
            "create_model": self.create_model,
            "trainers_data_dict": self.trainers_data_dict,
            "num_rounds": self.num_rounds,
            "desired_accuracy": self.desired_accuracy,
            "batch_size": self.batch_size,
            "train_ratio": self.train_ratio,
            "epochs": self.epochs,
            "lr": self.lr,
        }
        return reduce(lambda x, y: dict(x, **y), (named, self.args))

    def infer(self, model, trainers_data):
        local_accuracy = {}
        local_loss = {}
        for trainer_id, data in trainers_data.items():
            acc, loss = tools.infer(model, data.batch(self.batch_size))
            local_accuracy[trainer_id] = acc
            local_loss[trainer_id] = loss
        total_accuracy = sum(local_accuracy.values()) / len(local_accuracy)
        total_loss = sum(local_loss.values()) / len(local_loss)
        return total_accuracy, total_loss, local_accuracy, local_loss

    def split(self, trainers_data: {int: DataContainer}):
        train_trainers_data = {}
        test_trainers_data = {}
        for trainer_id, data in trainers_data.items():
            train_data, test_data = data.split(self.train_ratio)
            train_trainers_data[trainer_id] = train_data
            test_trainers_data[trainer_id] = test_data
        return train_trainers_data, test_trainers_data

    @abstractmethod
    def init(self) -> nn.Module:
        pass

    @abstractmethod
    def select(self, trainer_ids: [int], round_id: int) -> [int]:
        pass

    @abstractmethod
    def train(self, global_model_weights: nn.ModuleDict, trainers_data: {int: DataContainer}, round_id: int) -> (
            {int: nn.ModuleDict}, {int: int}):
        pass

    @abstractmethod
    def aggregate(self, trainers_models_weight_dict: {int: nn.ModuleDict}, sample_size: {int: int},
                  round_id: int) -> nn.ModuleDict:
        pass

    def get_global_model(self):
        return copy.deepcopy(self.aggregated_model)

    def broadcast(self, event_name, **kwargs):
        if event_name in self.events:
            for item in self.events[event_name]:
                item(kwargs)

    def register_event(self, event_name, action):
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
