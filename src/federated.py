import copy
from abc import abstractmethod
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


class FederatedEventPlug:
    def __init__(self, only: None or [] = None):
        self.only = only

    @abstractmethod
    def on_federated_started(self, params):
        pass

    @abstractmethod
    def on_federated_ended(self, params):
        pass

    @abstractmethod
    def on_init(self, params):
        pass

    @abstractmethod
    def on_training_start(self, params):
        pass

    @abstractmethod
    def on_training_end(self, params):
        pass

    @abstractmethod
    def on_aggregation_end(self, params):
        pass

    @abstractmethod
    def on_round_end(self, params):
        pass

    @abstractmethod
    def on_round_start(self, params):
        pass

    @abstractmethod
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

    def __init__(self, trainers_data_dict: {int: DataContainer}, create_model: callable, test_data: DataContainer,
                 num_rounds=10, desired_accuracy=0.9, trainer_per_round=4, batch_size=8):
        self.trainer_per_round = trainer_per_round
        self.create_model = create_model
        self.trainers_data_dict = trainers_data_dict
        self.test_data = test_data
        self.aggregated_model = create_model()
        self.num_rounds = num_rounds
        self.desired_accuracy = desired_accuracy
        self.batch_size = batch_size
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
            self.broadcast(Events.ET_TRAIN_START, trainers_data=selected_trainers)
            trainers_weights, sample_size = self.train(global_weights, selected_trainers, num_round)
            self.broadcast(Events.ET_TRAIN_END, trainers_weights=trainers_weights, sample_size=sample_size)
            global_weights = self.aggregate(trainers_weights, sample_size, num_round)
            self.broadcast(Events.ET_AGGREGATION_END, global_weights=global_weights)
            tools.load(self.aggregated_model, global_weights)
            accuracy, loss = self.infer(self.aggregated_model)
            self.broadcast(Events.ET_ROUND_FINISHED, round=num_round, accuracy=accuracy, loss=loss)
            num_round += 1
            if (num_round > 0 and num_round >= self.num_rounds) or accuracy >= self.desired_accuracy:
                self.broadcast(Events.ET_FED_END, aggregated_model=self.aggregated_model)
                break
        return self.aggregated_model

    def configs(self):
        return {
            "trainer_per_round": self.trainer_per_round,
            "create_model": self.create_model,
            "trainers_data_dict": self.trainers_data_dict,
            "test_data": self.test_data,
            "num_rounds": self.num_rounds,
            "desired_accuracy": self.desired_accuracy,
            "batch_size": self.batch_size,
        }

    def infer(self, model: nn.ModuleDict):
        return tools.infer(model, self.test_data.batch(self.batch_size))

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
