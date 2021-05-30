from abc import ABC


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
    ET_TRAINER_STARTED = 'trainer_started'
    ET_TRAINER_FINISHED = 'trainer_ended'


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

    def on_trainer_start(self, params):
        pass

    def on_trainer_end(self, params):
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
            Events.ET_TRAINER_STARTED: self.on_trainer_start,
            Events.ET_TRAINER_FINISHED: self.on_trainer_end,
        }
