from functools import reduce

from src.federated.protocols import Trainer


class TrainerManager:
    def __init__(self, trainer_class: type, batch_size: int, epochs: int, criterion, optimizer: callable, **args):
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.trainer_class = trainer_class
        self.trainer_args = args
        self.trainers = {}

    def create(self, trainer_id, **kwargs) -> Trainer:
        all_args = reduce(lambda x, y: dict(x, **y), (self.trainer_args, kwargs))
        trainer = self.trainer_class(**all_args)
        self._trainer_init(trainer)
        self.trainers[trainer_id] = trainer
        return trainer

    def _trainer_init(self, trainer: Trainer):
        getattr(trainer, '_Trainer__init')(optimizer=self.optimizer, criterion=self.criterion, epochs=self.epochs,
                                           batch_size=self.batch_size)

    def trainer(self, trainer_id, create_if_not_exists=True):
        if trainer_id not in self.trainers.keys():
            if create_if_not_exists:
                self.trainers[trainer_id] = self.create(trainer_id)
            else:
                raise Exception("requested trainer does not exists")
        return self.trainers[trainer_id]
