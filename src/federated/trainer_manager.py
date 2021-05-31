import logging
from abc import ABC, abstractmethod
from functools import reduce

from torch import nn

from src.apis.mpi import Comm
from src.data.data_container import DataContainer
from src.federated.protocols import Trainer


class TrainerManager:
    @abstractmethod
    def train_req(self, trainer_id, model, train_data, context):
        pass

    @abstractmethod
    def resolve(self):
        pass


class SeqTrainerManager(TrainerManager):
    def __init__(self, trainer_class: type, batch_size: int, epochs: int, criterion, optimizer: callable, **args):
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.trainer_class = trainer_class
        self.trainer_args = args
        self.trainers = {}
        self.train_requests = {}

    def create(self, trainer_id, **kwargs) -> Trainer:
        all_args = reduce(lambda x, y: dict(x, **y), (self.trainer_args, kwargs))
        trainer = self.trainer_class(optimizer=self.optimizer, criterion=self.criterion, epochs=self.epochs,
                                     batch_size=self.batch_size)
        self.trainers[trainer_id] = trainer
        return trainer

    def trainer(self, trainer_id, create_if_not_exists=True):
        if trainer_id not in self.trainers.keys():
            if create_if_not_exists:
                self.trainers[trainer_id] = self.create(trainer_id)
            else:
                raise Exception("requested trainer does not exists")
        return self.trainers[trainer_id]

    def train_req(self, trainer_id, model, train_data, context):
        request = [self.trainer(trainer_id).train, model, train_data, context]
        self.train_requests[trainer_id] = request
        return request

    def resolve(self):
        trainers_trained_weights = {}
        trainers_sample_size = {}
        for trainer_id, request in self.train_requests.items():
            trained_weights, sample_size = request[0](request[1], request[2], request[3])
            trainers_trained_weights[trainer_id] = trained_weights
            trainers_sample_size[trainer_id] = sample_size
        return trainers_trained_weights, trainers_sample_size


class MPITrainerManager(TrainerManager):

    def __init__(self):
        self.comm = Comm()
        self.procs = [i + 1 for i in range(self.comm.size() - 1)]
        self.used_procs = []
        self.requests = {}

    def train_req(self, trainer_id, model, train_data, context):
        pid = self.get_proc()
        self.comm.send(pid, (model, train_data, context), 1)
        req = self.comm.irecv(pid, 2)
        self.requests[trainer_id] = req

    def get_proc(self):
        for proc in self.procs:
            if proc not in self.used_procs:
                self.used_procs.append(proc)
                return proc
        raise Exception("no more available processes to answer the request. Increase mpi nb proc")

    def reset(self):
        self.used_procs = []
        self.requests = {}

    def resolve(self):
        trainers_trained_weights = {}
        trainers_sample_size = {}
        for trainer_id, req in self.requests.items():
            trained_weights, sample_size = req.wait()
            trainers_trained_weights[trainer_id] = trained_weights
            trainers_sample_size[trainer_id] = sample_size
        self.reset()
        return trainers_trained_weights, trainers_sample_size
