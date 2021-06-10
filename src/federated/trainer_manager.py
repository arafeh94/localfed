import logging
from abc import ABC, abstractmethod
from functools import reduce
from typing import List

from torch import nn

from src.apis.mpi import Comm
from src.data.data_container import DataContainer
from src.federated.protocols import Trainer, TrainerParams


class TrainerManager:
    @abstractmethod
    def train_req(self, trainer_id, model, train_data, context, config: TrainerParams):
        pass

    @abstractmethod
    def resolve(self):
        pass


class SeqTrainerManager(TrainerManager):
    class TrainerProvider(ABC):
        @abstractmethod
        def collect(self, trainer_id, config: TrainerParams) -> Trainer:
            pass

    def __init__(self, trainer_provider: TrainerProvider = None):
        self.train_requests = {}
        self.trainer_provider = trainer_provider
        if trainer_provider is None:
            self.trainer_provider = SharedTrainerProvider()

    def train_req(self, trainer_id, model, train_data, context, config):
        trainer = self.trainer_provider.collect(trainer_id, config)
        request = [trainer.train, model, train_data, context, config]
        self.train_requests[trainer_id] = request
        return request

    def resolve(self):
        trainers_trained_weights = {}
        trainers_sample_size = {}
        for trainer_id, request in self.train_requests.items():
            trained_weights, sample_size = request[0](request[1], request[2], request[3], request[4])
            trainers_trained_weights[trainer_id] = trained_weights
            trainers_sample_size[trainer_id] = sample_size
        self.train_requests = {}
        return trainers_trained_weights, trainers_sample_size


class SharedTrainerProvider(SeqTrainerManager.TrainerProvider):
    def __init__(self):
        self.trainers = {}

    def collect(self, trainer_id, config: TrainerParams) -> Trainer:
        return self._trainer(trainer_id, config)

    def _create(self, trainer_id, config: TrainerParams) -> Trainer:
        trainer = config.trainer_class()
        self.trainers[trainer_id] = trainer
        return trainer

    def _trainer(self, trainer_id, config):
        if trainer_id not in self.trainers.keys():
            self.trainers[trainer_id] = self._create(trainer_id, config)
        return self.trainers[trainer_id]


class MPITrainerManager(TrainerManager):

    def __init__(self):
        self.comm = Comm()
        self.procs = [i + 1 for i in range(self.comm.size() - 1)]
        self.used_procs = []
        self.requests = {}

    def train_req(self, trainer_id, model, train_data, context, config):
        pid = self.get_proc()
        self.comm.send(pid, (model, train_data, context, config), 1)
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

    @staticmethod
    def mpi_trainer_listener(comm: Comm):
        trainer: Trainer = None
        while True:
            model, train_data, context, config = comm.recv(0, 1)
            if trainer is None:
                trainer = config.trainer_class()
            trained_weights, sample_size = trainer.train(model, train_data, context, config)
            comm.send(0, (trained_weights, sample_size), 2)
