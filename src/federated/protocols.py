from abc import abstractmethod, ABC
from torch import nn

from src.data.data_container import DataContainer


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, trainers_models_weight_dict: {int: nn.ModuleDict}, sample_size: {int: int},
                  round_id: int) -> nn.ModuleDict:
        pass


class ModelInfer(ABC):
    def __init__(self, batch_size, criterion):
        self.batch_size = batch_size
        self.criterion = criterion

    @abstractmethod
    def infer(self, model, test_data):
        pass


class Trainer(ABC):
    def __init__(self, batch_size: int, epochs: int, criterion, optimizer: callable):
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

    @abstractmethod
    def train(self, model: nn.Module, train_data: DataContainer, context):
        pass


class ClientSelector:
    @abstractmethod
    def select(self, trainer_ids: [int], round_id: int) -> [int]:
        pass
