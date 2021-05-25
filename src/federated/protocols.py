from abc import abstractmethod, ABC
from functools import reduce
from typing import Dict, Tuple, List

from torch import nn, Tensor

from src.data.data_container import DataContainer


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        pass


class ModelInfer(ABC):
    def __init__(self, batch_size: int, criterion):
        self.batch_size = batch_size
        self.criterion = criterion

    @abstractmethod
    def infer(self, model: nn.Module, test_data: DataContainer):
        pass


class Trainer:
    def __init__(self, **kwargs):
        self.optimizer = None
        self.criterion = None
        self.epochs = None
        self.batch_size = None

    def __init(self, **kwargs):
        for item, value in kwargs.items():
            self.__setattr__(item, value)
        self.on_create()

    def on_create(self):
        pass

    @abstractmethod
    def train(self, model: nn.Module, train_data: DataContainer, context) -> Tuple[Dict[str, Tensor], int]:
        pass


class ClientSelector:
    @abstractmethod
    def select(self, trainer_ids: List[int], round_id: int) -> List[int]:
        pass
