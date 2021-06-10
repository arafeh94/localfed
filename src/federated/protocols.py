from abc import abstractmethod, ABC
from functools import reduce
from typing import Dict, Tuple, List

from torch import nn, Tensor

from src.data.data_container import DataContainer
from src.federated.components import params


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


class TrainerParams:
    def __init__(self, trainer_class: type, batch_size: int, epochs: int,
                 criterion: str, optimizer: str, **kwargs):
        self.epochs = epochs
        self.criterion = criterion
        self.batch_size = batch_size
        self.trainer_class = trainer_class
        self.optimizer = optimizer
        self.args = kwargs

    def get_optimizer(self):
        return params.optimizer(self.optimizer, **self.args)

    def get_criterion(self):
        return params.criterion(self.criterion, **self.args)


class Trainer:
    @abstractmethod
    def train(self, model: nn.Module, train_data: DataContainer, context, config: TrainerParams) -> Tuple[
        Dict[str, Tensor], int]:
        pass


class ClientSelector:
    @abstractmethod
    def select(self, client_ids: List[int], context) -> List[int]:
        pass
