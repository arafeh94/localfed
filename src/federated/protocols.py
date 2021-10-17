from abc import abstractmethod, ABC
from functools import reduce
from typing import Dict, Tuple, List

from torch import nn, Tensor

from src.data.data_container import DataContainer
from src.federated.components import params


class Identifiable(ABC):
    @abstractmethod
    def id(self):
        pass


class Aggregator(Identifiable):
    @abstractmethod
    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        pass


class ModelInfer(Identifiable):
    def __init__(self, batch_size: int, criterion):
        self.batch_size = batch_size
        self.criterion = criterion
        if isinstance(criterion, str):
            self.criterion = params.criterion(criterion)

    @abstractmethod
    def infer(self, model: nn.Module, test_data: DataContainer):
        pass


class TrainerParams(Identifiable):

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

    def id(self):
        args_str = ''
        for arg in self.args:
            args_str += f'{arg}({self.args[arg]})_'
        args_str = args_str.removesuffix('_')
        return f'e({self.epochs})_b({self.batch_size})_crt({self.criterion})_opt({self.optimizer})_{args_str}'


class Trainer(Identifiable):
    @abstractmethod
    def train(self, model: nn.Module, train_data: DataContainer, context, config: TrainerParams) -> Tuple[
        Dict[str, Tensor], int]:
        pass


class ClientSelector(Identifiable):
    @abstractmethod
    def select(self, client_ids: List[int], context) -> List[int]:
        pass
