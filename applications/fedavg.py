import random
from torch import nn

from src import tools
from src.data_container import DataContainer
from src.federated import AbstractFederated


class FedAVG(AbstractFederated):
    def init(self) -> nn.Module:
        return self.create_model()

    def select(self, trainer_ids: [int], round_id: int) -> [int]:
        selected_trainers = random.sample(trainer_ids, self.trainer_per_round)
        return selected_trainers

    def train(self, global_model_weights: nn.ModuleDict, trainers_data: {int: DataContainer}, round_id: int) -> (
            {int: nn.ModuleDict}, {int: int}):
        trained_client_model, sample_size_dict = \
            tools.threaded_train(self.get_global_model(), trainers_data, self.batch_size)
        return trained_client_model, sample_size_dict

    def aggregate(self, trainers_models_weight_dict: {int: nn.ModuleDict}, sample_size: {int: int},
                  round_id: int) -> nn.ModuleDict:
        return tools.aggregate(trainers_models_weight_dict, sample_size)
