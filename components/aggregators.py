import copy

import torch
from torch import nn

from src.federated.protocols import Trainer, Aggregator


class AVGAggregator(Aggregator):
    def aggregate(self, models_dict: {int: nn.ModuleDict}, sample_dict: {int: int},
                  round_id: int) -> nn.ModuleDict:
        model_list = []
        training_num = 0

        for idx in models_dict.keys():
            model_list.append((sample_dict[idx], copy.deepcopy(models_dict[idx])))
            training_num += sample_dict[idx]

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        return averaged_params
