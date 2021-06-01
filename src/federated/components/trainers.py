import logging
from abc import ABC
from typing import Tuple, Dict

from torch import nn, Tensor

from src.apis.mpi import Comm
from src.data.data_container import DataContainer
from src.federated.federated import FederatedLearning
from src.federated.protocols import Trainer, TrainerParams


class CPUTrainer(Trainer):
    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams) -> Tuple[any, int]:
        model.train()
        optimizer = config.get_optimizer()(model)
        criterion = config.get_criterion()

        epoch_loss = []
        for epoch in range(config.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data.batch(config.batch_size)):
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        weights = model.cpu().state_dict()
        return weights, len(train_data)


class CPUChunkTrainer(CPUTrainer):
    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams) -> Tuple[any, int]:
        round_id = context.round_id
        num_rounds = context.num_rounds
        total_size = len(train_data)
        round_data_size = total_size / num_rounds
        x = train_data.x[int(round_id * round_data_size):int((round_id * round_data_size) + round_data_size)]
        y = train_data.y[int(round_id * round_data_size):int((round_id * round_data_size) + round_data_size)]
        chunk = DataContainer(x, y)
        return super(CPUChunkTrainer, self).train(model, chunk, round_id, config)
