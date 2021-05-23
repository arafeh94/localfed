from torch import nn

from components.fedavg import FedAVG
from components.components import trainers
from src.data.data_container import DataContainer


class FedContinuous(FedAVG):
    """
    special implementation for continuous federated learning that have 2 modification on fedavg
    1- in each round all components are selected
    2- control the data of each round in while training (refer to method @self.train)
    """

    def select(self, trainer_ids: [int], round_id: int) -> [int]:
        return trainer_ids

    def train(self, global_model_weights: nn.ModuleDict, trainers_data: {int: DataContainer}, round_id: int) -> (
            {int: nn.ModuleDict}, {int: int}):
        """
        different from normal federated learning where each round we select all the client data,
        in continuous we select part of the client data in each round.
        for e.g. in client have 100 row of data, for 10 round federated each round will have 10 row of data
        :param global_model_weights:
        :param trainers_data:
        :param round_id:
        :return:
        """
        trainers_data_round = {}
        for trainer_id, data in trainers_data.items():
            total_size = len(data)
            round_data_size = total_size / self.num_rounds
            x = data.x[int(round_id * round_data_size):int((round_id * round_data_size) + round_data_size)]
            y = data.y[int(round_id * round_data_size):int((round_id * round_data_size) + round_data_size)]
            trainers_data_round[trainer_id] = DataContainer(x, y)

        trained_client_model, sample_size_dict = trainers.MultiTrainerTrain(
            self.batch_size, self.epochs, self.criterion, self.optimizer).train(self.get_global_model(), trainers_data)
        return trained_client_model, sample_size_dict
