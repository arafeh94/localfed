import random
from typing import List

import torch

from src.federated.protocols import Trainer, ClientSelector


class All(ClientSelector):
    def select(self, trainer_ids: List[int], round_id: int) -> List[int]:
        return trainer_ids


class Random(ClientSelector):
    def __init__(self, nb_trainers):
        self.nb_trainers = nb_trainers

    def select(self, trainer_ids: List[int], round_id: int) -> List[int]:
        selected_trainers = random.sample(trainer_ids, self.nb_trainers)
        return selected_trainers
