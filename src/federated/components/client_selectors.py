import random
from typing import List
from src.federated.protocols import ClientSelector


class All(ClientSelector):
    def select(self, trainer_ids: List[int], round_id: int) -> List[int]:
        return trainer_ids


class Random(ClientSelector):
    def __init__(self, num):
        self.num = num

    def select(self, trainer_ids: List[int], round_id: int) -> List[int]:
        select_size = self.num
        if self.num <= 1:
            select_size = int(self.num * len(trainer_ids))
        select_size = 3 if select_size < 3 else select_size
        selected_trainers = random.sample(trainer_ids, select_size)
        return selected_trainers
