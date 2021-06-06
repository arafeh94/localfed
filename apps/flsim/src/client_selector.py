import operator
from collections import defaultdict
from typing import List

import numpy as np

from src import tools
from src.federated.federated import FederatedLearning
from src.federated.protocols import ClientSelector


class RLSelector(ClientSelector):

    def __init__(self, per_round, client_director, w_first):
        self.per_round = per_round
        self.punishment = defaultdict(lambda: 0)
        self.client_director = client_director
        self.w_previous = w_first

    def select(self, client_ids: List[int], context: FederatedLearning.Context) -> List[int]:
        w_current = tools.flatten_weights(context.model.state_dict())
        w_previous = self.w_previous
        model_direction = w_current - w_previous
        model_direction = model_direction / np.sqrt(np.dot(model_direction, model_direction))
        self.w_previous = w_current
        client_score = {}
        for client_id, director in self.client_director.items():
            client_score[client_id] = np.dot(director, model_direction)

        for client_id in client_score:
            client_score[client_id] = client_score[client_id] * (0.9) ** self.punishment[client_id]

        selected_clients = []
        for _ in range(self.per_round):
            top_score_index = max(client_score.items(), key=operator.itemgetter(1))[0]
            selected_clients.append(top_score_index)
            client_score[top_score_index] = min(client_score.values()) - 1

        for item in client_ids:
            self.punishment[item] = self.punishment[item] + 1 if item in selected_clients else 0

        return selected_clients
