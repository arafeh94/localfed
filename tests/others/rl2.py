import logging
from collections import namedtuple

import numpy as np
import torch

from libs.model.dqn.rl import DQN
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider

logging.basicConfig(level=logging.INFO)
dp = PickleDataProvider("../../datasets/pickles/mnist10k.pkl")
dg = DataGenerator(dp)
client_data = dg.distribute_size(50, 50, 50)

Transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))
memory = []

model = DQN(50)

# state 1 (global_model, model1,model2,model3,...)
train_model = LogisticRegression(28 * 28, 10)
client_weights = {}
sample_size = {}
for client_id, data in client_data.items():
    weights = tools.train(train_model, data.batch(9999), 1)
    client_weights[client_id] = weights
    sample_size[client_id] = len(data)
global_model = tools.aggregate(client_weights, sample_size)
flattened = {}
for client_id, weight in client_weights.items():
    flattened[client_id] = tools.flatten_weights(weight)
global_model_flat = tools.flatten_weights(global_model)
all = np.array([global_model_flat, *flattened.values()])
state = torch.unsqueeze(torch.Tensor(all), 0)

action = model(state)

tools.load(train_model, global_model)
reward, loss = tools.infer(global_model, None)
memory.append((state, action, reward))
