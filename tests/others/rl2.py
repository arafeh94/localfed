import copy
import logging
import random
from collections import namedtuple
from random import randint

import numpy as np
import torch
from torch import optim, nn

from libs.model.dqn import DeepQNetwork
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider

logging.basicConfig(level=logging.INFO)

# data
test_data = PickleDataProvider('../../datasets/pickles/test_data.pkl').collect().as_tensor()
dp = PickleDataProvider("../../datasets/pickles/mnist10k.pkl")
dg = DataGenerator(dp)
client_data = dg.distribute_size(50, 50, 50)

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))

# DeepQNetwork configurations
dqn = DeepQNetwork(400350, 64, 16, 50)
optimizer = optim.Adam(dqn.parameters(), lr=1e-6)
criterion = nn.MSELoss()
dnq_learn_batch = 10
epsilon = 0.99
eps_reduction = 0.01
gamma = 0.99

# Federated Learning configurations
train_model = LogisticRegression(28 * 28, 10)
per_round = 10
batch_size = 50

# memory variables
client_weights = {}
sample_size = {}
memory = []


def client_train(clients_ids):
    train_data = tools.dict_select(clients_ids, client_data)
    for client_id, data in train_data.items():
        client_model = copy.deepcopy(train_model)
        weights = tools.train(client_model, data.batch(batch_size))
        client_weights[client_id] = weights
        sample_size[client_id] = len(data)


def aggregate(client_ids):
    client_models = tools.dict_select(client_ids, client_weights)
    samples = tools.dict_select(client_models, sample_size)
    new_weights = tools.aggregate(client_models, samples)
    tools.load(train_model, new_weights)


def get_state(as_tensor=True):
    states = [tools.flatten_weights(train_model.state_dict())]
    for client_id in sorted(client_weights.keys()):
        states.append(tools.flatten_weights(client_weights[client_id]))
    states = np.array(states)
    return torch.Tensor(states) if as_tensor else states


def memories(state, action, reward, new_state):
    transition = Transition(state, action, reward, new_state)
    memory.append(transition)


def action(state):
    global epsilon
    if epsilon > random.uniform(0, 1):
        epsilon += eps_reduction
        client_eval = np.random.dirichlet(np.ones(len(client_data)), size=1)[0]
        return torch.topk(torch.tensor(client_eval), per_round)[1].numpy(), client_eval
    else:
        client_eval = dqn(state.flatten())
        return torch.topk(client_eval, per_round)[1].numpy(), client_eval


def calculate_rewards():
    acc, _ = tools.infer(train_model, test_data.batch(batch_size))
    return acc


def learn():
    if len(memory) < 2:
        return
    learn_sample = random.sample(memory, 2)
    state_batch = torch.cat(tuple(torch.unsqueeze(d[0].flatten(), 0) for d in learn_sample))
    new_state_batch = torch.cat(tuple(torch.unsqueeze(d[3].flatten(), 0) for d in learn_sample))
    action_batch = torch.cat(tuple(torch.unsqueeze(d[1], 0) for d in learn_sample))
    reward_batch = torch.tensor([d[2] for d in learn_sample])
    output_batch = dqn(state_batch)
    output_1_batch = dqn(new_state_batch)
    y_batch = torch.Tensor([])
    for i in range(len(learn_sample)):
        y = reward_batch[i] + gamma * torch.max(output_1_batch[i])
        y_batch = torch.cat((y_batch, y.view(1)))
    q_value = torch.sum(output_batch * action_batch, dim=1)
    optimizer.zero_grad()
    y_batch = y_batch.detach()
    loss = criterion(q_value, y_batch)
    loss.backward()
    optimizer.step()


def main():
    client_train(clients_ids=client_data.keys())
    aggregate(client_ids=client_data.keys())

    current_state = get_state()
    while True:
        selected_clients, r_action = action(current_state)
        client_train(clients_ids=selected_clients)
        aggregate(client_ids=selected_clients)
        reward = calculate_rewards()
        next_state = get_state()
        memories(current_state, r_action, reward, next_state)
        learn()


main()
