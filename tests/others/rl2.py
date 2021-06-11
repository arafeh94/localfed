import copy
import logging
import random
from collections import namedtuple
from random import randint

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable

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
dqn = DeepQNetwork(2040, 64, 16, 50)
optimizer = optim.Adam(dqn.parameters(), lr=0.1)
criterion = nn.MSELoss()
dnq_learn_batch = 10
epsilon = 0.99
eps_reduction = 0.1
GAMMA = 0.99

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
    states = [tools.flatten_weights(train_model.state_dict(), compress=True)]
    for client_id in sorted(client_weights.keys()):
        states.append(tools.flatten_weights(client_weights[client_id], compress=True))
    states = np.array(states)
    return torch.Tensor(states) if as_tensor else states


def memories(state, action, reward, new_state):
    transition = Transition(state, action, reward, new_state)
    memory.append(transition)


def action(state):
    global epsilon
    if epsilon > random.uniform(0, 1):
        print('random')
        epsilon -= eps_reduction
        client_eval = np.random.dirichlet(np.ones(len(client_data)), size=1)[0]
        client_eval = torch.tensor(client_eval)
        return torch.topk(client_eval, per_round)[1].numpy(), client_eval.numpy()
    else:
        print('dqn')
        client_eval = dqn(state.flatten())
        return torch.topk(client_eval, per_round)[1].numpy(), client_eval.detach().numpy()


def calculate_rewards():
    acc, _ = tools.infer(train_model, test_data.batch(batch_size))
    return acc


def learn(batch_size):
    if len(memory) < batch_size:
        return
    learn_sample = random.sample(memory, batch_size)
    state_batch = torch.cat(tuple(torch.unsqueeze(d[0].flatten(), 0) for d in learn_sample))
    next_state = torch.cat(tuple(torch.unsqueeze(d[3].flatten(), 0) for d in learn_sample))
    action_batch = torch.cat(tuple(torch.unsqueeze(torch.tensor(d[1]), 0) for d in learn_sample))
    reward_batch = torch.tensor([d[2] for d in learn_sample])

    state_action_values = dqn(state_batch).gather(1, action_batch)
    next_state_values = torch.topk(dqn(next_state), per_round)[1]
    expected_state_action_values = (next_state_values * GAMMA) + (0.9 * ((64 ** (reward_batch - 0.99)) - 1))

    # print(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss = criterion(state_action_values.double(), expected_state_action_values.double())
    print("loss:", loss)
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
        print(reward, selected_clients)
        next_state = get_state()
        memories(current_state, selected_clients, reward, next_state)
        current_state = next_state
        learn(10)


torch.autograd.set_detect_anomaly(True)
main()
