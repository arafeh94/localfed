import torch


def sgd(lr):
    return lambda model: torch.optim.SGD(model.parameters(), lr=lr)
