import torch


def sgd(lr):
    """
    new instance of sgd optimizer
    :param lr: learn rate
    :return: new instance creator of sgd optim
    """
    return lambda model: torch.optim.SGD(model.parameters(), lr=lr)


def adam(lr, wd, amsgrad=True):
    """
    new instance of adam optimizer
    :param lr: learn rate
    :param wd: weight_decay
    :param amsgrad:
    :return: new instance creator of adam optim
    """
    return lambda model: torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd,
                                          amsgrad=amsgrad)
