import logging

from torch import nn

from src import tools
from src.apis import genetic
from src.apis.context import Context


def ga_module_creator(clients_data, init_model, max_iter=10, r_cross=0.1, r_mut=0.05, c_size=10,
                      p_size=20) -> nn.Module:
    context = Context(clients_data, init_model)
    context.build(ratio=0.1)
    clustered = tools.Clustered(context.cluster(10))
    best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=0, max_iter=max_iter,
                                     r_cross=r_cross, r_mut=r_mut, c_size=c_size, p_size=p_size)
    logging.getLogger('ga').info(best)
    global_model = context.aggregate_clients(best)
    return lambda: global_model
