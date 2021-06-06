import logging

from torch import nn

from apps.genetic_selectors.src.cluster_selector import ClusterSelector
from src import tools
from apps.genetic_selectors.src import genetic
from apps.genetic_selectors.src.context import Context


def ga_module_creator(clients_data, init_model, max_iter=10, r_cross=0.1, r_mut=0.05, c_size=10,
                      p_size=20, clusters=10, desired_fitness=0.5) -> nn.Module:
    context = Context(clients_data, init_model)
    context.train(ratio=0.1)
    clustered = ClusterSelector(context.cluster(clusters))
    best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=desired_fitness,
                                     max_iter=max_iter, r_cross=r_cross, r_mut=r_mut, c_size=c_size, p_size=p_size)
    logging.getLogger('ga').info(best)
    global_model = context.aggregate_clients(best)
    return lambda: global_model


def cluster_module_creator(clients_data, init_model, clusters=10, c_size=1):
    context = Context(clients_data, init_model)
    context.train(ratio=0.1)
    clustered = tools.ClusterSelector(context.cluster(clusters))
    selected_idx = []
    while len(selected_idx) < c_size:
        available = clustered.list()
        selected_idx.append(clustered.select(available[0]))
    global_model = context.aggregate_clients(selected_idx)
    return lambda: global_model
