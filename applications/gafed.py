from torch import nn

from applications.fedavg import FedAVG
from src import tools, genetic
from src.context import Context


class GAFedAVG(FedAVG):
    def init(self) -> nn.Module:
        context = Context(self.trainers_data_dict, self.test_data, self.create_model)
        context.build()
        clustered = tools.Clustered(context.cluster(10))
        best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=0, max_iter=10,
                                         r_cross=0.1, r_mut=0.05, c_size=10, p_size=20)
        print("best result:", best)
        global_model = context.aggregate_clients(best)
        return global_model
