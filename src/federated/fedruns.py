import logging
from typing import List, Dict

from matplotlib import pyplot as plt

from src.federated.federated import FederatedLearning


class FedRuns:
    def __init__(self, runs: Dict[str, FederatedLearning]):
        self.runs = runs
        self.logger = logging.getLogger('runs')

    def compare(self):
        for i, first in enumerate(self.runs):
            for j, second in enumerate(self.runs):
                if i > j:
                    self.logger.info(
                        f'comparing {first} to {second}: {self.runs[first].compare(self.runs[second])}')

    def plot(self):
        acc_plot = {}
        loss_plot = {}
        for name, run in self.runs.items():
            acc, loss = [], []
            for round_id, performance in run.context.history.items():
                acc.append(performance['acc'])
                loss.append(performance['loss'])
            acc_plot[name] = acc
            loss_plot[name] = loss
        for run_name, acc in acc_plot.items():
            plt.plot(acc, label=run_name)
            plt.legend()
            plt.tight_layout()
        plt.show()
        for run_name, loss in loss_plot.items():
            plt.plot(loss, label=run_name)
            plt.legend()
            plt.tight_layout()
        plt.show()
