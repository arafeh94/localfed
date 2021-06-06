import logging
import math
from collections import defaultdict
from typing import List, Dict

from matplotlib import pyplot as plt

from src.federated.federated import FederatedLearning


class FedRuns:
    def __init__(self, runs: Dict[str, FederatedLearning] or List[FederatedLearning]):
        self.runs = runs
        self.logger = logging.getLogger('runs')
        if isinstance(runs, list):
            new_dict = {}
            for index, run in enumerate(runs):
                new_dict[f'run{index}'] = run
            self.runs = new_dict

    def append(self, name, run):
        if isinstance(run, FederatedLearning):
            run = run.context
        self.runs[name] = run

    def compare_all(self):
        for i, first in enumerate(self.runs):
            for j, second in enumerate(self.runs):
                if i > j:
                    self.logger.info(
                        f'comparing {first} to {second}: {self.compare(self.runs[first], self.runs[second])}')

    def compare(self, first, second, verbose=1):
        local_history = first.history
        other_history = second.history
        performance_history = defaultdict(lambda: [])
        diff = {}
        for round_id, first_data in local_history.items():
            if round_id not in other_history:
                continue
            second_data = other_history[round_id]
            for item in first_data:
                if type(first_data[item]) in [int, float, str]:
                    performance_history[item].append(first_data[item] - second_data[item])
        for item, val in performance_history.items():
            diff[item] = math.fsum(val) / len(val)
        if verbose == 1:
            return diff
        else:
            return diff, performance_history

    def plot(self):
        acc_plot = {}
        loss_plot = {}
        for name, run in self.runs.items():
            acc, loss = [], []
            for round_id, performance in run.history.items():
                acc.append(performance['acc'])
                loss.append(performance['loss'])
            acc_plot[name] = acc
            loss_plot[name] = loss
        fig, axs = plt.subplots(2)

        for run_name, acc in acc_plot.items():
            axs[0].plot(acc, label=run_name)
            axs[0].set_title('Total Accuracy')
            axs[0].set_xticks(range(len(acc)))
        for run_name, loss in loss_plot.items():
            axs[1].plot(loss, label=run_name)
            axs[1].set_title('Total Loss')
            axs[1].set_xticks(range(len(loss)))

        plt.legend()
        plt.tight_layout()
        plt.show()
