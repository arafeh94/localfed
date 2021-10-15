import os
import pickle
import typing

import numpy as np
from matplotlib import pyplot as plt

from src import manifest
from src.apis.extensions import Serializable


class AccuracyCompare(Serializable):
    def __init__(self):
        super().__init__(manifest.DEFAULT_ACC_PATH)
        self.accuracies = {}
        self.load()

    def _append(self, tag, val):
        self.accuracies[tag] = val

    def append(self, tag, val):
        self.sync(self._append, tag, val)

    def save_accuracy(self, federated_learning, tag):
        def reducer(first, key, val):
            return [val['acc']] if first is None else np.append(first, val['acc'])

        all_acc = federated_learning.context.history.reduce(reducer)
        self.append(tag, all_acc)

    def get_saved_accuracy(self, filter: typing.Callable[[str], bool] = None):
        self.load()
        if filter is None:
            return self.accuracies
        else:
            accs = {}
            for tag, vals in self.accuracies.items():
                is_not_filtered = False if filter is None else filter(tag)
                if vals is not None and is_not_filtered:
                    accs[tag] = vals
            return accs

    def show_saved_accuracy_plot(self, filter: typing.Callable[[str], bool] = None, title=None):
        colors = {'cluster': '#AA4499', 'basic': '#DDCC77', 'genetic': 'blue', 'warmup': '#117733'}
        line = {'cluster': '--', 'basic': '-o', 'genetic': '-+', 'warmup': '-*'}
        accs = self.get_saved_accuracy(filter)
        if len(accs) < 1:
            return
        last_tag: str = None
        for tag, vals in accs.items():
            tag: str
            label = tag.split('_')[0].capitalize()
            plt.plot(vals, line[label.lower()], label=label, color=colors[label.lower()])
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.legend()
            last_tag = tag
        last_tag = " - ".join(map(lambda s: s.upper(), last_tag.split('_')[1:]))
        last_tag = last_tag.replace("- LR01", "").replace("E", "E: ").replace("B", "B: ") \
            .replace('- R', '- R: ').replace('- S', "- Ψ: ").replace('CR', 'CR: ') \
            .replace('01', '0.1').replace('05', '0.5').replace('02', '0.2').replace('CR: 10', 'CR: 1')
        title = last_tag if title is None else title
        plt.ylim()
        plt.title(title)
        plt.savefig('./pics/' + title.replace(':', '').replace('-', '').replace(' ', '').replace('.', '') + '.png',
                    bbox_inches='tight')
        plt.show()


class DivergenceCompare(Serializable):
    def __init__(self):
        super().__init__(manifest.DEFAULT_DIV_PATH)
        self.divergences = {}
        self.load()

    def _append(self, tag, val):
        self.divergences[tag] = val

    def append(self, tag, val):
        self.sync(self._append, tag, val)

    def save_divergence(self, div_evolution_list, tag):
        self.append(tag, div_evolution_list)

    def get_saved_divergences(self, filter: typing.Callable[[str], bool] = None):
        self.load()
        if filter is None:
            return self.divergences
        else:
            accs = {}
            for tag, vals in self.divergences.items():
                is_not_filtered = False if filter is None else filter(tag)
                if vals is not None and is_not_filtered:
                    accs[tag] = vals
            return accs

    def show_saved_divergences_plot(self, filter: typing.Callable[[str], bool] = None, title=None):
        colors = {'cluster': '#AA4499', 'basic': '#DDCC77', 'genetic': 'blue', 'warmup': '#117733'}
        line = {'cluster': '-', 'basic': '', 'genetic': '-', 'warmup': '-'}
        divs = self.get_saved_divergences(filter)
        if len(divs) == 0:
            return
        last_tag: str = None
        for tag, vals in divs.items():
            tag: str
            label = tag.split('_')[0].capitalize()
            plt.plot(vals, line[label.lower()], label=label, color=colors[label.lower()])
            plt.xlabel("Round")
            plt.ylabel("EMD")
            plt.legend()
            last_tag = tag
        last_tag = " - ".join(map(lambda s: s.upper(), last_tag.split('_')[1:]))
        last_tag = last_tag.replace("- LR01", "").replace("E", "E: ").replace("B", "B: ") \
            .replace('- R', '- R: ').replace('- S', "- Ψ: ").replace('CR', 'CR: ') \
            .replace('01', '0.1').replace('05', '0.5').replace('02', '0.2').replace('CR: 10', 'CR: 1')
        title = last_tag if title is None else title
        plt.ylim()
        plt.title(title)
        plt.savefig('./pics/' + title.replace(':', '').replace('-', '').replace(' ', '') + '.png', bbox_inches='tight')
        plt.show()


accuracies = AccuracyCompare()
divergences = DivergenceCompare()
