import time
import logging
from collections import defaultdict

import numpy as np

from src import tools
from src.federated import FederatedEventPlug, Events
import matplotlib.pyplot as plt


class FederatedTimer(FederatedEventPlug):
    def __init__(self, only=None):
        super().__init__(only)
        self.last_tick = 0
        self.first_tick = 0
        self.logger = logging.getLogger("fed_timer")

    def tick(self, name):
        now = time.process_time()
        dif = now - self.last_tick
        self.last_tick = now
        self.logger.info(f'{name}, elapsed: {dif}s')

    def on_federated_started(self, params):
        self.last_tick = time.process_time()
        self.first_tick = time.process_time()

    def on_federated_ended(self, params):
        self.tick('fed end')
        dif = time.process_time() - self.first_tick
        self.logger.info(f"federated total time: {dif}s")

    def on_init(self, params):
        pass

    def on_training_start(self, params):
        pass

    def on_training_end(self, params):
        self.tick('training')

    def on_aggregation_end(self, params):
        self.tick('aggregation')

    def on_round_end(self, params):
        pass

    def on_round_start(self, params):
        pass

    def on_trainers_selected(self, params):
        self.tick('trainer selection')


class FederatedLogger(FederatedEventPlug):
    def __init__(self, only=None, detailed_selection=False):
        super().__init__(only)
        self.detailed_selection = detailed_selection
        self.trainers_data_dict = None
        self.logger = logging.getLogger('federated')

    def on_federated_started(self, params):
        if self.detailed_selection:
            self.trainers_data_dict = params['trainers_data_dict']
        self.logger.info('federated learning started')

    def on_federated_ended(self, params):
        self.logger.info(f'federated learning ended {params}')

    def on_init(self, params):
        self.logger.info(f'federated learning initialized with initial model {params}')

    def on_training_start(self, params):
        self.logger.info(f"training started {params}")

    def on_training_end(self, params):
        self.logger.info(f"training ended {params}")

    def on_aggregation_end(self, params):
        self.logger.info(f"aggregation ended {params}")

    def on_round_end(self, params):
        self.logger.info(f"round ended {params}")
        self.logger.info("----------------------------------------")

    def on_round_start(self, params):
        self.logger.info(f"round started {params}")

    def force(self) -> []:
        return [Events.ET_FED_START]

    def on_trainers_selected(self, params):
        self.logger.info(f"selected trainers {params}")
        if self.detailed_selection:
            tools.detail(self.trainers_data_dict, params['trainers_ids'])


class FedPlot(FederatedEventPlug):
    def __init__(self):
        super().__init__()
        self.rounds_accuracy = []
        self.rounds_loss = []
        self.local_accuracies = defaultdict(lambda: [])
        self.local_losses = defaultdict(lambda: [])

    def on_federated_started(self, params):
        pass

    def on_federated_ended(self, params):
        fig, axs = plt.subplots(2)
        axs[0].plot(self.rounds_accuracy)
        axs[0].set_title('Total Accuracy')
        axs[1].plot(self.rounds_loss)
        axs[1].set_title('Total Loss')
        fig.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2)
        for trainer_id, trainer_accuracies in self.local_accuracies.items():
            axs[0].plot(range(len(trainer_accuracies)), trainer_accuracies, label=f'Trainer-{trainer_id}')
            axs[0].set_title('All Trainers Local Accuracy')
            axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        for trainer_id, trainer_losses in self.local_losses.items():
            axs[1].plot(range(len(trainer_losses)), trainer_losses, label=f'Trainer-{trainer_id}')
            axs[1].set_title('All Trainers Local Loss')
            axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        plt.show()

    def on_init(self, params):
        pass

    def on_training_start(self, params):
        pass

    def on_training_end(self, params):
        pass

    def on_aggregation_end(self, params):
        pass

    def on_round_end(self, params):
        self.rounds_accuracy.append(params['accuracy'])
        self.rounds_loss.append(params['loss'])
        local_accuracy = params['local_acc']
        local_loss = params['local_loss']
        for trainer_id, accuracy in local_accuracy.items():
            self.local_accuracies[trainer_id].append(accuracy)
        for trainer_id, loss in local_loss.items():
            self.local_losses[trainer_id].append(loss)
        fig, axs = plt.subplots(2)
        axs[0].bar(local_accuracy.keys(), local_accuracy.values())
        axs[0].set_title('Local Round Accuracy')
        axs[1].bar(local_loss.keys(), local_loss.values())
        axs[1].set_title('Local Round Loss')
        fig.tight_layout()
        plt.show()

    def on_round_start(self, params):
        pass

    def on_trainers_selected(self, params):
        pass
