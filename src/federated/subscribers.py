import atexit
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
from typing.io import IO

from src import tools, manifest
from src.apis.mpi import Comm
from src.data.data_container import DataContainer
from src.federated.events import FederatedEventPlug, Events
from src.federated.federated import FederatedLearning


class Timer(FederatedEventPlug):
    TRAINER = 'trainer'
    FEDERATED = 'federated'
    ROUND = 'round'
    TRAINING = 'training'
    AGGREGATION = 'aggregation'

    def __init__(self, show_only=None):
        super().__init__(None)
        self.ticks = defaultdict(lambda: 0)
        self.show_only = show_only
        self.logger = logging.getLogger("fed_timer")
        if show_only is not None:
            for item in show_only:
                if item not in [Timer.FEDERATED, Timer.ROUND, Timer.AGGREGATION, Timer.TRAINING]:
                    Exception('requested timer does not exists')

    def tick(self, name, is_done):
        now = time.time()
        now_cpu = time.process_time()
        if is_done:
            dif = now - self.ticks[name]
            dif_cpu = now_cpu - self.ticks[name + '_cpu']
            if self.show_only is not None and name not in self.show_only:
                return
            self.logger.info(f'{name}, elapsed: {round(dif, 3)}s')
            self.logger.info(f'{name}, elapsed: {round(dif_cpu, 3)}s of cpu time')
        else:
            self.ticks[name] = now
            self.ticks[name + '_cpu'] = now_cpu

    def on_federated_started(self, params):
        self.tick('federated', False)

    def on_federated_ended(self, params):
        self.tick(self.FEDERATED, True)

    def on_training_start(self, params):
        self.tick(self.TRAINING, False)

    def on_training_end(self, params):
        self.tick(self.TRAINING, True)
        self.tick(self.AGGREGATION, False)

    def on_round_start(self, params):
        self.tick(self.ROUND, False)

    def on_round_end(self, params):
        self.tick(self.ROUND, True)

    def on_aggregation_end(self, params):
        self.tick(self.AGGREGATION, True)

    def on_trainer_start(self, params):
        self.tick(self.TRAINER, False)

    def on_trainer_end(self, params):
        self.tick(self.TRAINER, True)

    def force(self) -> []:
        return [Events.ET_FED_START, Events.ET_TRAINER_FINISHED, Events.ET_TRAINER_STARTED, Events.ET_TRAIN_START,
                Events.ET_TRAIN_END, Events.ET_AGGREGATION_END, Events.ET_INIT, Events.ET_ROUND_START,
                Events.ET_ROUND_FINISHED]


class FederatedLogger(FederatedEventPlug):
    def __init__(self, only=None, detailed_selection=False):
        super().__init__(only)
        self.detailed_selection = detailed_selection
        self.trainers_data_dict = None
        self.logger = logging.getLogger('federated')

    def on_federated_started(self, params):
        params = tools.Dict.but(['context'], params)
        if self.detailed_selection:
            self.trainers_data_dict = params['trainers_data_dict']
        self.logger.info('federated learning started')

    def on_federated_ended(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f'federated learning ended {params}')

    def on_init(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f'federated learning initialized with initial model {params}')

    def on_training_start(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f"training started {params}")

    def on_training_end(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f"training ended {params}")

    def on_aggregation_end(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f"aggregation ended {params}")

    def on_round_end(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f"round ended {params}")
        self.logger.info("----------------------------------------")

    def on_round_start(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f"round started {params}")

    def on_trainer_start(self, params):
        params = tools.Dict.but(['context', 'train_data'], params)
        self.logger.info(f"trainer started {params}")

    def on_trainer_end(self, params):
        params = tools.Dict.but(['context', 'trained_model'], params)
        self.logger.info(f"trainer ended {params}")

    def force(self) -> []:
        return [Events.ET_FED_START]

    def on_trainers_selected(self, params):
        params = tools.Dict.but(['context'], params)
        self.logger.info(f"selected components {params}")
        if self.detailed_selection:
            tools.detail(self.trainers_data_dict, params['trainers_ids'])


class FedPlot(FederatedEventPlug):
    def __init__(self, per_rounds=False, per_federated_local=False, per_federated_total=True):
        super().__init__()
        self.rounds_accuracy = []
        self.rounds_loss = []
        self.local_accuracies = defaultdict(lambda: [])
        self.local_losses = defaultdict(lambda: [])
        plt.interactive(False)
        self.per_rounds = per_rounds
        self.per_federated_local = per_federated_local
        self.per_federated_total = per_federated_total

    def on_federated_ended(self, params):
        if self.per_federated_total:
            fig, axs = plt.subplots(2)
            axs[0].plot(self.rounds_accuracy)
            axs[0].set_title('Total Accuracy')
            axs[1].plot(self.rounds_loss)
            axs[1].set_title('Total Loss')
            fig.suptitle('Total Accuracy & Loss')
            fig.tight_layout()
            plt.show()

        if self.per_federated_local:
            fig, axs = plt.subplots(2)
            for trainer_id, trainer_accuracies in self.local_accuracies.items():
                axs[0].plot(range(len(trainer_accuracies)), trainer_accuracies, label=f'Trainer-{trainer_id}')
                axs[0].set_title('All Trainers Local Accuracy')
                axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                axs[0].set_xticks(range(len(trainer_accuracies)))
            for trainer_id, trainer_losses in self.local_losses.items():
                axs[1].plot(range(len(trainer_losses)), trainer_losses, label=f'Trainer-{trainer_id}')
                axs[1].set_title('All Trainers Local Loss')
                axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                axs[1].set_xticks(range(len(trainer_losses)))
            fig.tight_layout()
            plt.show()

    def on_round_end(self, params):
        self.rounds_accuracy.append(params['accuracy'])
        self.rounds_loss.append(params['loss'])
        local_accuracy = params['local_acc']
        local_loss = params['local_loss']
        for trainer_id, accuracy in local_accuracy.items():
            self.local_accuracies[trainer_id].append(accuracy)
        for trainer_id, loss in local_loss.items():
            self.local_losses[trainer_id].append(loss)
        if self.per_rounds:
            fig, axs = plt.subplots(2)
            axs[0].bar(local_accuracy.keys(), local_accuracy.values())
            axs[0].set_title('Local Round Accuracy')
            axs[0].set_xticks(range(len(self.local_accuracies)))
            axs[1].bar(local_loss.keys(), local_loss.values())
            axs[1].set_title('Local Round Loss')
            axs[1].set_xticks(range(len(self.local_losses)))
            fig.tight_layout()
            plt.show()


class CustomModelTestPlug(FederatedEventPlug):
    def __init__(self, test_data: DataContainer, batch_size, show_plot=True):
        super().__init__()
        self.test_data = test_data
        self.batch_size = 10
        self.history_acc = []
        self.history_loss = []
        self.show_plot = show_plot

    def on_aggregation_end(self, params):
        model = params['global_model']
        acc, loss = tools.infer(model, test_data=self.test_data.batch(self.batch_size))
        self.history_acc.append(acc)
        self.history_loss.append(loss)
        if 'context' in params:
            context: FederatedLearning.Context = params['context']
            context.store(custom_accuracy=acc, custom_loss=loss)
        logging.getLogger('custom_test').info(f'accuracy: {acc}, loss: {loss}')

    def on_federated_ended(self, params):
        fig, axs = plt.subplots(2)
        axs[0].plot(self.history_acc)
        axs[0].set_title('Total Accuracy')
        axs[0].set_xticks(range(len(self.history_acc)))
        axs[1].plot(self.history_loss)
        axs[1].set_title('Total Loss')
        axs[1].set_xticks(range(len(self.history_loss)))
        fig.suptitle('Custom Set Accuracy & Loss')
        fig.tight_layout()
        plt.show()


class FedSave(FederatedEventPlug):
    def __init__(self, folder_name="./logs/"):
        super().__init__()
        self.folder_name = folder_name
        self.file_name = "fedruns.pkl"

    def force(self) -> []:
        return [Events.ET_FED_END]

    def on_federated_ended(self, params):
        context = params['context']
        all = self.old_runs()
        all.append(context)
        with open(self.path(), 'wb') as file:
            pickle.dump(all, file)

    def is_old_exists(self):
        return os.path.exists(path=self.path())

    def old_runs(self):
        runs = []
        if self.is_old_exists():
            with open(self.path(), 'rb') as file:
                runs = pickle.load(file)
        return runs

    def path(self):
        return self.folder_name + self.file_name


class WandbLogger(FederatedEventPlug):
    def __init__(self, config=None):
        super().__init__()
        import wandb
        wandb.login(key='18de3183a3487d875345d2ee7948376df2a31c39')
        wandb.init(project='fedavg', entity='arafeh', config=config)
        self.wandb = wandb
        atexit.register(lambda: self.wandb.finish())

    def on_round_end(self, params):
        self.wandb.log({'acc': params['accuracy'], 'loss': params['loss']})

    def on_federated_ended(self, params):
        self.wandb.finish()


class MPIStopPlug(FederatedEventPlug):

    def on_federated_ended(self, params):
        Comm().stop()


class Resumable(FederatedEventPlug):
    def __init__(self, tag, federated: FederatedLearning, verbose=1):
        super().__init__()
        os.makedirs(manifest.ROOT_PATH + "/checkpoints", exist_ok=True)
        self.federated = federated
        self.file_name = manifest.ROOT_PATH + "/checkpoints" + "/run_" + tag + ".fed"
        self.verbose = verbose

    def on_init(self, params):
        if os.path.exists(self.file_name):
            self.log('found a checkpoint, loading...')
            file = open(self.file_name, 'rb')
            loaded = pickle.load(file)
            context = loaded['context']
            is_finished = loaded['is_finished']
            self.federated.context = context
            self.federated.is_finished = is_finished
            file.close()

    def on_round_end(self, params):
        file = open(self.file_name, 'wb')
        to_save = {
            'context': self.federated.context,
            'is_finished': self.federated.is_finished,
        }
        self.log('saving checkpoint...')
        pickle.dump(to_save, file)
        file.close()

    def log(self, msg):
        if self.verbose == 1:
            logging.getLogger('resumable').info(msg)
