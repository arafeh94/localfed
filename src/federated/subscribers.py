import atexit
import logging
import os
import pickle
import time
import typing
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    def __init__(self, plot_each_round=False, show_local=False, show_global=True, show_acc=True, show_loss=True):
        super().__init__()
        self.rounds_accuracy = []
        self.rounds_loss = []
        self.local_accuracies = defaultdict(lambda: [])
        self.local_losses = defaultdict(lambda: [])
        self.plot_each_round = plot_each_round
        self.show_local = show_local
        self.show_global = show_global
        self.show_acc = show_acc
        self.show_loss = show_loss
        plt.interactive(False)

    class PlotParams:
        def __init__(self, y, title, x=None):
            self.title = title
            self.y = y
            self.x = x
            if self.x is None:
                step = 1 if len(y) < 20 else 5
                self.x = range(0, len(y), step)

    def show(self, params: typing.List[PlotParams], title=None):
        fig, axs = plt.subplots(len(params))
        axs = [axs] if len(params) == 1 else axs
        for ax, param in zip(axs, params):
            if len(param.y) > 0 and isinstance(param.y[0], list):
                for y1 in param.y:
                    ax.plot(y1)
            else:
                ax.plot(param.y)
            ax.set_title(param.title)
            ax.set_xticks(param.x)

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    def show_acc_loss(self, acc_y, acc_title, loss_y, loss_title, big_title):
        config = []
        if self.show_acc:
            config.append(self.PlotParams(acc_y, acc_title))
        if self.show_loss:
            config.append(self.PlotParams(loss_y, loss_title))
        self.show(config, big_title)

    def on_federated_ended(self, params):
        if self.show_global:
            self.show_acc_loss(self.rounds_accuracy, 'Accuracy', self.rounds_loss, 'Loss', 'Federated Results')

        if self.show_local:
            acc_y = [trainer_accuracies for trainer_id, trainer_accuracies in self.local_accuracies.items()]
            loss_y = [trainer_losses for trainer_id, trainer_losses in self.local_losses.items()]
            self.show_acc_loss(acc_y, 'Local Accuracy', loss_y, 'Local Loss', 'Local AccLoss')

    def show_acc_loss(self, acc_y, acc_title, loss_y, loss_title, big_title):
        config = []
        if self.show_acc:
            config.append(self.PlotParams(acc_y, acc_title))
        if self.show_loss:
            config.append(self.PlotParams(loss_y, loss_title))
        self.show(config, big_title)

    def on_federated_ended(self, params):
        if self.show_global:
            self.show_acc_loss(self.rounds_accuracy, 'Accuracy', self.rounds_loss, 'Loss', 'Federated Results')

        if self.show_local:
            acc_y = [trainer_accuracies for trainer_id, trainer_accuracies in self.local_accuracies.items()]
            loss_y = [trainer_losses for trainer_id, trainer_losses in self.local_losses.items()]
            self.show_acc_loss(acc_y, 'Local Accuracy', loss_y, 'Local Loss', 'Local AccLoss')

    def on_round_end(self, params):
        self.rounds_accuracy.append(params['accuracy'])
        self.rounds_loss.append(params['loss'])
        local_accuracy = params['local_acc']
        local_loss = params['local_loss']
        for trainer_id, accuracy in local_accuracy.items():
            self.local_accuracies[trainer_id].append(accuracy)
        for trainer_id, loss in local_loss.items():
            self.local_losses[trainer_id].append(loss)
        if self.plot_each_round:
            self.show_acc_loss(self.rounds_accuracy, f'Round Acc', self.rounds_loss, f'Round Loss', 'Per Round')


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
    def __init__(self, tag, folder_name="./logs/"):
        super().__init__()
        self.folder_name = folder_name
        self.file_name = "fedruns.pkl"
        self.tag = tag
        os.makedirs(folder_name, exist_ok=True)

    def force(self) -> []:
        return [Events.ET_FED_END]

    def on_federated_ended(self, params):
        context = params['context']
        all_runs = self.old_runs()
        if self.tag not in all_runs:
            all_runs[self.tag] = []
        all_runs[self.tag].append(context)
        with open(self.path(), 'wb') as file:
            try:
                pickle.dump(all_runs, file)
            except Exception as e:
                raise Exception(f'used federated model cannot be saved because: {e}')

    def is_old_exists(self):
        return os.path.exists(path=self.path())

    def old_runs(self):
        runs = {}
        if self.is_old_exists():
            with open(self.path(), 'rb') as file:
                try:
                    runs = pickle.load(file)
                except:
                    runs = {}
        return runs

    def path(self):
        return self.folder_name + self.file_name

    @staticmethod
    def unpack(file_path) -> typing.Dict[str, FederatedLearning.Context]:
        return pickle.load(open(file_path, 'rb'))


class WandbLogger(FederatedEventPlug):
    def __init__(self, config=None):
        super().__init__()
        import wandb
        wandb.login(key='24db2a5612aaf7311dd29a5178f252a1c0a351a9')
        wandb.init(project='localfed_ubuntu_test05', entity='mwazzeh', config=config)
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
    def __init__(self, tag, federated: FederatedLearning, verbose=1, flush=False):
        super().__init__()
        os.makedirs(manifest.ROOT_PATH + "/checkpoints", exist_ok=True)
        self.federated = federated
        self.file_name = manifest.ROOT_PATH + "/checkpoints" + "/run_" + tag + ".fed"
        self.verbose = verbose
        self.flush = flush

    def on_init(self, params):
        if os.path.exists(self.file_name) and not self.flush:
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


class ShowWeightDivergence(FederatedEventPlug):
    def __init__(self, show_log=True, include_global_weights=False):
        super().__init__()
        self.logger = logging.getLogger('weights_divergence')
        self.show_log = show_log
        self.include_global_weights = include_global_weights
        self.trainers_weights = None
        self.global_weights = None

    def on_training_end(self, params):
        self.trainers_weights = params['trainers_weights']

    def on_aggregation_end(self, params):
        self.global_weights = params['global_weights']

    def on_round_end(self, params):
        acc = params['accuracy']
        trainers_weights = self.trainers_weights
        if self.include_global_weights:
            trainers_weights[len(trainers_weights)] = self.global_weights
        ids = list(trainers_weights.keys())
        heatmap = np.zeros((len(trainers_weights), len(trainers_weights)))
        id_mapper = lambda id: ids.index(id)
        for trainer_id, weights in trainers_weights.items():
            for trainer_id_1, weights_1 in trainers_weights.items():
                keys = weights.keys()
                for key in keys:
                    if key == 'linear.weight':
                        heatmap[id_mapper(trainer_id)][id_mapper(trainer_id_1)] = torch.var(
                            torch.subtract(weights[key], weights_1[key]))
        self.show_map(heatmap, round(acc, 4))
        if self.show_log:
            self.logger.info(heatmap)

    def show_map(self, data, acc):
        fig, ax = plt.subplots()
        ax.imshow(data)
        ax.set_xticks(np.arange(len(data)))
        ax.set_yticks(np.arange(len(data)))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(data)):
            for j in range(len(data)):
                ax.text(j, i, '', ha="center", va="center", color="w")
        ax.set_title(f"Weight Divergence")
        ax.set_xlabel(f'Acc {acc}')
        fig.tight_layout()
        plt.show()
