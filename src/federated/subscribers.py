import atexit
import logging
import math
import os
import pickle
import re
import statistics
import time
import typing
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import wasserstein_distance

from src import tools, manifest
from src.apis import plots, files
from src.apis.extensions import Dict
from src.apis.mpi import Comm
from src.data.data_container import DataContainer
from src.federated.events import FederatedEventPlug, Events
from src.federated.federated import FederatedLearning
from src.manifest import WandbAuth


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

    def on_model_status_eval(self, params):
        params = tools.Dict.but(['context', 'trained_model'], params)
        self.logger.info(f"model status: {params}")

    def force(self) -> []:
        return [Events.ET_FED_START, Events.ET_MODEL_STATUS]

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


class FedSave(FederatedEventPlug):
    def __init__(self, tag=None, folder_name=manifest.COMPARE_PATH):
        super().__init__()
        self.folder_name = folder_name
        self.file_name = "fedruns.pkl"
        self.tag = tag
        os.makedirs(folder_name, exist_ok=True)

    def on_init(self, params):
        context: FederatedLearning.Context = params['context']
        self.tag = self.tag or 'sv_' + context.id

    def force(self) -> []:
        return [Events.ET_FED_END]

    def on_federated_ended(self, params):
        context: FederatedLearning.Context = params['context']
        all_runs = self.old_runs()
        all_runs[self.tag] = context
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
    def unpack(file_path=None) -> typing.Dict[str, FederatedLearning.Context]:
        if file_path is None:
            file_path = manifest.COMPARE_PATH + 'fedruns.pkl'
        return pickle.load(open(file_path, 'rb'))


class WandbLogger(FederatedEventPlug):
    def __init__(self, config=None, resume=False, id: str = None):
        super().__init__()
        import wandb
        wandb.login(key=WandbAuth.key)
        self.wandb = wandb
        self.config = config
        self.id = id
        self.resume = resume
        atexit.register(lambda: self.wandb.finish())

    def on_init(self, params):
        if self.resume:
            self.wandb.init(project=WandbAuth.project, entity=WandbAuth.entity, config=self.config, id=self.id,
                            resume="allow")
        else:
            self.wandb.init(project=WandbAuth.project, entity=WandbAuth.entity, config=self.config)

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        self.wandb.log({'acc': params['accuracy'], 'loss': params['loss'], 'last_round': params['round'] + 1},
                       step=context.round_id)

    def on_federated_ended(self, params):
        self.wandb.finish()


class MPIStopPlug(FederatedEventPlug):

    def on_federated_ended(self, params):
        Comm().stop()


# noinspection PyUnresolvedReferences
class Resumable(FederatedEventPlug):
    def __init__(self, federated: FederatedLearning, distributor: 'Distributor' = None, tag='',
                 save_path=manifest.ROOT_PATH, ignore_rounds=True, save_each=50, verbose=logging.INFO):
        super().__init__()
        os.makedirs(manifest.ROOT_PATH + "/checkpoints", exist_ok=True)
        self.federated = federated
        self.verbose = verbose
        self.logger = logging.getLogger('resumable')
        self.save_each = save_each
        self.file_path = f"{save_path}/checkpoints.fed"
        self.ignore_rounds = ignore_rounds
        self.tag = f'{tag}_' if len(tag) > 0 else ''
        if distributor:
            self.tag = self.tag + distributor.id() + '_'

    def on_init(self, params):
        context: FederatedLearning.Context = params['context']
        run_title: str = context.id
        if self.ignore_rounds:
            run_title = re.sub('_([0-9]*?r)_', '_', run_title)
        self.tag = f'{self.tag}{run_title}'
        if os.path.exists(self.file_path):
            file = open(self.file_path, 'rb')
            checkpoints = pickle.load(file)
            if self.tag in checkpoints:
                self.log(f'found a checkpoint [{self.tag}], loading...')
                loaded_context: FederatedLearning.Context = checkpoints[self.tag]
                loaded_context.num_rounds = context.num_rounds
                self.federated.context = loaded_context
            file.close()
            del checkpoints
        else:
            writer = open(self.file_path, 'wb')
            pickle.dump({}, writer)
            writer.close()

    def _save(self, context):
        self.log('saving checkpoint...')
        reader = open(self.file_path, 'rb')
        checkpoints = pickle.load(reader)
        checkpoints[self.tag] = context
        reader.close()
        writer = open(self.file_path, 'wb')
        pickle.dump(checkpoints, writer)
        writer.close()
        del checkpoints

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        round_id = context.round_id
        if round_id % self.save_each == 0:
            self._save(context)

    def on_federated_ended(self, params):
        context: FederatedLearning.Context = params['context']
        self._save(context)

    def log(self, msg):
        self.logger.log(self.verbose, msg)


class ShowDataDistribution(FederatedEventPlug):
    def __init__(self, label_count, per_round=False, save_dir=None):
        super().__init__()
        self.logger = logging.getLogger('data_distribution')
        self.label_count = label_count
        self.per_round = per_round
        self.save_dir = save_dir
        self.round_id = -1
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def on_federated_started(self, params):
        clients_data: Dict[int, DataContainer] = params['trainers_data_dict']
        self.plot(clients_data)

    def on_training_start(self, params):
        self.round_id = params['context'].round_id
        if self.per_round:
            clients_data = params['trainers_data']
            self.plot(clients_data)

    def plot(self, clients_data):
        tick = time.time()
        self.logger.info('building data distribution...')
        ids = list(clients_data.keys())
        id_mapper = lambda id: ids.index(id)

        client_label_count = np.zeros((len(clients_data), self.label_count))
        for client_id, data in clients_data.items():
            for y in data.y:
                client_label_count[id_mapper(client_id)][y] += 1
        save_dir = f"./{self.save_dir}/round_{self.round_id}_dd.png" if self.save_dir is not None else None
        plots.heatmap(client_label_count, 'Clients Data Distribution', 'x:Client - y:Class', save_dir)
        self.logger.info(f'building data distribution finished {time.time() - tick}')


class ShowWeightDivergence(FederatedEventPlug):
    def __init__(self, show_log=False, include_global_weights=False, save_dir=None, plot_type='matrix'):
        """
        plot_type = matrix | linear
        Returns:
            object: FederatedEventPlug
        """
        super().__init__()
        self.logger = logging.getLogger('weights_divergence')
        self.show_log = show_log
        self.include_global_weights = include_global_weights
        self.trainers_weights = None
        self.global_weights = None
        self.save_dir = save_dir
        self.round_id = 0
        self.plot_type = plot_type
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_training_end(self, params):
        self.trainers_weights = params['trainers_weights']

    def on_aggregation_end(self, params):
        self.global_weights = params['global_weights']

    def on_round_end(self, params):
        tick = time.time()
        self.logger.info('building weights divergence...')
        self.round_id = params['context'].round_id
        save_dir = f"./{self.save_dir}/round_{self.round_id}_wd.png" if self.save_dir is not None else None
        acc = params['accuracy']
        trainers_weights = self.trainers_weights
        if self.include_global_weights:
            trainers_weights[len(trainers_weights)] = self.global_weights
        ids = list(trainers_weights.keys())
        self.logger.info(f'building weights divergence finished {time.time() - tick}')
        if self.plot_type == 'matrix':
            id_mapper = lambda id: ids.index(id)
            heatmap = np.zeros((len(trainers_weights), len(trainers_weights)))
            for trainer_id, weights in trainers_weights.items():
                for trainer_id_1, weights_1 in trainers_weights.items():
                    w0 = tools.flatten_weights(weights)
                    w1 = tools.flatten_weights(weights_1)
                    heatmap[id_mapper(trainer_id)][id_mapper(trainer_id_1)] = np.var(np.subtract(w0, w1))
            plots.heatmap(heatmap, 'Weight Divergence', f'Acc {round(acc, 4)}', save_dir)
            if self.show_log:
                self.logger.info(heatmap)
        elif self.plot_type == 'linear':
            weight_dict = defaultdict(lambda: [])
            for trainer_id, weights in trainers_weights.items():
                weights = tools.flatten_weights(weights)
                weights = tools.compress(weights, 10, 1)
                weight_dict[trainer_id] = weights
            plots.linear(weight_dict, "Model's Weights", f'R: {self.round_id}', save_dir)
        else:
            raise Exception('plot type should be a string with a value either "linear" or "matrix"')


class ShowAvgWeightDivergence(FederatedEventPlug):
    def __init__(self, save_dir=None, divergence_tag=None, plot_each_round=False):
        super().__init__()
        self.logger = logging.getLogger('weights_divergence')
        self.trainers_weights = None
        self.global_weights = None
        self.save_dir = save_dir
        self.round_id = 0
        self.all_wd = []
        self.divergence_tag = divergence_tag
        self.plot_each_round = plot_each_round
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_init(self, params):
        context: FederatedLearning.Context = params['context']
        self.divergence_tag = self.divergence_tag or 'div_' + context.id

    def on_training_end(self, params):
        self.trainers_weights = params['trainers_weights']

    def on_aggregation_end(self, params):
        self.global_weights = params['global_weights']

    def on_round_end(self, params):
        trainers_weights = self.trainers_weights
        global_model_dict = params['context'].model.state_dict()
        save_dir = f"./{self.save_dir}/round_{self.round_id}_wd.png" if self.save_dir is not None else None
        avg_weight_divergence = self._get_average_weight_divergence(global_model_dict, trainers_weights)
        self.all_wd.append(avg_weight_divergence)
        if self.plot_each_round:
            plt.plot(self.all_wd)
            plt.show()

    def on_federated_ended(self, params):
        plt.plot(self.all_wd)
        if self.divergence_tag is not None:
            files.divergences.save_divergence(self.all_wd, self.divergence_tag)
        if self.save_dir is not None:
            save_dir = f"./{self.save_dir}/avg_wd.png" if self.save_dir is not None else None
            plt.savefig(save_dir)
        plt.show()

    def _get_average_weight_divergence(self, global_model_dict, trainers_weights):
        all_results = []
        flattened_global_weights = tools.flatten_weights(global_model_dict)
        for trained_id, trainers_weight in trainers_weights.items():
            flattened_trainer_weights = tools.flatten_weights(trainers_weight)
            result = wasserstein_distance(flattened_global_weights, flattened_trainer_weights)
            all_results.append(result)
        final_result = sum(all_results) / len(all_results)
        return final_result
