import copy
import logging
import os
import pickle

from src import manifest
from src.apis import checkpoints_utils
from src.federated.events import FederatedEventPlug
from src.federated.federated import FederatedLearning


class Resumable(FederatedEventPlug):
    def __init__(self, id: str, cp_file_path=manifest.CHECKPOINTS_PATH, save_ratio=50,
                 verbose=logging.INFO):
        super().__init__()
        self.verbose = verbose
        self.logger = logging.getLogger('resumable')
        self.save_ratio = save_ratio
        self.cp_file_path = cp_file_path
        self.id = id

    def on_init(self, params):
        if os.path.exists(self.cp_file_path):
            checkpoints = checkpoints_utils.read(self.cp_file_path)
            if self.id in checkpoints:
                self.log(f'found a checkpoint [{self.id}], loading...')
                loaded_context: FederatedLearning.Context = checkpoints[self.id]
                context: FederatedLearning.Context = params['context']
                federated = context.federated
                context.__dict__ = loaded_context.__dict__
                context.federated = federated
            del checkpoints
        else:
            self.init_file(self.cp_file_path)

    def init_file(self, cp_file_path):
        writer = open(self.cp_file_path, 'wb')
        pickle.dump({}, writer)
        writer.close()

    def _save(self, context):
        self.log('saving checkpoint...')
        cps = checkpoints_utils.read(self.cp_file_path)
        context_copy = self.context_copy(context)
        cps[self.id] = context_copy
        checkpoints_utils.write(cps)

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        round_id = context.round_id
        if round_id % self.save_ratio == 0:
            self._save(context)

    def on_federated_ended(self, params):
        context: FederatedLearning.Context = params['context']
        self._save(context)

    def log(self, msg):
        self.logger.log(self.verbose, msg)

    def context_copy(self, context):
        cp = FederatedLearning.Context(context.federated)
        cp.history = copy.deepcopy(context.history)
        cp.model = copy.deepcopy(context.model)
        cp.round_id = copy.deepcopy(context.round_id)
        cp.timestamp = copy.deepcopy(context.timestamp)
        del cp.federated
        return cp
