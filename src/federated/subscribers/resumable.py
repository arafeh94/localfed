import copy
import logging
from src.apis.rw import IODict
from src.federated.events import FederatedEventPlug
from src.federated.federated import FederatedLearning


class Resumable(FederatedEventPlug):
    def __init__(self, io: IODict, save_ratio=5, verbose=logging.INFO):
        super().__init__()
        self.verbose = verbose
        self.logger = logging.getLogger('resumable')
        self.save_ratio = save_ratio
        self.io = io

    def on_init(self, params):
        loaded_context: FederatedLearning.Context = self.io.read('context', absent_ok=True)
        if loaded_context:
            context: FederatedLearning.Context = params['context']
            context.__dict__.update(loaded_context.__dict__)

    def _save(self, context):
        self.log('saving checkpoint...')
        context_copy = self.context_copy(context)
        self.io.write('context', context_copy)

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
