import atexit

from src.app import session
from src.app.federated_app import FederatedApp
from src.app.session import Session
from src.federated.events import FederatedEventPlug
from src.federated.federated import FederatedLearning
from src.manifest import WandbAuth


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
        self.wandb.log({'acc': params['accuracy'], 'loss': params['loss']})

    def on_federated_ended(self, params):
        self.wandb.finish()


