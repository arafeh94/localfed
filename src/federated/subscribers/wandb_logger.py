import atexit

from src.federated.events import FederatedSubscriber
from src.manifest import wandb_config


class WandbLogger(FederatedSubscriber):
    def __init__(self, project, config=None, resume=False, id: str = None):
        super().__init__()
        import wandb
        wandb.login(key=wandb_config['key'])
        self.wandb = wandb
        self.project = project
        self.config = config
        self.id = id
        self.resume = resume
        atexit.register(lambda: self.wandb.finish())

    def on_init(self, params):
        if self.resume:
            self.wandb.init(project=self.project, entity=wandb_config['entity'], config=self.config,
                            id=self.id, resume="allow")
        else:
            self.wandb.init(project=self.project, entity=wandb_config['entity'], config=self.config)

    def on_round_end(self, params):
        self.wandb.log({'acc': params['accuracy'], 'loss': params['loss']})

    def on_federated_ended(self, params):
        self.wandb.finish()
