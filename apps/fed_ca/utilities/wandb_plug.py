import wandb as wandb

from src.federated.subscribers import WandbLogger


class FedCAWandb(WandbLogger):
    def __init__(self, config=None):
        super().__init__(config)
        wandb.login(key='24db2a5612aaf7311dd29a5178f252a1c0a351a9')
        wandb.init(project='localfed_ubuntu_test05', entity='mwazzeh', config=config)
