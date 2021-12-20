from src.apis.mpi import Comm
from src.federated.events import FederatedEventPlug


class MPIStop(FederatedEventPlug):
    def __init__(self, comm: Comm):
        super().__init__()
        self.comm = comm

    def on_federated_ended(self, params):
        self.comm.stop()
