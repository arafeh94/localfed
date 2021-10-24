import logging

from src import tools
from src.federated.events import Events, FederatedEventPlug


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
        self.logger.info(f"selected clients {params}")
        if self.detailed_selection:
            tools.detail(self.trainers_data_dict, params['trainers_ids'])
