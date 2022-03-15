import logging
import pickle
from datetime import datetime

from torch import nn

from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.resnet import resnet56
from src.data.data_distributor import UniqueDistributor
from src.data.data_provider import PickleDataProvider
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.federated import Events, FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.wandb_logger import WandbLogger

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

dataset = 'umdaa02_fd_filtered_cropped'
# total number of clients from umdaa02-fd is 44
labels_number = 3
ud = UniqueDistributor(labels_number, 500, 500)
client_data = PickleDataProvider("../../../../datasets/pickles/umdaa02_fd_filtered_cropped.pkl").collect()
# tools.detail(client_data)
client_data = ud.distribute(client_data)
dataset_used = dataset + '_' + ud.id()


def load_warmup():
    model = pickle.load(open(
        "umdaa02fd_pretrained_models\warmup_umdaa02_fd_filtered_cropped_unique_3c_75.0mn_75.0mx_resnet56()_lr_0.001_e_2_b_24_acc_0.2889_loss_1.1095.pkl",
        'rb'))
    return model


# building Hyperparameters
input_shape = 128 * 128
percentage_nb_client = labels_number

# number of models that we are using
initial_models = {
    # 'LR': LogisticRegression(input_shape, labels_number),
    # 'MLP': MLP(input_shape, labels_number)
    #  'CNN': CNN_DropOut(False),
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg(False)
    'resnet56': resnet56(labels_number, 3, 128)
}

# runs = {}
for model_name, gen_model in initial_models.items():

    hyper_params = {'batch_size': [24], 'epochs': [1], 'num_rounds': [2]}
    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = 0.001

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds},'
            f' initial_model={model_name} ')
        trainer_manager = SeqTrainerManager()
        trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=batch_size,
                                       epochs=epochs, optimizer='sgd', criterion='cel', lr=learn_rate)

        federated = FederatedLearning(
            trainer_manager=trainer_manager,
            trainer_config=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=batch_size, criterion=nn.CrossEntropyLoss()),
            # client_selector=client_selectors.All(),
            client_selector=client_selectors.Random(percentage_nb_client),
            trainers_data_dict=client_data,
            initial_model=load_warmup,
            num_rounds=num_rounds,
            desired_accuracy=1
        )

        # federated.add_subscriber(subscribers.Resumable('warmup_femnist200', federated, flush=False))

        federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

        federated.add_subscriber(WandbLogger(config={'lr': learn_rate, 'batch_size': batch_size, 'epochs': epochs,
                                                     'num_rounds': num_rounds, 'data_file': dataset_used,
                                                     'model': model_name + '',
                                                     'selected_clients': percentage_nb_client}))

        logger.info("----------------------")
        logger.info("start federated")
        logger.info("----------------------")
        federated.start()
        # runs[model_name] = federated.context
        end_time = datetime.now()
        print('Total Duration: {}'.format(end_time - start_time))

# r = fedruns.FedRuns(runs)
# r.plot()
end_time = datetime.now()
print('Total Duration: {}'.format(end_time - start_time))
