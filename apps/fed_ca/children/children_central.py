import atexit
import logging

import wandb

from apps.fed_ca.children import visualizations
from libs.model.cv.cnn import ConvNet1D_test
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src import tools, manifest
from src.apis import lambdas, federated_tools
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds

from datetime import datetime

from src.data.data_provider import PickleDataProvider

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

labels_number = 118

# dataset_used = 'children_touch_3_3_f'
dataset_used = 'children_touch_all_3_3_f'
client_data = preload(dataset_used)
# client_data = client_data.reduce(lambdas.dict2dc)
# visualizations.visualize(client_data)
# ud = LabelDistributor(labels_number, 1, 500, 500)
# client_data = ud.distribute(client_data)
# visualizations.visualize(client_data)

train_data, test_data = client_data.reduce(lambdas.dict2dc).shuffle(128).as_tensor().split(0.75)
#
# train_data = all_data.map(lambdas.dc_split(0.8, 0))
# test_data = all_data.map(lambdas.dc_split(0.8, 1))

# tools.detail(train_data)
# tools.detail(test_data)

# number of models that we are using
initial_models = {
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg(),
    # 'LogisticsRegression': LogisticRegression(3, labels_number),
    'resnet56': resnet56(labels_number, 1, 3)
    # 'ConvNet1D_test': ConvNet1D_test(labels_number)

}

# building Hyperparameters
percentage_nb_client = 1

for model_name, gen_model in initial_models.items():
    hyper_params = {'batch_size': [256], 'epochs': [3], 'num_rounds': [100], 'learn_rate': [0.01]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = config['learn_rate']

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}')

        wandb.login(key=manifest.wandb_config['key'])
        wandb.init(project='children', entity=manifest.wandb_config['entity'], config={
            'lr': learn_rate, 'batch_size': batch_size,
            'epochs': epochs,
            'num_rounds': num_rounds, 'data_file': dataset_used + '_central',
            'model': model_name,
            'selected_clients': percentage_nb_client
        })

        for i in range(num_rounds):
            federated_tools.train(gen_model, train_data=train_data.batch(batch_size), epochs=epochs, lr=learn_rate)
            acc, loss = federated_tools.infer(gen_model, test_data.batch(batch_size))
            print(acc, loss, i + 1)
            wandb.log({'acc': acc, 'loss': loss, 'last_round': i + 1})
            atexit.register(lambda: wandb.finish())
        wandb.finish()

        end_time = datetime.now()
        print('Total Duration: {}'.format(end_time - start_time))
