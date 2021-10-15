import atexit
import json
import logging

from libs.model.collection import CNNCifar
from libs.model.cv.resnet import Cifar10, CNN_Cifar10
from src import tools, manifest
from src.apis import lambdas
from src.data.data_provider import PickleDataProvider
from apps.fed_ca.utilities.hp_generator import generate_configs, calculate_max_rounds
from libs.model.cv.cnn import CNN_OriginalFedAvg
import wandb
from src.manifest import WandbAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

wandb_project = 'localfed_ca01'
dataset_used = 'cifar10'

urls = json.load(open(manifest.DATA_PATH + "urls.json", 'r'))

data = PickleDataProvider(urls[dataset_used]).collect().map(lambdas.reshape((32, 32, 3))).map(
    lambdas.transpose((2, 0, 1))).shuffle().as_tensor().split(0.8)
train = data[0]
test = data[1]

# tools.detail(train)
# tools.detail(test)

# number of models that we are using
initial_models = {
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg(False),
    # 'CNNCifar(10)': CNNCifar(10)
    # model = Net()
    # model = CNN_DropOut()
    # model = CNN_OriginalFedAvg()
    # model = LogisticRegression(28*28, 10)
    # model = CNN_OriginalFedAvg(False)
    # model = resnet56(62, 1, 28)
    # 'Cifar10(10)': Cifar10()
    'CNN_Cifar10()': CNN_Cifar10()

}

# building Hyperparameters
input_shape = 32 * 32
labels_number = 10
percentage_nb_client = 10

for model_name, gen_model in initial_models.items():

    hyper_params = {'batch_size': [100], 'epochs': [1], 'num_rounds': [800], 'learn_rate': [0.001]}

    configs = generate_configs(model_param=gen_model, hyper_params=hyper_params)

    logger.info(calculate_max_rounds(hyper_params))
    for config in configs:
        batch_size = config['batch_size']
        epochs = config['epochs']
        num_rounds = config['num_rounds']
        initial_model = config['initial_model']
        learn_rate = config['learn_rate']

        print(
            f'Applied search: lr={learn_rate}, batch_size={batch_size}, epochs={epochs}, num_rounds={num_rounds}, '
            f'initial_model={initial_model} ')

        wandb.login(key=WandbAuth.key)
        wandb.init(project=wandb_project, entity=WandbAuth.entity, config={
            'lr': learn_rate, 'batch_size': batch_size,
            'epochs': epochs,
            'num_rounds': num_rounds, 'data_file': dataset_used,
            'model': model_name,
            'selected_clients': percentage_nb_client
        })

        wandb = wandb

        for i in range(num_rounds):
            tools.train(gen_model, train_data=train.batch(batch_size), epochs=epochs, lr=learn_rate)
            acc, loss = tools.infer(gen_model, test.batch(batch_size))
            print(acc, loss, i + 1)
            wandb.log({'acc': acc, 'loss': loss, 'last_round': i + 1})
            atexit.register(lambda: wandb.finish())
        wandb.finish()
