import atexit
import logging
from os import path

import wandb

from apps.fed_ca.umdaa002fd.pretrained.inception_resnet_v1 import InceptionResnetV1
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src import tools, manifest
from src.apis import lambdas
from src.data.data_distributor import UniqueDistributor
from src.data.data_provider import PickleDataProvider
from apps.fed_ca.utilities.hp_generator import generate_configs, build_random, calculate_max_rounds

from datetime import datetime

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

labels_number = 10
input_shape = 128 * 128

dataset_used = 'umdaa02_fd_filtered_cropped'
ud = UniqueDistributor(labels_number, 200, 200)
client_data = PickleDataProvider("../../../../datasets/pickles/umdaa02_fd_filtered_cropped.pkl").collect()
client_data = ud.distribute(client_data)
dataset_used = dataset_used + '_' + ud.id() + '_central'

client_data = client_data.map(lambdas.reshape((-1, 128, 128, 3))).map(lambdas.transpose((0, 3, 1, 2)))


generated_filename = "../../../../datasets/pickles/"+ dataset_used +".pkl"
if(path.exists(generated_filename) == False):
    PickleDataProvider.save(client_data, generated_filename)



train, test = PickleDataProvider(generated_filename).collect().reduce(lambdas.dict2dc).shuffle(47).as_tensor().split(
    0.8)

# train, test = PickleDataProvider(generated_filename).collect().shuffle(47).as_tensor().split(
#     0.8)


tools.detail(train)
# tools.detail(test)

vggface2 = InceptionResnetV1(pretrained='vggface2', num_classes=labels_number, classify=True, device='cuda').eval()

for param in list(vggface2.children()):
    param.requires_grad = False
for param in list(vggface2.children())[-5:]:
    param.requires_grad = True

# number of models that we are using
initial_models = {
    # 'CNN_OriginalFedAvg': CNN_OriginalFedAvg(),
    # 'LogisticsRegression': LogisticRegression(28 * 28, 10),
    # 'resnet56': resnet56(labels_number, 3, 128)
    'VGGFace2': vggface2

}

# building Hyperparameters
percentage_nb_client = labels_number

for model_name, gen_model in initial_models.items():
    # learn rate of 0.0001 is the best for umdaa02_filtered central
    hyper_params = {'batch_size': [24], 'epochs': [1], 'num_rounds': [200], 'learn_rate': [0.001]}

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
        wandb.init(project='umdaa-02-fd-filtered-cropped', entity=manifest.wandb_config['entity'], config={
            'lr': learn_rate, 'batch_size': batch_size,
            'epochs': epochs,
            'num_rounds': num_rounds, 'data_file': dataset_used,
            'model': model_name,
            'selected_clients': percentage_nb_client
        })

        for i in range(num_rounds):
            tools.train(gen_model, train_data=train.batch(batch_size), epochs=epochs, lr=learn_rate)
            acc, loss = tools.infer(gen_model, test.batch(batch_size))
            print(acc, loss, i + 1)
            wandb.log({'acc': acc, 'loss': loss, 'last_round': i + 1})
            atexit.register(lambda: wandb.finish())
        wandb.finish()

        end_time = datetime.now()
        print('Total Duration: {}'.format(end_time - start_time))
