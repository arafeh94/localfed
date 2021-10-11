import atexit
import logging

import torch

import wandb
from datasets.pickles.status import urls
from libs.model.collection import CNNCifar

from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import Cifar10
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.data.data_provider import PickleDataProvider
from src.manifest import WandbAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')

data = PickleDataProvider(urls['cifar10']).collect().map(lambdas.reshape((32, 32, 3))).map(
    lambdas.transpose((2, 0, 1))).shuffle(47).as_tensor().split(0.8)
train = data[0]
test = data[1]

# train, test = PickleDataProvider("../../datasets/pickles/cifar10.pkl").collect().shuffle().as_tensor().split(0.8)
# train = train.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
# test = test.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

tools.detail(train)
tools.detail(test)

# model = Net()
# model = CNN_DropOut()
# model = CNN_OriginalFedAvg()
# model = LogisticRegression(28*28, 10)
# model = CNN_OriginalFedAvg(False)
# model = resnet56(62, 1, 28)
model = CNNCifar(10)
model = Cifar10()

print(model)

batch_size = 10
epochs = 1
num_rounds = 800
model_name = 'Cifar10'
percentage_nb_client = 10
dataset_used = 'cifar-10'
learn_rate = 0.001

wandb.login(key=WandbAuth.key)
wandb.init(project=WandbAuth.project, entity=WandbAuth.entity, config={
    'lr': learn_rate, 'batch_size': batch_size,
    'epochs': epochs,
    'num_rounds': num_rounds, 'data_file': dataset_used,
    'model': model_name,
    'selected_clients': percentage_nb_client
})
wandb = wandb

# PATH = 'D:\\Github\\my_repository\\localfed\\tests\\test_model.pth'




for i in range(num_rounds):
    # if i != 0:
    #     # model = CNN_OriginalFedAvg()
    #     model.load_state_dict(torch.load(PATH), strict=True)

    tools.train(model, train_data=train.batch(batch_size), epochs=epochs, lr=learn_rate)

    acc, loss = tools.infer(model, test.batch(batch_size))

    print(acc, loss, i + 1)

    # torch.save(model.state_dict(), PATH)

    wandb.log({'acc': acc, 'loss': loss, 'last_round': i + 1})
    atexit.register(lambda: wandb.finish())

wandb.finish()
