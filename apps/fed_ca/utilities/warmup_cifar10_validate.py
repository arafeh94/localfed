import copy
import logging
import pickle
import sys

from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import CNN_Cifar10
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload
from src.data.data_provider import PickleDataProvider
from src.federated.components.trainers import TorchTrainer
from src.manifest import dataset_urls

# client_data = data_loader.cifar10_10shards_100c_400min_400max()

# total images per class is 6000. 5% is 300. 80% for training is 240, that's 4% of the data.
dist = UniqueDistributor(10, 480, 480)
dataset_name = 'cifar10'

client_data = preload(dataset_name, dist)
client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

logging.basicConfig(level=logging.INFO)

# train, test = client_data.reduce(lambdas.dict2dc).shuffle().as_tensor().split(0.8)

client_data = client_data.reduce(lambdas.dict2dc).shuffle().as_tensor()
validate, client_data = client_data.split(0.2)
train, test = client_data.split(0.8)

tools.detail(train)
tools.detail(test)

models = {}

learn_rates = [0.1, 0.01, 0.001, 0.0001]
epochs = 600
batch_size = 100
for learn_rate in learn_rates:
    model = CNN_Cifar10()
    model_name = 'CNN_Cifar10()'

    trainer = TorchModel(model)
    trainer.train(train.batch(batch_size), lr=learn_rate, epochs=epochs)
    acc, loss = trainer.infer(test.batch(batch_size))
    acc = round(acc, 4)
    loss = round(loss, 4)
    file_name = 'warmup_' + dist.id() + '_' + model_name + '_lr_' + str(learn_rate) + '_e_' + str(epochs) + '_b_' + str(
        batch_size) + '_acc_' + str(acc) + '.pkl'
    print(file_name, '\n', 'acc = ', acc, ' loss = ', loss)
    print('Validating results = ', tools.infer(model, validate.batch(batch_size)))
    # pickle.dump(model, open('cifar10_pretrained_models\\' + file_name, 'wb'))
    # models[str(learn_rate)] = model


# validate is useful in order to know if the model will do good on predicting results
# for lr, model in models.items():
#     print(tools.infer(model, validate.batch(batch_size)))
