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
dist = UniqueDistributor(10, 380, 380)
dataset_name = 'cifar10'

client_data = preload(dataset_name, dist)
client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

logging.basicConfig(level=logging.INFO)

train, test = client_data.reduce(lambdas.dict2dc).shuffle().as_tensor().split(0.8)

tools.detail(train)
tools.detail(test)

model = CNN_Cifar10()
model_name = 'CNN_Cifar10()'

trainer = TorchModel(model)
# gave 0.57 acc with 600 epochs
# gave 0.57 acc with 1000 epochs
# gave 0.6 acc with 600 epochs 0.1 lr

learn_rate = 0.001
epochs = 1
batch_size = 100
trainer.train(train.batch(batch_size), lr=learn_rate, epochs=epochs)
acc, loss = trainer.infer(test.batch(batch_size))
print(acc, loss)
file_name = 'warmup_' + dist.id() + '_' + model_name + '_lr_' + str(learn_rate) + '_e_' + str(epochs) + '_b_' + str(batch_size) + '.pkl'
pickle.dump(model, open(file_name, 'wb'))
