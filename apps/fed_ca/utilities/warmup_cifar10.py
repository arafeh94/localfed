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
dist = UniqueDistributor(10, 300, 300)
dataset_name = 'cifar10'

client_data = preload(dataset_name, dist)
client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))

logging.basicConfig(level=logging.INFO)

train, test = client_data.reduce(lambdas.dict2dc).shuffle().as_tensor().split(0.8)

tools.detail(train)
tools.detail(test)

model = CNN_Cifar10()

trainer = TorchModel(model)
# gave 0.57 acc with 600 epochs
trainer.train(train.batch(100), lr=0.01, epochs=600)
acc, loss = trainer.infer(test.batch(100))
print(acc, loss)
pickle.dump(model, open("warmup_model_cifar10_240.pkl", 'wb'))
