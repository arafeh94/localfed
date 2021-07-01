import copy
import logging
import pickle
import sys

from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data import data_loader, data_generator
from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider
from src.federated.components.trainers import TorchTrainer
from src.manifest import dataset_urls

logging.basicConfig(level=logging.INFO)
clients_data = data_generator.load("../../../datasets/pickles/femnist_62c_200mn_200mx.pkl").distributed
train, test = clients_data.reduce(lambdas.dict2dc).shuffle().as_tensor().split(0.8)

model = CNN_OriginalFedAvg(False)

trainer = TorchModel(model)
trainer.train(train.batch(64), lr=0.1, epochs=150)
acc, loss = trainer.infer(test.batch(64))
print(acc, loss)
pickle.dump(model, open("warmup_model.pkl", 'wb'))
