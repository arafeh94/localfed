import copy
import logging
import pickle
import sys

from libs.model.linear.lr import LogisticRegression
from src.apis.extensions import TorchModel
from src.data.data_provider import PickleDataProvider
from src.federated.components.trainers import TorchTrainer
from src.manifest import dataset_urls

logging.basicConfig(level=logging.INFO)
train, test = PickleDataProvider(dataset_urls('mnist')).collect().shuffle().as_tensor().split(0.8)

model = LogisticRegression(28 * 28, 10)

trainer = TorchModel(model)
trainer.train(train.batch(50))
acc, loss = trainer.infer(test.batch(50))

pickle.dump(model, open("./model.pkl", 'wb'))

