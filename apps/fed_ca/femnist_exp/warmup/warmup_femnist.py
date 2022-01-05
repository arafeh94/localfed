import logging
import logging
import pickle

from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data import data_generator

logging.basicConfig(level=logging.INFO)
clients_data = data_generator.load("../../../datasets/pickles/femnist_62c_200mn_200mx.pkl").distributed
train, test = clients_data.reduce(lambdas.dict2dc).shuffle().as_tensor().split(0.8)

model = CNN_OriginalFedAvg(False)

trainer = TorchModel(model)
trainer.train(train.batch(10), lr=0.1, epochs=600)
acc, loss = trainer.infer(test.batch(10))
print(acc, loss)
pickle.dump(model, open("warmup_model_femnist200.pkl", 'wb'))
