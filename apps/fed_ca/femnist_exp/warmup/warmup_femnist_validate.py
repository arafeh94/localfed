import logging
import os
import pickle

from libs.model.cv.cnn import CNN_OriginalFedAvg
from src import tools
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload

# total images per class is 6000. 5% is 300. 80% for training is 240, that's 4% of the data.

dataset_name = 'femnist'
percentage_needed = 0.05

dist = UniqueDistributor(62, 3000, 3000)

client_data = preload(dataset_name, dist)
logging.basicConfig(level=logging.INFO)

file_name = f'./{dataset_name}_percentage_{percentage_needed}.pkl'
if os.path.exists(file_name):
    print("loading...")
    cache = pickle.load(open(file_name, 'rb'))
    train = cache['train']
    test = cache['test']
else:
    print("distributing...")
    train = client_data.map(lambda ci, dc: dc.shuffle(42).split(percentage_needed)[0]).reduce(
        lambdas.dict2dc).as_tensor()
    task_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(percentage_needed)[1])
    test = task_client_data.map(lambda ci, dc: dc.shuffle(42).split(percentage_needed * 0.2)[0]).reduce(
        lambdas.dict2dc).as_tensor()
    print('saving...')
    to_save = {'train': train, 'test': test}
    pickle.dump(to_save, open(file_name, 'wb'))
    print('finished')

print('Train data')
tools.detail(train)
print('Test data')
tools.detail(test)

learn_rates = [0.1, 0.01, 0.001, 0.0001]
epochs = 1
batch_size = 100
for learn_rate in learn_rates:
    model = CNN_OriginalFedAvg(False)
    model_name = 'CNN_OriginalFedAvg(False)'

    trainer = TorchModel(model)
    trainer.train(train.batch(batch_size), lr=learn_rate, epochs=epochs)
    acc, loss = trainer.infer(test.batch(batch_size))
    # acc_val, loss_eval = tools.infer(model, validate.batch(batch_size))
    acc = round(acc, 4)
    loss = round(loss, 4)
    # acc_val = round(acc_val, 4)
    file_name = 'warmup_percentage_' + str(percentage_needed) + '_' + model_name + '_lr_' + str(
        learn_rate) + '_e_' + str(epochs) + '_b_' + str(
        batch_size) + '_acc_' + str(acc) + '.pkl'

    print(file_name, '\n', 'acc = ', acc, ' loss = ', loss)
    pickle.dump(model, open('femnist_pretrained_models\\' + file_name, 'wb'))
    # models[str(learn_rate)] = model

# validate is useful in order to know if the model will do good on predicting results
# for lr, model in models.items():
#     print(tools.infer(model, validate.batch(batch_size)))
