import logging
import pickle

from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.cv.resnet import resnet56
from src import tools
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload

# client_data = data_loader.cifar10_10shards_100c_400min_400max()

# total images per class is 6000. 5% is 300. 80% for training is 240, that's 4% of the data.
# 5% of 1000 from mnist, that's 50 data sample, plus 10 for testing
from src.data.data_provider import PickleDataProvider

ratio = 0.15
labels_number = 3
ud = UniqueDistributor(labels_number, 500 * ratio, 500* ratio)
dataset_name = 'umdaa02_fd_filtered_cropped'

client_data = PickleDataProvider("../../../../datasets/pickles/umdaa02_fd_filtered_cropped.pkl").collect()
client_data = ud.distribute(client_data)
dataset_used = dataset_name + '_' + ud.id()


logging.basicConfig(level=logging.INFO)

train, test = client_data.reduce(lambdas.dict2dc).shuffle().as_tensor().split(0.8)

tools.detail(train)
tools.detail(test)

# learn_rates = [0.1, 0.01, 0.001, 0.0001]
learn_rates = [0.001]
epochs = 600
batch_size = 24
for learn_rate in learn_rates:
    model = resnet56(labels_number, 3, 128)
    model_name = 'resnet56()'

    trainer = TorchModel(model)
    trainer.train(train.batch(batch_size), lr=learn_rate, epochs=epochs)
    acc, loss = trainer.infer(test.batch(batch_size))
    acc = round(acc, 4)
    loss = round(loss, 4)
    file_name = 'warmup_' + dataset_used + '_' + model_name + '_lr_' + str(learn_rate) + '_e_' + str(epochs) + '_b_' + str(
        batch_size) + '_acc_' + str(acc) + '_loss_' + str(loss) + '.pkl'
    print(file_name, '\n', 'acc = ', acc, ' loss = ', loss)
    pickle.dump(model, open('umdaa02fd_pretrained_models\\' + file_name, 'wb'))
