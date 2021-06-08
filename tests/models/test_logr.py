import logging

import libs.model.collection
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data.data_generator import load
from src.data.data_provider import LocalMnistDataProvider, PickleDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../datasets/pickles/100_10_400_mnist.pkl'
logger.info('Generating Data --Started')
dp = PickleDataProvider('../../datasets/pickles/mnist.pkl').collect()
logger.info('Shuffling')
dp = dp.shuffle().as_tensor()
logger.info('Splitting')
x, y = dp.split(0.8)
logger.info('Generating Data --Ended')

lr = LogisticRegression(28 * 28, 10)
for i in range(20):
    tools.train(lr, x.batch(64))
    acc, loss = tools.infer(lr, y.batch(64))
    print(acc)
    print(loss)
