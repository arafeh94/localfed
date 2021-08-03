import time

import numpy as np
from matplotlib import pyplot as plt

from src.apis import lambdas
from src.data import data_loader
from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider
from src.manifest import dataset_urls

t = time.time()


def tick(msg=''):
    global t
    n = time.time()
    print(msg, n - t)
    t = n


dt = PickleDataProvider(dataset_urls('mnist')).collect()

dt = dt.map(lambdas.reshape((28, 28)))
tick('reshape')

dt = dt.shuffle()
tick('shuffle')

dt = dt.as_tensor()
tick('as tensor')

dt = dt.as_list()
tick('as list')

dt = dt.as_tensor()
tick('as tensor')
