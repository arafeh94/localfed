import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.apis import lambdas
from src.data.data_container import DataContainer
from src.data.data_distributor import UniqueDistributor
from src.data.data_provider import PickleDataProvider

file_path = "../../../datasets/pickles/umdaa02touch.pkl"
labels_number = 48
client_data = PickleDataProvider(file_path).collect()


xs = []
ys = []

for _ , client in client_data.items():
    xs.append(client.x)
    ys.append(client.y)


n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xs)



print('')