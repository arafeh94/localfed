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

for _, client in client_data.items():
    for data in client:
        xs.append(data.x)
        ys.append(data.y)

n_clusters = 8


# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xs)
# pickle.dump(kmeans, open(f'kmeans_{n_clusters}.pkl', "wb"))

kmeans = pickle.load(open(f'kmeans_{n_clusters}.pkl', "rb"))
label = kmeans.predict(xs)

pca = PCA(2)
df = pca.fit_transform(xs)
centroids = kmeans.cluster_centers_
cen = PCA(2).fit_transform(centroids)
u_labels = np.unique(label)


for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i, s=1)
    plt.scatter(cen[:, 0], cen[:, 1], s=20, color='k')
plt.legend()
plt.show()


print('')
