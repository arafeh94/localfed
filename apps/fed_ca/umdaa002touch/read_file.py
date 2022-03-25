import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data.data_container import DataContainer
from src.data.data_distributor import UniqueDistributor
from src.data.data_provider import PickleDataProvider

file_path = "../../../datasets/pickles/umdaa02touch.pkl"
labels_number = 48
client_data = PickleDataProvider(file_path).collect()

xs = []
for i in range(labels_number):
    xs.extend(client_data[i].x)

xs = DataContainer(xs, range(len(xs))).shuffle(47).x
# 70% train, 10% validate, 20% test
train, validate, test = np.split(xs, [int(len(xs) * 0.7), int(len(xs) * 0.99)])

print(len(test))
# create a k-means model with n_clusters
n_clusters = 8
loaded_model = KMeans(n_clusters=n_clusters, random_state=0).fit(test)
pickle.dump(loaded_model, open(f'kmeans_{n_clusters}.pkl', "wb"))
print(loaded_model.labels_)
# load the model and predict
# loaded_model = pickle.load(open(f'kmeans_{n_clusters}.pkl', "rb"))
label = loaded_model.predict(test)

pca = PCA(2)
# Transform the data
df = pca.fit_transform(test)
# Getting the Centroids
centroids = loaded_model.cluster_centers_
cen = PCA(2).fit_transform(centroids)
# Getting unique labels
u_labels = np.unique(label)



# plotting the results:
for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i, s=1)
    # plt.scatter(cen[:, 0], cen[:, 1], s=20, color='k')
plt.legend()
plt.show()
