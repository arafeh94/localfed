import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
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
# pickle.dump(loaded_model, open(f'kmeans_{n_clusters}.pkl', "wb"))
print(loaded_model.labels_)
# load the model and predict
# loaded_model = pickle.load(open(f'kmeans_{n_clusters}.pkl', "rb"))
label = loaded_model.predict(test)

#Calculates the standard deviation of each column of the embeddings/PCA
def standard_dev(PCA):
    stds = []
    for i in range(np.shape(PCA)[1]):
        column = PCA[:,i]
        s = np.std(column)
        stds.append(s)
    return stds



pca = PCA(2)
# Transform the data
df = pca.fit_transform(test)

# GMM using gaussian filter for the test data of the known users
stds = standard_dev(df)

#Creating gauss_list, containing the columns of the PCA filtered one by one by a Gaussian filter
gauss_list = []
for k in range(len(stds)):
    gauss_list.append(gaussian_filter(df[:, k],stds[k]/100000))

df = np.array(gauss_list).T
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
