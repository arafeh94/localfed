import logging
from collections import defaultdict

from sklearn import decomposition

from libs.model.cv.cnn import CNN_OriginalFedAvg
from libs.model.linear.lr import LogisticRegression
from src import tools
from apps.genetic_selectors.src.context import Context
import src.data.data_generator as dgg
from src.data.data_provider import PickleDataProvider


def compress(weights):
    weights = weights.reshape(10, -1)
    pca = decomposition.PCA(n_components=4)
    pca.fit(weights)
    weights = pca.transform(weights)
    return weights.flatten()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
clients_data = dgg.DataGenerator(PickleDataProvider('../../datasets/pickles/mnist10k.pkl')) \
    .distribute_shards(30, 1, 100, 100)
tools.detail(clients_data)
context = Context(clients_data, lambda: LogisticRegression(28 * 28, 10))
context.train(0.1)
clustered = context.cluster(10)
logger.info(clustered)
data = defaultdict(lambda: [])
for client_id, cluster in clustered.items():
    data[cluster].append(client_id)
logging.info(data)
