import logging
from collections import defaultdict

from libs.model.cv.cnn import CNN_OriginalFedAvg
from src import tools
from apps.genetic_selectors.context import Context
import src.data.data_generator as dgg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
dg = dgg.load('../../datasets/pickles/mnist_2shards_70c_600mn_600mx.pkl')
client_data = tools.dict_select(range(30), dg.distributed)
dg.describe(range(30))
context = Context(client_data, lambda: CNN_OriginalFedAvg())
context.build(0.1)
clustered = context.cluster(10)
logger.info(clustered)
data = defaultdict(lambda: [])
for client_id, cluster in clustered.items():
    data[cluster].append(client_id)
logging.info(data)

for cluster in data:
    logger.info(f"client of cluster {cluster}")
    dg.describe(data[cluster])
