import logging

from src import tools
from src.data.data_provider import PickleDataProvider


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = PickleDataProvider("../../../datasets/pickles/umdaa02_fd_filtered.pkl").collect()

tools.detail(client_data)

# Data shape
# x= [array([49152])]
# y = []  0 0 0 1 1 ...


