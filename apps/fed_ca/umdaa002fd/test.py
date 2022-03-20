

import pickle

from src.data.data_loader import preload
from src.data.data_provider import PickleDataProvider

client_data = PickleDataProvider("../../../datasets/pickles/umdaa02_fd_filtered_cropped.pkl").collect()

print()