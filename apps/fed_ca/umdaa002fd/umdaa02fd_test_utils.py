import logging
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from src import tools
from src.data.data_distributor import UniqueDistributor
from src.data.data_provider import PickleDataProvider
import apps.fed_ca.utilities.utils as utils

start_time = datetime.now()
print(start_time)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

# dataset = 'umdaa002fd'
# total number of clients from umdaa02-fd is 44
# labels_number = 10
# ud = UniqueDistributor(labels_number, 10, 10)

client_data = PickleDataProvider("../../../datasets/pickles/umdaa002fd_unique_10c_500mn_500mx_central.pkl").collect()
# tools.detail(client_data)


# client_data = ud.distribute(client_data)

# utils.tensor_to_image(client_data[0].x[0]).show()

# pixels = client_data[0].x[0]
# plt.imshow(np.reshape(pixels, (128, 128, 3)))
# plt.show()

for item in range(len(client_data)):
    img = np.reshape(client_data[0].x[item], (128, 128, 3))
    # make it within this range [0:1] if it is between [-1, 1] after calling the amax() and amin() then
    img = img / 2 + 0.5
    plt.imshow(img)
    plt.show()

# tools.detail(client_data)

print('end')
