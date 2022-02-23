import datetime
import time

from src.apis.utils import factors
from src.data.data_container import DataContainer
from src.data.data_loader import preload
import numpy as np


def transformer(dct):
    def client_transformer(client_id, data_container: DataContainer):
        dc = data_container.filter(lambda x, y: (y in [0, 1, 3, 4, 5, 8, 15]))
        dc = dc[0:len(dc) - len(dc) % 100]
        dc = dc.map(lambda x, y: (x, client_id - 1))
        dc = dc.reshape((-1, 100, 11))
        dc.y = dc.y[0:len(dc.x)]
        return dc

    return dct.map(client_transformer)


client_data = preload('fall_ar_by_client', tag='fall_013458_15_2d_100', transformer=transformer)

# client_data = preload('fall_ar_by_client', tag='fall_013458_15_2',
#                       transformer=lambda dct: dct.map(
#                           lambda client_id, dc:
#                           dc.filter(lambda x, y: (y in [0, 1, 3, 4, 5, 8, 15])).map(
#                               lambda x, y: (x, client_id - 1)).reshap(11, )
#                       ))
print()

vb = preload('fall_013458_15')

print()
