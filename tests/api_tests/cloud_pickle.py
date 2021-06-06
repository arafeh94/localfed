import logging

from src.data.data_provider import PickleDataProvider
logging.basicConfig(level=logging.INFO)

data = PickleDataProvider('https://www.dropbox.com/s/nd25svv30chttln/mnist.zip?dl=1').collect()
print(data.x)
