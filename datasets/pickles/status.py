import json

from src import tools
from src.data.data_provider import PickleDataProvider

urls = json.load(open('urls.json', 'r'))
dc = PickleDataProvider(urls['fekdd_train']).collect()
dc1 = PickleDataProvider(urls['fekdd_test']).collect()
dc1 = dc1.map(lambda x, y: (x, y if y == 0 else 1))
dc1 = dc1.filter(lambda x, y: True if y == 1 else False)
tools.detail({0: dc, 1: dc1}, display=print)
