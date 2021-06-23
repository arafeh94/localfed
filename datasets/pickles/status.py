import json
import sys

sys.path.append("../../")

from src import tools
from src.data.data_provider import PickleDataProvider

url = sys.argv[1]
urls = json.load(open('urls.json', 'r'))
dc = PickleDataProvider(urls[url]).collect()
tools.detail({0: dc}, display=print)
