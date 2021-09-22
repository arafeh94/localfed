import json
from collections import namedtuple
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
DATA_PATH = ROOT_PATH + "/datasets/pickles/"
COMPARE_PATH = ROOT_PATH + "/compares/"
DEFAULT_ACC_PATH = COMPARE_PATH + "acc.pkl"
DEFAULT_DIV_PATH = COMPARE_PATH + "div.pkl"
__urls_datasets_links_ = None


class WandbAuth:
    key = '3c35c1f04ebc7ffc1825f3056a6aabe714a1ccdc'
    project = 'geneticfed'
    entity = 'arafeh'


def dataset_urls(dataset: str):
    global __urls_datasets_links_
    if __urls_datasets_links_ is None:
        __urls_datasets_links_ = json.load(open(DATA_PATH + "urls.json", 'r'))
    return __urls_datasets_links_[dataset]
