import json
from collections import namedtuple
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
DATA_PATH = ROOT_PATH + "/datasets/pickles/"
COMPARE_PATH = ROOT_PATH + "/compares/"
DEFAULT_ACC_PATH = COMPARE_PATH + "acc.plt"
__urls_datasets_links_ = None


class WandbAuth:
    key = '24db2a5612aaf7311dd29a5178f252a1c0a351a9'
    # localfed_ca01 localfed_ubuntu_test05
    project = 'localfed_ubuntu_test05'
    entity = 'mwazzeh'


def dataset_urls(dataset: str):
    global __urls_datasets_links_
    if __urls_datasets_links_ is None:
        __urls_datasets_links_ = json.load(open(DATA_PATH + "urls.json", 'r'))
    return __urls_datasets_links_[dataset]
