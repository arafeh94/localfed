import json
from collections import namedtuple
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
DATA_PATH = ROOT_PATH + "/datasets/pickles/"
COMPARE_PATH = ROOT_PATH + "/compares/"
CHECKPOINTS_PATH = f'{ROOT_PATH}/checkpoints.fed'
DEFAULT_ACC_PATH = COMPARE_PATH + "acc.pkl"
DEFAULT_DIV_PATH = COMPARE_PATH + "div.pkl"
DB_PATH = COMPARE_PATH + "perf.db"
__urls_datasets_links_ = None

wandb_config = {
    'key': '24db2a5612aaf7311dd29a5178f252a1c0a351a9',
    # localfed_ca01 localfed_ubuntu_test05 umdaa-02-fd
    'project': 'umdaa-02-fd-filtered-cropped',
    'entity': 'mwazzeh',
}

datasets_urls = {
    "mnist10k": "https://www.dropbox.com/s/5zhudqpupg061of/mnist10k.zip?dl=1",
    "mnist": "https://www.dropbox.com/s/ohj5f4feshpvfej/mnist.zip?dl=1",
    "femnist": "https://www.dropbox.com/s/v3khohsd1g3pfyx/femnist.zip?dl=1",
    "kdd": "https://www.dropbox.com/s/thvyd30nbrd47qi/kdd.zip?dl=1",
    "kdd_train": "https://www.dropbox.com/s/ys82zxpb4fvv44p/kdd_train.zip?dl=1",
    "kdd_test": "https://www.dropbox.com/s/2ucs7owcl5wsbak/kdd_test.zip?dl=1",
    "fekdd_test": "https://www.dropbox.com/s/ijqxm1x5sy1us5n/fekdd_test.zip?dl=1",
    "fekdd_train": "https://www.dropbox.com/s/su1uucnd3z2072z/fekdd_train.zip?dl=1",
    "signs": "https://www.dropbox.com/s/ni85ukowhs9ghkb/signs.zip?dl=1",
    "cifar10": "https://www.dropbox.com/s/2x8176jyrs6ydqi/cifar10.zip?dl=1",
    "fall_by_client": "https://www.dropbox.com/s/y7glz4pnzflbha4/fall_by_client.zip?dl=1",
    "fall_ar_by_client": "https://www.dropbox.com/s/txo6e1noq6gbrvz/fall_ar_by_client.zip?dl=1",
    "mnist10k_mr1": "https://mega.nz/file/nxE32SDA#aKxvOQ_Aq2ypFpO8pNJdpte8ScuxjD8fNbdTCxfclbk",
    "umdaa02touch": "https://mega.nz/file/6s9EnbJB#MbffkyAUW2PAvu7xqy8-Q0GSOQU-auoVlvBrw1fKI4o"
}
