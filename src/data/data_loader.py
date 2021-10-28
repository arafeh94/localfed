import json
import os
import pickle
import typing

import src
from src import manifest
import logging

from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.data.data_distributor import Distributor, LabelDistributor, SizeDistributor, UniqueDistributor
from src.data.data_provider import PickleDataProvider

logger = logging.getLogger('data_loader')


def preload(dataset, distributor: Distributor, tag=None) -> typing.Union[Dict[int, DataContainer], DataContainer]:
    """
    Args:
        tag: file name without postfix (file type, auto-filled with .pkl)
        dataset: dataset used, should be exists inside urls
        distributor: distribution function, dg.distribute_shards or dg.distribute_size ...

    Returns: clients data of type typing.Dict[int, DataContainer]

    """
    tag = tag or dataset + '_' + distributor.id() if distributor else ''
    file_path = manifest.DATA_PATH + tag + ".pkl"
    logger.info(f'searching for {file_path}...')
    if os.path.exists(file_path):
        logger.info('distributed data file exists, loading...')
        data = pickle.load(open(file_path, 'rb'))
        logger.info(data)
        return data
    else:
        logger.info('distributed data file does not exists, distributing...')
        data = PickleDataProvider(manifest.dataset_urls(dataset)).collect()
        if distributor:
            data = distributor.distribute(data)
        pickle.dump(data, open(file_path, 'wb'))
        return data


def mnist_10shards_100c_400min_400max() -> Dict[int, DataContainer]:
    return preload('mnist', LabelDistributor(100, 10, 400, 400))


def cifar10_10shards_100c_400min_400max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 10, 400, 400))


def mnist_2shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('mnist', LabelDistributor(100, 2, 600, 600))


def cifar10_2shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 2, 600, 600))


def cifar10_1shard_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 1, 600, 600))


def mnist_1shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('mnist', LabelDistributor(100, 1, 600, 600))


def femnist_2shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('femnist', LabelDistributor(100, 2, 600, 600))


def femnist_100c_2000min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', SizeDistributor(100, 2000, 2000))


def femnist_2shards_100c_2000min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', LabelDistributor(100, 2, 2000, 2000))


def kdd_100c_400min_400max() -> Dict[int, DataContainer]:
    return preload('kdd', SizeDistributor(100, 400, 400))


def femnist_1shard_62c_2000min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', UniqueDistributor(62, 2000, 2000))


def femnist_1shard_62c_200min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', UniqueDistributor(62, 200, 2000))

def cifar10_10c_6000min_6000max() -> Dict[int, DataContainer]:
    return preload('cifar10', UniqueDistributor(10, 6000, 6000))
