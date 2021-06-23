import json
import os
import typing

import src
from src import manifest
import logging

from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider

urls = json.load(open(manifest.DATA_PATH + "urls.json", 'r'))

logger = logging.getLogger('data_loader')


def preload(name, dataset, distributor: typing.Callable[[DataGenerator], None]):
    file_path = manifest.DATA_PATH + name + ".pkl"
    if os.path.exists(file_path):
        logger.info('distributed data file exists, loading...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info('distributed data file does not exists, distributing...')
        data_provider = PickleDataProvider(urls[dataset])
        data_generator = DataGenerator(data_provider)
        client_data = distributor(data_generator)
        data_generator.save(file_path)
        return client_data


def mnist_10shards_100c_400min_400max():
    return preload('mnist_10shards_100c_400min_400max', 'mnist', lambda dg: dg.distribute_shards(100, 10, 400, 400))


def mnist_2shards_100c_600min_600max():
    return preload('mnist_2shards_100c_600min_600max', 'mnist', lambda dg: dg.distribute_shards(100, 2, 600, 600))


def mnist_1shards_100c_600min_600max():
    return preload('mnist_1shards_100c_600min_600max', 'mnist', lambda dg: dg.distribute_shards(100, 1, 600, 600))


def femnist_2shards_100c_600min_600max():
    return preload('femnist_2shards_100c_600min_600max', 'femnist', lambda dg: dg.distribute_shards(100, 2, 600, 600))


def femnist_100c_2000min_2000max():
    return preload('femnist_100c_2000min_2000max', 'femnist', lambda dg: dg.distribute_size(100, 2000, 2000))


def femnist_2shards_100c_2000min_2000max():
    return preload('femnist_2shards_100c_2000min_2000max', 'femnist',
                   lambda dg: dg.distribute_shards(100, 2, 2000, 2000))


def kdd_100c_400min_400max():
    return preload('kdd_100c_400min_400max', 'kdd', lambda dg: dg.distribute_size(100, 400, 400))


def femnist_1shard_62c_2000min_2000max():
    return preload('femnist_1shard_62c_2000min_2000max', 'femnist', lambda dg: dg.distribute_continuous(62, 2000, 2000))


def femnist_1shard_62c_200min_2000max():
    return preload('femnist_1shard_62c_200min_2000max', 'femnist', lambda dg: dg.distribute_continuous(62, 200, 2000))
