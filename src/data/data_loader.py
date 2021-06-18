import json
import os

import src
from src import manifest
import logging

from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider

urls = json.load(open(manifest.DATA_PATH + "urls.json", 'r'))

logger = logging.getLogger('data_loader')


def mnist_10shards_100c_400min_400max():
    file_path = manifest.DATA_PATH + "mnist_10shards_100c_400mn_400mx.pkl"
    if os.path.exists(file_path):
        logger.info('distributed data file exists, loading...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info('distributed data file does not exists, distributing...')
        data_provider = PickleDataProvider(urls['mnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(100, 10, 400, 400)
        data_generator.save(file_path)
        return client_data


def mnist_2shards_100c_600min_600max():
    file_path = manifest.DATA_PATH + "mnist_2shards_100c_600mn_600mx.pkl"
    if os.path.exists(file_path):
        logger.info('distributed data file exists, loading...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info('distributed data file does not exists, distributing...')
        data_provider = PickleDataProvider(urls['mnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(100, 2, 600, 600)
        data_generator.save(file_path)
        return client_data


def mnist_1shards_100c_600min_600max():
    file_path = manifest.DATA_PATH + "mnist_1shards_100c_600mn_600mx.pkl"
    if os.path.exists(file_path):
        logger.info('distributed data file exists, loading...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info('distributed data file does not exists, distributing...')
        data_provider = PickleDataProvider(urls['mnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(100, 1, 600, 600)
        data_generator.save(file_path)
        return client_data


def femnist_2shards_100c_600min_600max():
    file_path = manifest.DATA_PATH + "femnist_2shards_100c_600mn_600mx.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(100, 2, 600, 600)
        data_generator.save(file_path)
        return client_data


def kdd_100c_400min_400max():
    file_path = manifest.DATA_PATH + "kdd_100c_400min_400max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['kdd'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_size(100, 400, 400)
        data_generator.save(file_path)
        return client_data


def femnist_1shard_31c_1000min_1000max():
    file_path = manifest.DATA_PATH + "femnist_1s_31c_1000min_1000max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_continuous(31, 1000, 1000)
        data_generator.save(file_path)
        return client_data


def femnist_1s_5c_2000min_2000max():
    file_path = manifest.DATA_PATH + "femnist_1s_5c_2000min_2000max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_continuous(5, 2000, 2000)
        data_generator.save(file_path)
        return client_data


def femnist_1shard_62c_200min_2000max():
    file_path = manifest.DATA_PATH + "femnist_1s_62c_200min_2000max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_continuous(62, 200, 2000)
        data_generator.save(file_path)
        return client_data

def femnist_1shard_10c_1000min_1000max():
    file_path = manifest.DATA_PATH + "femnist_1shard_10c_1000min_1000max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_continuous(10, 1000, 1000)
        data_generator.save(file_path)
        return client_data


def femnist_10shards_100c_600min_600max():
    file_path = manifest.DATA_PATH + "femnist_10shards_100c_600min_600max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(100, 10, 600, 600)
        data_generator.save(file_path)
        return client_data


def femnist_62shards_62c_600min_600max():
    file_path = manifest.DATA_PATH + "femnist_62shards_62c_600min_600max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(62, 62, 600, 600)
        data_generator.save(file_path)
        return client_data


def femnist_62shards_62c_2000min_2000max():
    file_path = manifest.DATA_PATH + "femnist_62shards_62c_2000min_2000max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(62, 62, 2000, 2000)
        data_generator.save(file_path)
        return client_data


def femnist_1s_62c_2000min_2000max():
    file_path = manifest.DATA_PATH + "femnist_1s_62c_2000min_2000max.pkl"
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls['femnist'])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_continuous(62, 2000, 2000)
        data_generator.save(file_path)
        return client_data

