import json
import os

import src
from src import manifest
import logging

from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider

urls = json.load(open(manifest.DATA_PATH + "urls.json", 'r'))

logger = logging.getLogger('data_loader')


def pickle_distribute_continuous(dataset_name, clients_nb, min_samples, max_samples):
    file_path = manifest.DATA_PATH + str(
        dataset_name + "_" + clients_nb + "c_" + min_samples + "mn_" + max_samples + "mx.pkl")
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls[dataset_name])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_continuous(clients_nb, min_samples, max_samples)
        data_generator.save(file_path)
        return client_data


def pickle_distribute_shards(dataset_name, shards_nb, clients_nb, min_samples, max_samples):
    file_path = manifest.DATA_PATH + str(
        dataset_name + "_" + shards_nb + "shards_" + clients_nb + "c_" + min_samples + "mn_" + max_samples + "mx.pkl")
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls[dataset_name])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_shards(clients_nb, shards_nb, min_samples, max_samples)
        data_generator.save(file_path)
        return client_data


def pickle_distribute_size(dataset_name, clients_nb, min_samples, max_samples):
    file_path = manifest.DATA_PATH + str(
        dataset_name + "_" + clients_nb + "c_" + min_samples + "mn_" + max_samples + "mx.pkl")
    if os.path.exists(file_path):
        logger.info(f'distributed data file exists, loading from {file_path}...')
        return src.data.data_generator.load(file_path).get_distributed_data()
    else:
        logger.info(f'distributed data file does not exists, distributing into {file_path}...')
        data_provider = PickleDataProvider(urls[dataset_name])
        data_generator = DataGenerator(data_provider)
        client_data = data_generator.distribute_size(clients_nb, min_samples, max_samples)
        data_generator.save(file_path)
        return client_data
