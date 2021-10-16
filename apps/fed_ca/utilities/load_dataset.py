import json
import os
import pickle

import src
from src import manifest
import logging

from src.data.data_provider import PickleDataProvider

urls = json.load(open(manifest.DATA_PATH + "urls.json", 'r'))
logger = logging.getLogger('data_loader')


class LoadData:

    def __init__(self, dataset_name='mnist', shards_nb=0, clients_nb=10, min_samples=1000, max_samples=1000):
        """
        """
        self.dataset_name = dataset_name
        self.shards_nb = shards_nb
        self.clients_nb = clients_nb
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.file_path, self.filename = self.get_dataset_filename()

    def get_dataset_filename(self):
        file_path = manifest.DATA_PATH

        shards_keywords = ''
        if self.shards_nb != 0:
            shards_keywords = str(self.shards_nb) + "shards_"

        file_name = self.dataset_name + "_" + shards_keywords + str(self.clients_nb) + "c_" + str(
            self.min_samples) + "mn_" + str(
            self.max_samples) + "mx.pkl"
        dataset_filename = file_path + file_name
        return dataset_filename, file_name

    def is_path_exists(self):
        return os.path.exists(self.file_path)

    def set_logger(self):
        if self.is_path_exists:
            logger.info(f'distributed data file exists, loading from {self.file_path}...')
        else:
            logger.info(f'distributed data file does not exists, distributing into {self.file_path}...')

    def distribute_data(self):
        return pickle.load(self.file_path)



    def pickle_distribute_continuous(self):
        is_path_exists = self.is_path_exists()
        self.set_logger()
        if is_path_exists:
            return self.distribute_data()
        else:
            data_generator = self.get_data_generator()
            client_data = data_generator.distribute_continuous(self.clients_nb, self.min_samples, self.max_samples)
            data_generator.save(self.file_path)
            return client_data

    def pickle_distribute_shards(self):
        is_path_exists = self.is_path_exists()
        self.set_logger()
        if is_path_exists:
            return self.distribute_data()
        else:
            data_generator = self.get_data_generator()
            client_data = data_generator.distribute_shards(self.clients_nb, self.shards_nb, self.min_samples,
                                                           self.max_samples)
            data_generator.save(self.file_path)
            return client_data

    def pickle_distribute_size(self):
        is_path_exists = self.is_path_exists()
        self.set_logger()
        if is_path_exists:
            return self.distribute_data()
        else:
            data_generator = self.get_data_generator()
            client_data = data_generator.distribute_size(self.clients_nb, self.min_samples, self.max_samples)
            data_generator.save(self.file_path)
            return client_data
