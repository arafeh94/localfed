import json
import logging
import os
import pickle
import sys
from abc import abstractmethod
from zipfile_deflate64 import ZipFile

import wget

from src import manifest
from src.data.data_container import DataContainer
import libs.language_tools as lt
import validators
import urllib.parse

logger = logging.getLogger('data_provider')


class DataProvider:
    @abstractmethod
    def collect(self) -> DataContainer:
        pass


class PickleDataProvider(DataProvider):
    def __init__(self, file_path):
        self.uri = file_path

    def collect(self) -> DataContainer:
        self._handle_url()
        file = open(self.uri, 'rb')
        return pickle.load(file)

    @staticmethod
    def save(container, file_path):
        file = open(file_path, 'wb')
        pickle.dump(container, file)

    def _handle_url(self):
        if not validators.url(self.uri):
            return
        url_parts = urllib.parse.urlparse(self.uri)
        file_name = url_parts[2].rpartition('/')[-1]

        downloaded_file_path = manifest.DATA_PATH + file_name
        local_file = downloaded_file_path.replace('zip', 'pkl')
        if self._file_exists(local_file):
            logger.info(f'file exists locally, loading path {local_file}')
            self.uri = downloaded_file_path.replace('zip', 'pkl')
        else:
            downloaded = self._download(self.uri, downloaded_file_path)
            if downloaded:
                self.uri = local_file
            else:
                raise Exception('error while downloaded the file')

    def _bar_progress(self, current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    def _file_exists(self, downloaded_file):
        return os.path.isfile(downloaded_file)

    def _download(self, url, downloaded_file_path):
        try:
            logger.info(f'downloading file into {downloaded_file_path}')
            wget.download(url, downloaded_file_path, bar=self._bar_progress)
            logger.info('\nextracting...')
            with ZipFile(downloaded_file_path, 'r') as zipObj:
                zipObj.extractall(manifest.DATA_PATH)
            logger.info('loading...')
            return True
        except Exception as e:
            logger.info('error while downloading the file')
            raise e
        finally:
            if self._file_exists(downloaded_file_path):
                os.remove(downloaded_file_path)


class SQLDataProvider(DataProvider):
    def __init__(self, host, user, password, database, query, fetch_x_y: callable):
        import mysql.connector
        self.db = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.query = query
        self.fetcher = fetch_x_y

    def collect(self) -> DataContainer:
        cursor = self.db.cursor()
        cursor.execute(self.query)
        xs = []
        ys = []
        for row in cursor.fetchall():
            x, y = self.fetcher(row)
            xs.append(x)
            ys.append(y)
        return DataContainer(xs, ys)


class LocalMnistDataProvider(SQLDataProvider):
    def __init__(self, query=None, limit=0):
        super().__init__(
            host='localhost',
            password='root',
            user='root',
            database='mnist',
            query=query,
            fetch_x_y=lambda row: (json.loads(row[0]), row[1])
        )
        if query is None:
            self.query = 'select data,label from sample'
        if limit > 0:
            self.query += ' limit ' + str(limit)


class LocalKDDDataProvider(SQLDataProvider):
    def __init__(self, query=None, limit=0):
        super().__init__(
            host='localhost',
            password='root',
            user='root',
            database='kdd',
            query=query,
            fetch_x_y=lambda row: (json.loads(row[0]), int(row[1]))
        )
        if query is None:
            self.query = 'select data,label from sample'
        if limit > 0:
            self.query += ' limit ' + str(limit)


class LocalShakespeareDataProvider(SQLDataProvider):
    def __init__(self, query=None, limit=0):
        super().__init__(
            host='localhost',
            password='root',
            user='root',
            database='shakespeare',
            query=query,
            fetch_x_y=lambda row: (row[0], row[1])
        )
        if query is None:
            self.query = 'select data,label from sample'
        if limit > 0:
            self.query += ' limit ' + str(limit)

    def collect(self) -> DataContainer:
        collected = super().collect()
        new_x = []
        new_y = []
        for index in range(len(collected.x)):
            new_x.append(lt.word_to_indices(collected.x[index]))
            new_y.append(lt.letter_to_index(collected.y[index]))
        return DataContainer(new_x, new_y)
