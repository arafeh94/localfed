import json
import pickle
from abc import abstractmethod
import mysql.connector
from src.data.data_container import DataContainer
import libs.language_tools as lt


class DataProvider:
    @abstractmethod
    def collect(self) -> DataContainer:
        pass


class PickleDataProvider(DataProvider):
    def __init__(self, file_path):
        self.file_path = file_path

    def collect(self) -> DataContainer:
        file = open(self.file_path, 'rb')
        return pickle.load(file)

    @staticmethod
    def save(container, file_path):
        file = open(file_path, 'wb')
        pickle.dump(container, file)


class SQLDataProvider(DataProvider):
    def __init__(self, host, user, password, database, query, fetch_x_y: callable):
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
            fetch_x_y=lambda row: (row[0], row[1])
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
