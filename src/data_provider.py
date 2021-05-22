import json
import pickle
from abc import abstractmethod
import mysql.connector
from src.data_container import DataContainer


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
            fetch_x_y=lambda row: (json.loads(row[0]), row[1])
        )
        if query is None:
            self.query = 'select data,label from sample'
        if limit > 0:
            self.query += ' limit ' + str(limit)
