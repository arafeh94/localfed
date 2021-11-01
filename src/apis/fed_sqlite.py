import sqlite3


class FedDB:
    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)

    def execute(self, query, params=None):
        cur = self.con.cursor()
        if params:
            return cur.execute(query, params)
        else:
            return cur.execute(query)

    def get(self, table_name, field):
        query = f'select {field} from {table_name}'
        records = self.execute(query)
        values = list(map(lambda x: x[0], records))
        return values

    def acc(self, table_name):
        query = f'select acc from {table_name}'
        records = self.execute(query)
        acc = list(map(lambda x: x[0], records))
        return acc

    def tables(self):
        query = 'select * from session'
        records = self.execute(query)
        records = list(map(lambda x: {x[0]: x[1]}, records))
        tables = {}
        for r in records:
            tables.update(r)
        return tables
