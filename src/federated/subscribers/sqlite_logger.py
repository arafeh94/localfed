import logging
import re
import sqlite3

from src import manifest
from src.federated.events import FederatedEventPlug
from src.federated.federated import FederatedLearning


# noinspection SqlNoDataSourceInspection
class SQLiteLogger(FederatedEventPlug):
    def __init__(self, id, db_path=manifest.DB_PATH, tag=''):
        super().__init__()
        self.id = id
        self.con = sqlite3.connect(db_path)
        self.check_table_creation = True
        self.logger = logging.getLogger('sqlite')
        self.tag = str(tag)

    def on_federated_started(self, params):
        query = 'create table if not exists session (session_id text primary key, config text)'
        self._execute(query)
        query = f"insert or ignore into session values (?,?)"
        self._execute(query, [self.id, self.tag])

    def _create_table(self, params):
        if self.check_table_creation:
            self.check_table_creation = False
            sub_query = ''
            for param in params:
                sub_query += f'{param[0]} {param[1]},'
            sub_query = sub_query.rstrip(',')
            query = f'''
            create table if not exists {self.id}(
                {sub_query}
            )
            '''
            self._execute(query)

    def _insert(self, params):
        sub_query = ' '.join(['?,' for _ in range(len(params))]).rstrip(',')
        query = f'insert OR IGNORE into {self.id} values ({sub_query})'
        values = list(map(lambda v: str(v) if isinstance(v, (list, dict)) else v, params.values()))
        self._execute(query, values)

    def _execute(self, query, params=None):
        cursor = self.con.cursor()
        self.logger.debug(f'executing {query}')
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.con.commit()

    def _extract_params(self, history):
        def param_map(val):
            if isinstance(val, int):
                return 'INTEGER'
            elif isinstance(val, str):
                return 'TEXT'
            elif isinstance(val, float):
                return 'FLOAT'
            else:
                return 'text'

        params = [('round_id', 'INTEGER PRIMARY KEY')]
        for key, val in history[list(history.keys())[0]].items():
            params.append((key, param_map(val)))
        return params

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        history: dict = context.history
        history_columns = self._extract_params(history)
        self._create_table(history_columns)
        last_inserted = {'round_id': context.round_id, **history[list(history.keys())[-1]]}
        self._insert(last_inserted)
