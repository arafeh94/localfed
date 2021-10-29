import atexit
import logging
import sqlite3

from src import manifest
from src.app import session
from src.app.federated_app import FederatedApp
from src.app.session import Session
from src.federated.events import FederatedEventPlug
from src.federated.federated import FederatedLearning
from src.manifest import WandbAuth


# noinspection SqlNoDataSourceInspection
class SQLiteLogger(FederatedEventPlug):
    def __init__(self, id, db_path=manifest.DB_PATH):
        super().__init__()
        self.id = id
        self.con = sqlite3.connect(db_path)
        self.check_table_creation = True
        self.logger = logging.getLogger('sqlite')

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
        sub_query = ''
        for param, value in params.items():
            if not isinstance(value, (int, float)):
                value = f'"{value}"'
            sub_query += f'{value},'
        sub_query = sub_query.rstrip(',')
        query = f'insert OR IGNORE into {self.id} values ({sub_query})'
        self._execute(query)

    def _execute(self, query):
        cursor = self.con.cursor()
        self.logger.debug(f'executing {query}')
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
