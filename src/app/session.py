import copy
import hashlib
import inspect
import os.path
import random
from datetime import date

from src.app.cache import Cache
from src.app.settings import Settings


class Session:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: Cache = settings.get('cache')
        self._inspect_cache()
        self._inspect_id()

    def write(self, key, obj):
        self.cache.write(key, obj)

    def read(self, key):
        return self.cache.read(key, absent_ok=True)

    def _hash(self):
        config_copy = copy.deepcopy(self.settings.configs)
        if 'rounds' in config_copy:
            del config_copy['rounds']
        full_hash = str.encode(str(config_copy))
        hashed = hashlib.md5(full_hash).hexdigest()
        return hashed

    def _generate_id(self):
        return f'session_{random.randint(0, 99999)}'

    def session_id(self, reset=False):
        if reset or not self.cache.read('session_id', absent_ok=True):
            self.cache.write('session_id', self._generate_id())
            self.cache.write('created_at', f'{date.today()}')
        return self.cache.read('session_id')

    def _inspect_id(self):
        new_hash = self._hash()
        old_hash = self.cache.read('hash', absent_ok=True)
        if old_hash:
            if old_hash != new_hash:
                self.session_id(True)
                self.write('hash', new_hash)
        else:
            self.write('hash', new_hash)

    def _inspect_cache(self):
        if self.cache.file_path is None:
            self.cache.file_path = f'./cache/{self._hash()}'
            self.cache.write('updated_at', f'{date.today()}')
