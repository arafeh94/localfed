import copy
import importlib
import json
import logging
import typing


class Clazz:
    def __init__(self, clz_str: str):
        self.module_name = clz_str[0:clz_str.rfind('.')]
        self.class_name = clz_str[clz_str.rfind('.') + 1:len(clz_str)]

    def create(self, params: dict = None):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        instance = class_(**params) if params else class_(**params)
        return instance

    def refer(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        return class_

    @staticmethod
    def is_class(val: str):
        try:
            clz = Clazz(val)
            clz.create()
            return True
        except:
            return False


class Settings:
    MULTIVALUES_KEYS = ['epochs', 'lr', 'batch_size', 'client_ratio', 'optimizer', 'criterion', 'distributor',
                        'dataset', 'model']
    UNSUPPORTED_MULTIVALUES = ['rounds']

    def __init__(self, config):
        self.configs = self._init(config)
        self.cursor = 0
        self.zero_stop = True

    def __iter__(self):
        self.cursor = 0
        self.zero_stop = True
        return self

    def __next__(self):
        if self.zero_stop:
            self.zero_stop = False
            return self

        self.cursor = self.cursor + 1
        if self.cursor >= len(self.configs):
            raise StopIteration
        return self

    def get_config(self):
        return self.configs[self.cursor]

    def set_cursor(self, cursor: int):
        self.cursor = cursor

    def _init(self, config):
        if not isinstance(config, list):
            config = [config]
        all_configs = []
        for part in config:
            all_configs.extend(self._decompile(part))
        return all_configs

    def _decompile(self, config):
        immutable_multi_configs = [{}]
        for key in config:
            if key in Settings.UNSUPPORTED_MULTIVALUES and isinstance(config[key], list):
                raise Exception(f'config [{key}] does not support multi_key, make sure it\'s not a list')
            if key in Settings.MULTIVALUES_KEYS and isinstance(config[key], list):
                num = len(config[key])
                multi_configs = []
                for cf in immutable_multi_configs:
                    copies = [copy.deepcopy(cf) for i in range(num)]
                    for index, cp in enumerate(copies):
                        value = config[key][index]
                        cp[key] = value
                        multi_configs.append(cp)
                immutable_multi_configs = copy.deepcopy(multi_configs)
            else:
                for cf in immutable_multi_configs:
                    cf[key] = config[key]

        return immutable_multi_configs

    def get(self, key, absent_ok=True) -> typing.Any:
        if key not in self.configs[self.cursor]:
            if absent_ok:
                return None
            raise Exception(f'key {key} does not exists')
        return self._create(self.configs[self.cursor][key])

    def __len__(self):
        return len(self.configs)

    def _create(self, obj) -> typing.Any:
        if isinstance(obj, dict):
            if 'class_ref' in obj:
                class_ref = obj.get('class_ref')
                return Clazz(class_ref).refer()
            if 'class' in obj:
                class_name = obj.get('class')
                obj_params = {}
                for key, item in obj.items():
                    if key != 'class':
                        obj_params[key] = self._create(item)
                return Clazz(class_name).create(obj_params)
        if isinstance(obj, list):
            initializations = []
            for o in obj:
                initialized = self._create(obj)
                initializations.append(initialized)
            return initializations
        return obj

    @staticmethod
    def from_json_file(file_path):
        configs = json.load(open(file_path, 'r'))
        return Settings(configs)
