import importlib
import json
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
    def __init__(self, config):
        self.configs = config

    def get(self, key, absent_ok=True) -> typing.Any:
        if key not in self.configs:
            if absent_ok:
                return None
            raise Exception(f'key {key} does not exists')
        return self._create(self.configs[key])

    def _create(self, obj) -> typing.Any:
        if isinstance(obj, dict):
            if 'class_ref' in obj:
                class_ref = obj.pop('class_ref')
                return Clazz(class_ref).refer()
            if 'class' in obj:
                class_name = obj.pop('class')
                obj_params = {}
                for key, item in obj.items():
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
