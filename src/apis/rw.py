from src.apis.extensions import Serializable


class IODict(Serializable):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.cached = {}

    def read(self, key, default=None, absent_ok=False):
        self.load()
        if key in self.cached:
            return self.cached[key]
        if default is not None:
            return default
        if absent_ok:
            return None
        raise Exception('requested key does not exists')

    def write(self, key, obj, overwrite=True, raise_exception=False):
        self.load()
        if not overwrite and obj in self.cached:
            if raise_exception:
                raise Exception(f'object of key {key} cannot be overwritten')
            else:
                return False
        self.cached[key] = obj
        self.save()
        return True
