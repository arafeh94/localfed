import os
import pickle


def save(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pickle.dump(obj, open(file_path, 'wb'))


def load(file_path):
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    return None


def append(obj, tag, file_path):
    old_map = load(file_path)
    if old_map is None:
        old_map = {}
    old_map[tag] = obj
    save(old_map, file_path)
