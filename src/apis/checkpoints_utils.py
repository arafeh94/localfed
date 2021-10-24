import os
import pickle
from datetime import datetime

from src import manifest
from src.manifest import CHECKPOINTS_PATH

cp_file_path = CHECKPOINTS_PATH


def read(file_path=cp_file_path):
    reader = open(file_path, 'rb')
    checkpoints = pickle.load(reader)
    reader.close()
    return checkpoints


def write(checkpoints, file_path=cp_file_path):
    writer = open(file_path, 'wb')
    pickle.dump(checkpoints, writer)
    writer.close()


def delete(checkpoints, file_path=cp_file_path):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    cps = read(file_path)
    deleted = []
    for checkpoint in checkpoints:
        if checkpoint in cps:
            deleted.append(checkpoint)
            del cps[checkpoint]
        else:
            print(checkpoint, 'does not exists')
    write(cps, file_path)
    return deleted


def merge(other_path, current_path=manifest.CHECKPOINTS_PATH):
    others: dict = read(other_path)
    current: dict = read(current_path)
    combined = {**current, **others}
    write(combined, current_path)
    return combined


def detail(file_path=cp_file_path, extended=True, display_keys=True):
    checkpoints = read(file_path)
    print('checkpoints:')
    for index, checkpoint in enumerate(checkpoints):
        if display_keys:
            display_keys = False
            print('content: ', checkpoints[checkpoint].__dict__.keys())
        print(f'{index}.', checkpoint, f'@{datetime.fromtimestamp(checkpoints[checkpoint].timestamp)}')
        if extended:
            for k, v in checkpoints[checkpoint].__dict__.items():
                print(f'\t{k}:', ((str(v).replace('\n', '').replace(' ', ' ')[0:55]) + '...') if len(
                    str(v)) >= 55 else str(v))
