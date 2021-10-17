import os
import pickle
from datetime import datetime

from src.manifest import CHECKPOINTS_PATH

cp_file_path = CHECKPOINTS_PATH
if not os.path.exists(cp_file_path):
    raise Exception('checkpoint file does not exists, no resumable federated')


def read():
    reader = open(cp_file_path, 'rb')
    checkpoints = pickle.load(reader)
    reader.close()
    return checkpoints


def write(checkpoints):
    writer = open(cp_file_path, 'wb')
    pickle.dump(checkpoints, writer)
    writer.close()


def delete(checkpoints):
    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]
    cps = read()
    deleted = []
    for checkpoint in checkpoints:
        if checkpoint in cps:
            deleted.append(checkpoint)
            del cps[checkpoint]
        else:
            print(checkpoint, 'does not exists')
    write(cps)
    return deleted


def detail(extended=True, display_keys=True):
    checkpoints = read()
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
