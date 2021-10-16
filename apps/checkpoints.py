import os.path
import pickle
import sys

from src import manifest

cp_file_path = f'{manifest.ROOT_PATH}/checkpoints.fed'
if not os.path.exists(cp_file_path):
    print('checkpoint file does not exists, no resumable federated')
    exit(0)

reader = open(cp_file_path, 'rb')
checkpoints = pickle.load(reader)
display_keys = True
print('checkpoints:')
for checkpoint in checkpoints:
    if display_keys:
        display_keys = False
        print('content: ', checkpoints[checkpoint].__dict__.keys())
    print('title:', checkpoint)

reader.close()
